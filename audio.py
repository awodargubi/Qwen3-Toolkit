import os
import io
import librosa
import subprocess
import numpy as np
import soundfile as sf
import requests
import base64
import time
from typing import Optional, Union

from silero_vad import get_speech_timestamps


WAV_SAMPLE_RATE = 16000


def load_audio(file_path: Union[str, bytes]) -> np.ndarray:
    # Handle URL downloads
    if isinstance(file_path, str) and (file_path.startswith('http://') or file_path.startswith('https://')):
        print(f"Downloading audio from URL: {file_path}")
        response = requests.get(file_path)
        response.raise_for_status()
        file_data = io.BytesIO(response.content)
        try:
            wav_data, _ = librosa.load(file_data, sr=WAV_SAMPLE_RATE, mono=True)
            return wav_data
        except Exception as e:
            # Fallback to soundfile for URL content
            file_data.seek(0)
            wav_data, sr = sf.read(file_data, dtype='float32')
            if sr != WAV_SAMPLE_RATE:
                wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=WAV_SAMPLE_RATE)
            return wav_data

    # Handle bytes data
    if isinstance(file_path, bytes):
        file_data = io.BytesIO(file_path)
        try:
            wav_data, _ = librosa.load(file_data, sr=WAV_SAMPLE_RATE, mono=True)
            return wav_data
        except Exception:
            file_data.seek(0)
            wav_data, sr = sf.read(file_data, dtype='float32')
            if sr != WAV_SAMPLE_RATE:
                wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=WAV_SAMPLE_RATE)
            return wav_data

    # Handle local file paths
    try:
        # Try librosa first, because it is usually faster for standard formats.
        wav_data, _ = librosa.load(file_path, sr=WAV_SAMPLE_RATE, mono=True)
        return wav_data
    except Exception as e:
        print(e)
        # After librosa fails, use a more powerful ffmpeg as a backup.
        try:
            command = [
                'ffmpeg',
                '-i', file_path,
                '-ar', str(WAV_SAMPLE_RATE),
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-f', 'wav',
                '-'
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout_data, stderr_data = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg error processing local file: {stderr_data.decode('utf-8', errors='ignore')}")

            with io.BytesIO(stdout_data) as data_io:
                wav_data, sr = sf.read(data_io, dtype='float32')

            return wav_data
        except Exception as ffmpeg_e:
            raise RuntimeError(
                f"Failed to load audio from local file '{file_path}' even with ffmpeg. Error: {ffmpeg_e}")


def process_vad(wav: np.ndarray, worker_vad_model, segment_threshold_s: int = 120, max_segment_threshold_s: int = 180) -> list[np.ndarray]:
    try:
        vad_params = {
            'sampling_rate': WAV_SAMPLE_RATE,
            'return_seconds': False,
            'min_speech_duration_ms': 1500,
            'min_silence_duration_ms': 500
        }

        speech_timestamps = get_speech_timestamps(
            wav,
            worker_vad_model,
            **vad_params
        )

        if not speech_timestamps:
            raise ValueError("No speech segments detected by VAD.")

        potential_split_points_s = {0.0, len(wav)}
        for i in range(len(speech_timestamps)):
            start_of_next_s = speech_timestamps[i]['start']
            potential_split_points_s.add(start_of_next_s)
        sorted_potential_splits = sorted(list(potential_split_points_s))

        final_split_points_s = {0.0, len(wav)}
        segment_threshold_samples = segment_threshold_s * WAV_SAMPLE_RATE
        target_time = segment_threshold_samples
        while target_time < len(wav):
            closest_point = min(sorted_potential_splits,
                                key=lambda p: abs(p - target_time))
            final_split_points_s.add(closest_point)
            target_time += segment_threshold_samples
        final_ordered_splits = sorted(list(final_split_points_s))

        max_segment_threshold_samples = max_segment_threshold_s * WAV_SAMPLE_RATE
        new_split_points = [0.0]

        # Make sure that each audio segment does not exceed max_segment_threshold_s
        for i in range(1, len(final_ordered_splits)):
            start = final_ordered_splits[i - 1]
            end = final_ordered_splits[i]
            segment_length = end - start

            if segment_length <= max_segment_threshold_samples:
                new_split_points.append(end)
            else:
                num_subsegments = int(
                    np.ceil(segment_length / max_segment_threshold_samples))
                subsegment_length = segment_length / num_subsegments

                for j in range(1, num_subsegments):
                    split_point = start + j * subsegment_length
                    new_split_points.append(split_point)

                new_split_points.append(end)

        segmented_wavs = []
        for i in range(len(new_split_points) - 1):
            start_sample = int(new_split_points[i])
            end_sample = int(new_split_points[i + 1])
            segmented_wavs.append(wav[start_sample:end_sample])
        return segmented_wavs

    except Exception as e:
        segmented_wavs = []
        total_samples = len(wav)
        max_chunk_size_samples = max_segment_threshold_s * WAV_SAMPLE_RATE

        for start_sample in range(0, total_samples, max_chunk_size_samples):
            end_sample = min(
                start_sample + max_chunk_size_samples, total_samples)
            segment = wav[start_sample:end_sample]
            if len(segment) > 0:
                segmented_wavs.append(segment)

        return segmented_wavs


def save_audio_file(wav: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, wav, WAV_SAMPLE_RATE)


def separate_vocals_with_demucs(audio_file_path: str, replicate_token: str) -> str:
    """
    Separate vocals from audio using Replicate's demucs model.
    Returns URL to the separated vocals file.
    """
    import replicate

    # Set up Replicate client
    client = replicate.Client(api_token=replicate_token)

    print("Starting vocal separation with demucs...")

    # Run the demucs model with specific version hash
    output = client.run(
        "ryan5453/demucs:5a7041cc9b82e5a558fea6b3d7b12dea89625e89da33f0447bd727c2d0ab9e77",
        input={
            "audio": open(audio_file_path, "rb"),
            "model": "htdemucs",  # Use htdemucs model
            "stem": "vocals",     # Extract vocals only
            "output_format": "mp3"
        }
    )

    print("Vocal separation completed!")

    # Handle different output formats
    if isinstance(output, dict) and 'vocals' in output:
        # When stem="vocals" is specified, output contains vocals FileOutput object
        vocals_file_output = output['vocals']
        vocals_url = str(vocals_file_output)  # Convert FileOutput to URL string
        print(f"Vocals URL: {vocals_url}")
        return vocals_url
    elif isinstance(output, str):
        # Direct URL output
        vocals_url = output
        print(f"Vocals URL: {vocals_url}")
        return vocals_url
    else:
        raise RuntimeError(f"Unexpected output format from demucs: {output}")


def download_audio_from_url(url: str, output_path: str) -> str:
    """
    Download audio file from URL and save to local path.
    Returns the local file path.
    """
    print(f"Downloading vocals from: {url}")

    response = requests.get(url)
    response.raise_for_status()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(response.content)

    print(f"Vocals saved to: {output_path}")
    return output_path


def process_audio_with_vocal_separation(
    audio_file_path: str,
    vad_model,
    replicate_token: str,
    extract_vocals: bool = True,
    vocals_output_path: Optional[str] = None,
    segment_threshold_s: int = 120,
    max_segment_threshold_s: int = 180
) -> tuple[list[np.ndarray], Optional[str]]:
    """
    Process audio with optional vocal separation using demucs, then apply VAD.

    Args:
        audio_file_path: Path to input audio file
        vad_model: Loaded VAD model
        replicate_token: Replicate API token
        extract_vocals: Whether to separate vocals first
        vocals_output_path: Where to save separated vocals (optional)
        segment_threshold_s: Target segment length in seconds
        max_segment_threshold_s: Maximum segment length in seconds

    Returns:
        tuple: (list of audio segments, path to vocals file if extracted)
    """
    vocals_file_path = None

    if extract_vocals:
        print("Step 1: Separating vocals using demucs...")

        # Separate vocals using Replicate demucs
        vocals_url = separate_vocals_with_demucs(audio_file_path, replicate_token)

        # Set output path for vocals
        if vocals_output_path is None:
            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
            vocals_output_path = f"C:\\Github\\Test toolkit gemini\\{base_name}_vocals.mp3"

        # Download the separated vocals
        vocals_file_path = download_audio_from_url(vocals_url, vocals_output_path)

        print("Step 2: Loading separated vocals...")
        # Load the separated vocals
        wav_data = load_audio(vocals_file_path)
    else:
        print("Step 1: Loading original audio (skipping vocal separation)...")
        # Load original audio without separation
        wav_data = load_audio(audio_file_path)

    print("Step 3: Applying VAD and segmentation...")
    # Apply VAD processing
    segments = process_vad(
        wav_data,
        vad_model,
        segment_threshold_s=segment_threshold_s,
        max_segment_threshold_s=max_segment_threshold_s
    )

    print(f"Processing complete! Found {len(segments)} vocal segments.")
    return segments, vocals_file_path
