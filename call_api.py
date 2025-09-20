import os
import argparse
import tempfile
import concurrent.futures
from typing import List, Tuple
from collections import Counter
from tqdm import tqdm

# Import your audio processor
from audio import process_audio_with_vocal_separation, save_audio_file, load_audio
from silero_vad import load_silero_vad

# Import Qwen ASR (you'll need to install this)
# pip install qwen3-asr-toolkit
from qwen3_asr_toolkit.qwen3asr import QwenASR
import dashscope


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete audio processing pipeline: Demucs vocal separation + VAD segmentation + Qwen3-ASR transcription"
    )
    parser.add_argument("--input-file", '-i', type=str, required=True,
                        help="Input audio/video file path")
    parser.add_argument("--context", '-c', type=str, default="",
                        help="Context for Qwen3-ASR (topic, names, etc.)")
    parser.add_argument("--dashscope-api-key", type=str,
                        help="DashScope API key for Qwen3-ASR")
    parser.add_argument("--replicate-token", type=str, required=True,
                        help="Replicate API token for Demucs")
    parser.add_argument("--extract-vocals", action="store_true",
                        help="Extract vocals using Demucs before transcription")
    parser.add_argument("--num-threads", '-j', type=int, default=4,
                        help="Number of threads for parallel API calls")
    parser.add_argument("--segment-threshold", type=int, default=120,
                        help="Target segment length in seconds")
    parser.add_argument("--max-segment-threshold", type=int, default=180,
                        help="Maximum segment length in seconds")
    parser.add_argument("--output-dir", '-o', type=str, default="./output",
                        help="Output directory for results")
    parser.add_argument("--silence", '-s', action="store_true",
                        help="Reduce output verbosity")
    return parser.parse_args()


def setup_qwen_api(api_key: str) -> QwenASR:
    """Setup Qwen3-ASR with API key"""
    if api_key:
        dashscope.api_key = api_key
    else:
        if "DASHSCOPE_API_KEY" not in os.environ:
            raise ValueError(
                "Please set DASHSCOPE_API_KEY environment variable or use --dashscope-api-key")

    return QwenASR(model="qwen3-asr-flash")


def save_segments_to_temp(segments: List, base_name: str, output_dir: str) -> List[str]:
    """Save audio segments to temporary files for API processing"""
    temp_dir = os.path.join(output_dir, f"{base_name}_segments")
    os.makedirs(temp_dir, exist_ok=True)

    segment_paths = []
    for idx, segment in enumerate(segments):
        segment_path = os.path.join(temp_dir, f"segment_{idx:03d}.wav")
        save_audio_file(segment, segment_path)
        segment_paths.append(segment_path)

    return segment_paths


def transcribe_segments_parallel(
    segment_paths: List[str],
    qwen_asr: QwenASR,
    context: str,
    num_threads: int,
    silence: bool = False
) -> Tuple[str, str]:
    """
    Transcribe audio segments in parallel using Qwen3-ASR
    Returns: (full_text, detected_language)
    """
    results = []
    languages = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all transcription tasks
        future_dict = {
            executor.submit(qwen_asr.asr, segment_path, context): idx
            for idx, segment_path in enumerate(segment_paths)
        }

        # Progress bar
        if not silence:
            pbar = tqdm(total=len(future_dict),
                        desc="Transcribing segments with Qwen3-ASR")

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_dict):
            idx = future_dict[future]
            try:
                language, transcription = future.result()
                results.append((idx, transcription))
                languages.append(language)

                if not silence:
                    pbar.update(1)
            except Exception as e:
                print(f"Error transcribing segment {idx}: {e}")
                # Empty transcription for failed segments
                results.append((idx, ""))

        if not silence:
            pbar.close()

    # Sort results by original segment order
    results.sort(key=lambda x: x[0])

    # Combine transcriptions
    full_text = " ".join(text for _, text in results if text.strip())

    # Detect most common language
    if languages:
        detected_language = Counter(languages).most_common(1)[0][0]
    else:
        detected_language = "unknown"

    return full_text, detected_language


def save_transcription_results(
    input_file: str,
    full_text: str,
    language: str,
    output_dir: str,
    vocals_file_path: str = None
) -> str:
    """Save transcription results to file"""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_transcription.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Original File: {input_file}\n")
        if vocals_file_path:
            f.write(f"Vocals File: {vocals_file_path}\n")
        f.write(f"Detected Language: {language}\n")
        f.write(f"Transcription:\n\n{full_text}\n")

    return output_file


def cleanup_temp_files(temp_dir: str):
    """Clean up temporary segment files"""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def main():
    args = parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file '{args.input_file}' not found!")

    # Setup APIs
    qwen_asr = setup_qwen_api(args.dashscope_api_key)

    if not args.silence:
        print(f"Starting audio processing pipeline for: {args.input_file}")
        if args.extract_vocals:
            print("✓ Vocal separation enabled")
        print(f"✓ Target segment length: {args.segment_threshold}s")
        print(f"✓ Parallel threads: {args.num_threads}")

    # Step 1: Load VAD model
    if not args.silence:
        print("\nStep 1: Loading VAD model...")
    vad_model = load_silero_vad(onnx=True)

    # Step 2: Process audio (vocal separation + VAD segmentation)
    if not args.silence:
        print("\nStep 2: Processing audio with vocal separation and segmentation...")

    segments, vocals_file_path = process_audio_with_vocal_separation(
        audio_file_path=args.input_file,
        vad_model=vad_model,
        replicate_token=args.replicate_token,
        extract_vocals=args.extract_vocals,
        segment_threshold_s=args.segment_threshold,
        max_segment_threshold_s=args.max_segment_threshold
    )

    if not args.silence:
        print(f"✓ Created {len(segments)} audio segments")

    # Step 3: Save segments to temporary files
    if not args.silence:
        print("\nStep 3: Preparing segments for transcription...")

    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    segment_paths = save_segments_to_temp(segments, base_name, args.output_dir)

    # Step 4: Transcribe segments in parallel
    if not args.silence:
        print(
            f"\nStep 4: Transcribing {len(segment_paths)} segments using Qwen3-ASR...")

    full_text, detected_language = transcribe_segments_parallel(
        segment_paths=segment_paths,
        qwen_asr=qwen_asr,
        context=args.context,
        num_threads=args.num_threads,
        silence=args.silence
    )

    # Step 5: Save results
    if not args.silence:
        print("\nStep 5: Saving transcription results...")

    output_file = save_transcription_results(
        input_file=args.input_file,
        full_text=full_text,
        language=detected_language,
        output_dir=args.output_dir,
        vocals_file_path=vocals_file_path
    )

    # Step 6: Cleanup
    temp_segments_dir = os.path.join(args.output_dir, f"{base_name}_segments")
    cleanup_temp_files(temp_segments_dir)

    # Final results
    if not args.silence:
        print(f"\n{'='*60}")
        print(f"TRANSCRIPTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Detected Language: {detected_language}")
        print(f"Total segments processed: {len(segments)}")
        print(f"Transcription saved to: {output_file}")
        if vocals_file_path:
            print(f"Separated vocals saved to: {vocals_file_path}")
        print(f"\nTranscription preview:")
        print(f"{full_text[:500]}{'...' if len(full_text) > 500 else ''}")
    else:
        print(f"Transcription saved to: {output_file}")


if __name__ == "__main__":
    main()
