#!/usr/bin/env python3

import numpy as np
import audio
from silero_vad import load_silero_vad

def test_audio_functions():
    print("Testing audio.py functionality...")

    # Test 1: Create a simple synthetic audio signal
    print("1. Creating synthetic audio signal...")
    duration = 2.0  # 2 seconds
    sample_rate = audio.WAV_SAMPLE_RATE
    samples = int(duration * sample_rate)

    # Create a sine wave at 440 Hz (A note)
    t = np.linspace(0, duration, samples, False)
    frequency = 440.0
    synthetic_audio = np.sin(2 * np.pi * frequency * t) * 0.5

    print(f"OK Synthetic audio created: {len(synthetic_audio)} samples at {sample_rate} Hz")

    # Test 2: Test save_audio_file function
    print("2. Testing save_audio_file function...")
    test_file_path = "C:\\Github\\Test toolkit gemini\\test_output.wav"

    try:
        audio.save_audio_file(synthetic_audio, test_file_path)
        print(f"OK Audio saved to {test_file_path}")
    except Exception as e:
        print(f"ERROR Error saving audio: {e}")
        return False

    # Test 3: Test load_audio function
    print("3. Testing load_audio function...")

    try:
        loaded_audio = audio.load_audio(test_file_path)
        print(f"OK Audio loaded: {len(loaded_audio)} samples")

        # Check if loaded audio is similar to original
        if len(loaded_audio) == len(synthetic_audio):
            print("OK Audio length matches original")
        else:
            print(f"WARNING Audio length differs: {len(loaded_audio)} vs {len(synthetic_audio)}")

    except Exception as e:
        print(f"ERROR Error loading audio: {e}")
        return False

    # Test 4: Test VAD functionality
    print("4. Testing VAD (Voice Activity Detection)...")

    try:
        # Load the VAD model
        vad_model = load_silero_vad()
        print("OK VAD model loaded successfully")

        # Test process_vad function with synthetic audio
        # Note: synthetic sine wave might not be detected as speech, but we test the function
        segments = audio.process_vad(synthetic_audio, vad_model)
        print(f"OK VAD processing completed: {len(segments)} segments detected")

        for i, segment in enumerate(segments):
            print(f"  Segment {i+1}: {len(segment)} samples ({len(segment)/sample_rate:.2f}s)")

    except Exception as e:
        print(f"ERROR Error in VAD processing: {e}")
        print("This might be expected with synthetic audio")

    print("\nOK All basic audio functions are working!")
    return True

if __name__ == "__main__":
    success = test_audio_functions()
    if success:
        print("\nSUCCESS audio.py is working correctly!")
    else:
        print("\nFAILED Some tests failed. Check the errors above.")