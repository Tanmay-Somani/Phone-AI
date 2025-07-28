import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Input and output directories
input_dir = Path("/Users/apple/Desktop/Tanmay S/processed/ujs/wav")
output_dir = Path("/Users/apple/Desktop/Tanmay S/processed/ujs/wav_8k")
output_dir.mkdir(parents=True, exist_ok=True)

# Function to convert using sox
def downsample(in_path):
    out_path = output_dir / in_path.name
    command = [
        "sox",
        str(in_path),
        "-r", "8000",   # Sample rate
        "-c", "1",      # Mono
        str(out_path)
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {in_path.name}: {e}")

# Gather all .wav files
wavs = list(input_dir.glob("*.wav"))
print(f"Found {len(wavs)} .wav files to downsample...")

# Process in parallel (8 threads)
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(downsample, wavs)

print(f"Downsampling complete! Saved to: {output_dir}")
