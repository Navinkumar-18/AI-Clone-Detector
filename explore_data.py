"""
explore_data.py - Data Exploration & Sanity Verification
=========================================================

This script performs initial exploratory data analysis (EDA) on the ASVspoof 2019 LA dataset:
1. Parses train, dev, and eval protocol files.
2. Prints key label distributions (counts and ratios of bonafide vs spoof speech).
3. Performs a SPLIT-SIZE SANITY CHECK: compares resolved audio files on disk against 
   protocol entries and ASVspoof 2019 LA benchmark numbers to detect path/file resolution bugs.
4. Loads bonafide and spoof audio samples and plots side-by-side waveforms + Mel-Spectrograms.

Benchmark ASVspoof 2019 LA counts:
- Train: 2,580 bonafide + 22,800 spoof = 25,380 total
- Dev:   2,548 bonafide + 22,296 spoof = 24,844 total
- Eval:  7,355 bonafide + 63,882 spoof = 71,237 total
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from utils import parse_asvspoof_protocol, locate_split_paths, load_and_resample_audio

# Standard benchmark sizes for ASVspoof 2019 LA
BENCHMARK_COUNTS = {
    "train": {"bonafide": 2580, "spoof": 22800, "total": 25380},
    "dev":   {"bonafide": 2548, "spoof": 22296, "total": 24844},
    "eval":  {"bonafide": 7355, "spoof": 63882, "total": 71237},
}


def explore_split(data_dir: str, split_name: str):
    """
    Parse protocol for a split, verify label distribution, and check resolved files on disk.
    """
    protocol_file, audio_dir = locate_split_paths(data_dir, split_name)
    
    print(f"\n==================== SPLIT: {split_name.upper()} ====================")
    print(f"Protocol file location: {protocol_file}")
    print(f"Audio directory:        {audio_dir}")
    
    if protocol_file is None or not os.path.exists(protocol_file):
        print(f" WARNING: Protocol file for '{split_name}' split not found under {data_dir}.")
        print(f" Expected benchmark: {BENCHMARK_COUNTS.get(split_name, {})}")
        return None, None, []
        
    entries = parse_asvspoof_protocol(protocol_file)
    total_protocol_count = len(entries)
    
    bonafide_count = sum(1 for e in entries if e['target'] == 1)
    spoof_count = sum(1 for e in entries if e['target'] == 0)
    
    print(f"\n--- Protocol Label Distribution ---")
    print(f" Total Protocol Entries: {total_protocol_count:,}")
    print(f"   - Bonafide (Real):     {bonafide_count:,} ({bonafide_count/total_protocol_count*100:.2f}%)")
    print(f"   - Spoof (AI-Cloned):   {spoof_count:,} ({spoof_count/total_protocol_count*100:.2f}%)")
    
    # Check against known ASVspoof 2019 LA benchmark numbers
    bench = BENCHMARK_COUNTS.get(split_name, None)
    if bench:
        print(f"\n--- Benchmark Comparison (ASVspoof 2019 LA) ---")
        print(f" Expected Total:         {bench['total']:,}")
        if total_protocol_count == bench['total']:
            print(" Protocol count matches standard ASVspoof 2019 LA split size perfectly!")
        else:
            print(f" WARNING: Protocol count ({total_protocol_count:,}) differs from standard benchmark ({bench['total']:,}).")
            
    # Check physical audio files on disk (Split-Size Sanity Print)
    resolved_count = 0
    missing_count = 0
    resolved_entries = []
    
    if audio_dir and os.path.exists(audio_dir):
        for entry in entries:
            fname = entry['filename']
            # Search for .flac or .wav extension
            candidate_flac = os.path.join(audio_dir, f"{fname}.flac")
            candidate_wav = os.path.join(audio_dir, f"{fname}.wav")
            
            if os.path.exists(candidate_flac):
                entry['full_path'] = candidate_flac
                resolved_entries.append(entry)
                resolved_count += 1
            elif os.path.exists(candidate_wav):
                entry['full_path'] = candidate_wav
                resolved_entries.append(entry)
                resolved_count += 1
            else:
                missing_count += 1
                
        print(f"\n--- Disk File Resolution Sanity Check ---")
        print(f" Files resolved on disk: {resolved_count:,} / {total_protocol_count:,}")
        if missing_count > 0:
            print(f" WARNING: {missing_count:,} audio files listed in protocol were NOT found in {audio_dir}.")
        else:
            print(" 100% of protocol audio files successfully resolved on disk!")
    else:
        print(f"\n WARNING: Audio directory for '{split_name}' not found on disk at {audio_dir}.")
        
    return protocol_file, audio_dir, resolved_entries


def plot_side_by_side(bonafide_path: str, spoof_path: str, output_image_path: str = "data_visualization.png"):
    """
    Plot Waveform and Mel-Spectrogram for a Bonafide clip and a Spoof clip side by side.
    """
    print(f"\nPlotting comparative visualizations...")
    print(f"  Bonafide sample: {bonafide_path}")
    print(f"  Spoof sample:    {spoof_path}")
    
    # Load audio clips using standardized utils
    y_bona = load_and_resample_audio(bonafide_path, target_sr=16000).numpy()
    y_spoof = load_and_resample_audio(spoof_path, target_sr=16000).numpy()
    
    sr = 16000
    
    # Compute Mel-spectrograms (N_FFT=1024, Hop_Length=512)
    S_bona = librosa.feature.melspectrogram(y=y_bona, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    S_bona_db = librosa.power_to_db(S_bona, ref=np.max)
    
    S_spoof = librosa.feature.melspectrogram(y=y_spoof, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    S_spoof_db = librosa.power_to_db(S_spoof, ref=np.max)
    
    # Create 2x2 Grid Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ASVspoof 2019 LA: Bonafide (Real) vs Spoof (AI-Cloned) Audio Comparison", fontsize=14, fontweight='bold')
    
    # 1. Bonafide Waveform
    axes[0, 0].plot(np.linspace(0, len(y_bona)/sr, len(y_bona)), y_bona, color='#1f77b4')
    axes[0, 0].set_title("Bonafide (Real Human Speech) Waveform", fontsize=11, fontweight='bold', color='#1f77b4')
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spoof Waveform
    axes[0, 1].plot(np.linspace(0, len(y_spoof)/sr, len(y_spoof)), y_spoof, color='#d62728')
    axes[0, 1].set_title("Spoof (AI-Generated Synthetic) Waveform", fontsize=11, fontweight='bold', color='#d62728')
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bonafide Mel-Spectrogram
    img1 = librosa.display.specshow(S_bona_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=axes[1, 0], cmap='magma')
    axes[1, 0].set_title("Bonafide Mel-Spectrogram", fontsize=11, fontweight='bold')
    fig.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')
    
    # 4. Spoof Mel-Spectrogram
    img2 = librosa.display.specshow(S_spoof_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=axes[1, 1], cmap='magma')
    axes[1, 1].set_title("Spoof Mel-Spectrogram", fontsize=11, fontweight='bold')
    fig.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f" Visualization successfully saved to: {os.path.abspath(output_image_path)}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ASVspoof 2019 LA Dataset Exploration & Sanity Verification")
    parser.add_argument("--data_dir", type=str, default="./data/asvspoof2019LA", help="Path to ASVspoof 2019 LA dataset root")
    parser.add_argument("--save_plot", type=str, default="data_visualization.png", help="Output path for comparative plot")
    args = parser.parse_args()
    
    print(f"Starting Data Exploration on dataset path: {args.data_dir}")
    
    train_proto, train_audio, train_entries = explore_split(args.data_dir, "train")
    dev_proto, dev_audio, dev_entries = explore_split(args.data_dir, "dev")
    eval_proto, eval_audio, eval_entries = explore_split(args.data_dir, "eval")
    
    # Find sample bonafide and spoof audio clips for plotting
    sample_bona = None
    sample_spoof = None
    
    all_entries = train_entries + dev_entries + eval_entries
    for e in all_entries:
        if sample_bona is None and e['target'] == 1 and 'full_path' in e:
            sample_bona = e['full_path']
        if sample_spoof is None and e['target'] == 0 and 'full_path' in e:
            sample_spoof = e['full_path']
        if sample_bona and sample_spoof:
            break
            
    if sample_bona and sample_spoof:
        plot_side_by_side(sample_bona, sample_spoof, args.save_plot)
    else:
        print("\nNote: Audio files not found on disk yet. To visualize waveforms/spectrograms, download ASVspoof 2019 LA to ./data/asvspoof2019LA.")


if __name__ == "__main__":
    main()
