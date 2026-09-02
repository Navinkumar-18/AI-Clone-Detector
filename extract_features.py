"""
extract_features.py - Frozen SSL Feature Extraction & Versioned Embedding Caching
===================================================================================

DESIGN RATIONALE & KEY ARCHITECTURAL DECISIONS (FOR REVIEW/DEFENSE):
-------------------------------------------------------------------
1. WHY FREEZE THE PRETRAINED BACKBONE (WavLM / Wav2Vec2)?
   - Compute Efficiency: Pretrained Self-Supervised Speech Models (like WavLM-base) contain 
     ~95 Million parameters. Fine-tuning the full model requires massive GPU VRAM and 
     hours of gradient computation. Freezing the backbone lets us run feature extraction 
     rapidly even on free single-GPU environments (Colab T4).
   - Preventing Catastrophic Overfitting: ASVspoof dataset audio clips share similar background 
     acoustics. Fine-tuning a huge 95M-parameter model on a small classifier task often leads to 
     the backbone "memorizing" specific channel noise or speakers rather than learning general 
     spoofing signatures. Freezing retains rich, uncorrupted SSL speech representations.

2. WHY MEAN-POOLING ACROSS TIME FRAMES?
   - Input audio clips vary in duration (e.g., 2.1s to 5.4s). The WavLM encoder produces a 
     sequence of feature frames of shape (Batch, Sequence_Length, 768).
   - Mean-pooling computes the average across the temporal dimension: 
       Embedding = Mean(Hidden_States, dim=1) -> shape (768,)
   - This condenses variable-length audio frame sequences into a fixed 768-dimensional utterance vector, 
     capturing overall spectral, phase, and acoustic characteristics without needing dynamic 
     padding or complex RNN sequence modeling.

3. WHY VERSION CACHED EMBEDDINGS BY MODEL NAME?
   - Filenames include sanitized model tags (e.g., `X_train_wavlm_base.npy`).
   - If a teammate switches from WavLM to Wav2Vec2 or Whisper later, versioned filenames guarantee 
     that stale cache files from a different model architecture are never silently reused.
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel, Wav2Vec2Model

from utils import parse_asvspoof_protocol, locate_split_paths, load_and_resample_audio, get_model_tag


def load_frozen_backbone(model_name: str, device: torch.device):
    """
    Load a HuggingFace self-supervised speech backbone in frozen eval mode.
    """
    print(f"Loading backbone model: {model_name}...")
    try:
        if "wavlm" in model_name.lower():
            model = WavLMModel.from_pretrained(model_name)
        else:
            model = Wav2Vec2Model.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: Failed to load {model_name} ({e}). Falling back to 'facebook/wav2vec2-base'")
        model_name = "facebook/wav2vec2-base"
        model = Wav2Vec2Model.from_pretrained(model_name)
        
    # Use AutoFeatureExtractor (pure audio) instead of AutoProcessor (which requires ASR text vocab)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    # Freeze ALL backbone parameters (requires_grad = False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    model.to(device)
    return model, feature_extractor, model_name


def extract_split_features(
    model, 
    processor, 
    entries: list, 
    device: torch.device, 
    max_samples: int = None
):
    """
    Process audio clips through the frozen backbone and mean-pool last hidden states.
    
    Returns:
        X: numpy array of shape (N, 768) - mean-pooled embeddings
        y: numpy array of shape (N,) - binary targets (1=bonafide, 0=spoof)
    """
    embeddings = []
    targets = []
    
    if max_samples and max_samples < len(entries):
        print(f" Subsampling {max_samples} items out of {len(entries)} for fast hackathon demo run.")
        entries = entries[:max_samples]
        
    for entry in tqdm(entries, desc="Extracting WavLM Features"):
        audio_path = entry.get('full_path', None)
        if not audio_path or not os.path.exists(audio_path):
            continue
            
        try:
            # 1. Load audio clip & resample to 16kHz mono
            waveform = load_and_resample_audio(audio_path, target_sr=16000)
            
            # Convert to numpy for HF processor
            input_values = processor(
                waveform.numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_values.to(device)
            
            # 2. Pass through frozen SSL model (no gradient computation)
            with torch.no_grad():
                outputs = model(input_values)
                # last_hidden_state shape: (1, sequence_length, 768)
                hidden_states = outputs.last_hidden_state
                
                # 3. Mean-pool along sequence dimension (dim=1) -> shape: (1, 768)
                mean_pooled = torch.mean(hidden_states, dim=1)
                
            embeddings.append(mean_pooled.squeeze(0).cpu().numpy())
            targets.append(entry['target'])
            
        except Exception as err:
            print(f" Error processing {audio_path}: {err}")
            continue
            
    if len(embeddings) == 0:
        return np.empty((0, 768)), np.empty((0,))
        
    X = np.array(embeddings, dtype=np.float32)
    y = np.array(targets, dtype=np.int64)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Extract Frozen SSL Embeddings from ASVspoof Audio")
    parser.add_argument("--data_dir", type=str, default="./data/asvspoof2019LA", help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./data/features", help="Directory to save cached .npy embeddings")
    parser.add_argument("--model_name", type=str, default="microsoft/wavlm-base", help="HuggingFace speech model ID")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of audio files processed per split (for fast testing)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, processor, resolved_model_name = load_frozen_backbone(args.model_name, device)
    model_tag = get_model_tag(resolved_model_name)
    print(f"Model tag for versioned caching: '{model_tag}'")
    
    splits = ["train", "dev", "eval"]
    for split in splits:
        print(f"\n==================== EXTRACTING FEATURES: {split.upper()} ====================")
        
        # Define versioned output paths
        x_out_path = os.path.join(args.output_dir, f"X_{split}_{model_tag}.npy")
        y_out_path = os.path.join(args.output_dir, f"y_{split}_{model_tag}.npy")
        
        if os.path.exists(x_out_path) and os.path.exists(y_out_path):
            print(f" Cache already exists for '{split}' split at: {x_out_path}")
            print(" Skipping extraction. Delete existing .npy files if you wish to recompute.")
            continue
            
        protocol_file, audio_dir = locate_split_paths(args.data_dir, split)
        if not protocol_file or not os.path.exists(protocol_file):
            print(f" Protocol file for '{split}' missing. Skipping...")
            continue
            
        entries = parse_asvspoof_protocol(protocol_file)
        
        # Attach resolved physical paths
        valid_entries = []
        for e in entries:
            fname = e['filename']
            candidate_flac = os.path.join(audio_dir, f"{fname}.flac") if audio_dir else ""
            candidate_wav = os.path.join(audio_dir, f"{fname}.wav") if audio_dir else ""
            
            if os.path.exists(candidate_flac):
                e['full_path'] = candidate_flac
                valid_entries.append(e)
            elif os.path.exists(candidate_wav):
                e['full_path'] = candidate_wav
                valid_entries.append(e)
                
        print(f" Found {len(valid_entries)} audio clips on disk for split '{split}'.")
        
        if len(valid_entries) == 0:
            print(f" WARNING: No audio files resolved for '{split}'. Skipping feature extraction.")
            continue
            
        X, y = extract_split_features(model, processor, valid_entries, device, max_samples=args.max_samples)
        
        print(f" Saving versioned embeddings -> X shape: {X.shape}, y shape: {y.shape}")
        np.save(x_out_path, X)
        np.save(y_out_path, y)
        print(f" Saved to: {x_out_path} and {y_out_path}")


if __name__ == "__main__":
    main()
