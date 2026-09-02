"""
utils.py - Shared Utilities for ASVspoof 2019 LA Audio Deepfake Detector
========================================================================

This module provides reusable helper functions for:
1. Parsing ASVspoof 2019 LA protocol text files.
2. Computing Equal Error Rate (EER), the standard biometrics/anti-spoofing metric.
3. Audio loading, mono-channel conversion, and resampling (16kHz).
4. Sanitizing model names for versioned feature caching.
"""

import os
import glob
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def get_model_tag(model_name: str) -> str:
    """
    Sanitize HuggingFace model identifiers into safe filename strings.
    Example: 'microsoft/wavlm-base' -> 'wavlm-base'
    """
    return model_name.split("/")[-1].replace("-", "_").lower()


def parse_asvspoof_protocol(protocol_path: str):
    """
    Parse an ASVspoof 2019 LA protocol file.
    
    Standard protocol line format:
    LA_0079 LA_T_1138241 - - bonafide
    LA_0079 LA_T_1271820 A01 - spoof
    
    Fields:
    0: SPEAKER_ID (e.g., LA_0079)
    1: AUDIO_FILE_NAME (e.g., LA_T_1138241)
    2: ENVIRONMENT/SYSTEM_ID (e.g., '-', 'A01', 'A02' ... 'A19')
    3: TRASH/RESERVED (e.g., '-')
    4: KEY / LABEL ('bonafide' or 'spoof')
    
    Returns:
        List of dicts: [{'speaker': str, 'filename': str, 'system_id': str, 'label': str, 'target': int}]
        where target: 1 for bonafide, 0 for spoof.
    """
    if not os.path.exists(protocol_path):
        raise FileNotFoundError(f"Protocol file not found at: {protocol_path}")
        
    entries = []
    with open(protocol_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            speaker_id = parts[0]
            audio_name = parts[1]
            system_id = parts[2]
            key = parts[4].lower()
            
            # Map key to binary target: 1 = bonafide (real human), 0 = spoof (AI-cloned)
            target = 1 if key == "bonafide" else 0
            
            entries.append({
                "speaker": speaker_id,
                "filename": audio_name,
                "system_id": system_id,
                "label": key,
                "target": target
            })
            
    return entries


def locate_split_paths(base_dir: str, split: str):
    """
    Locates the protocol file and flac/wav audio folder for a given split ('train', 'dev', 'eval').
    Supports common ASVspoof 2019 LA folder layouts.
    """
    split = split.lower()
    
    # Common protocol file patterns
    protocol_patterns = [
        os.path.join(base_dir, "LA", "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{split}.*.txt"),
        os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{split}.*.txt"),
        os.path.join(base_dir, f"{split}_protocol.txt"),
        os.path.join(base_dir, f"{split}.txt"),
    ]
    
    protocol_file = None
    for pattern in protocol_patterns:
        matches = glob.glob(pattern)
        if matches:
            protocol_file = matches[0]
            break
            
    # Common audio folder patterns
    audio_dir_patterns = [
        os.path.join(base_dir, "LA", f"ASVspoof2019_LA_{split}", "flac"),
        os.path.join(base_dir, "LA", f"ASVspoof2019_LA_{split}", "wav"),
        os.path.join(base_dir, f"ASVspoof2019_LA_{split}", "flac"),
        os.path.join(base_dir, f"ASVspoof2019_LA_{split}", "wav"),
        os.path.join(base_dir, split, "flac"),
        os.path.join(base_dir, split, "wav"),
        os.path.join(base_dir, split),
    ]
    
    audio_dir = None
    for adir in audio_dir_patterns:
        if os.path.exists(adir):
            audio_dir = adir
            break
            
    return protocol_file, audio_dir


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute Equal Error Rate (EER) and the corresponding optimal threshold.
    
    BACKGROUND / DESIGN DECISION:
    -----------------------------
    Equal Error Rate (EER) is the standard performance metric in biometric authentication 
    and anti-spoofing tasks (like voice clone detection).
    
    - False Acceptance Rate (FAR): Probability that a SPOOF (fake) clip is incorrectly 
      accepted as BONAFIDE (real).
    - False Rejection Rate (FRR): Probability that a BONAFIDE (real) clip is incorrectly 
      rejected as SPOOF (fake).
      
    EER is the operating point on the ROC curve where FAR == FRR.
    Unlike standard classification Accuracy (which is heavily biased when datasets are 
    imbalanced, e.g. 90% spoof vs 10% bonafide), EER measures the fundamental security 
    trade-off independent of class distribution or arbitrary threshold selection (like 0.5).
    
    Args:
        y_true: 1D array of ground truth targets (1 for bonafide, 0 for spoof).
        y_scores: 1D array of continuous scores (higher = more likely bonafide).
        
    Returns:
        eer (float): Equal Error Rate as a percentage (e.g., 5.42 -> 5.42%).
        threshold (float): Score threshold at which FAR equals FRR.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR) across thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # In our convention:
    # Target 1 = Bonafide, Target 0 = Spoof
    # FRR = False Rejection Rate = 1 - TPR (Bonafide misclassified as Spoof)
    # FAR = False Acceptance Rate = FPR (Spoof misclassified as Bonafide)
    frr = 1.0 - tpr
    far = fpr
    
    # Find the threshold where far == frr using linear interpolation
    # brentq finds the zero root of (far - frr)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        threshold = float(interp1d(fpr, thresholds)(eer))
    except Exception:
        # Fallback if numerical interpolation hits edge bounds
        idx = np.nanargmin(np.abs(far - frr))
        eer = far[idx]
        threshold = thresholds[idx]
        
    return eer * 100.0, threshold


def load_and_resample_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio clip, convert to mono, and resample to target_sr (default 16kHz).
    
    Args:
        audio_path: Path to audio file (.flac or .wav)
        target_sr: Target sample rate expected by WavLM/Wav2Vec2 models (16000 Hz)
        
    Returns:
        waveform: 1D PyTorch float tensor of shape (num_samples,)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    try:
        # Primary loader: torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception:
        # Fallback loader: soundfile + librosa
        speech_data, sample_rate = sf.read(audio_path)
        waveform = torch.from_numpy(speech_data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
            
    # Convert multi-channel (stereo) to mono by averaging channels
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Resample if sample rate differs from 16kHz
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        
    # Squeeze to 1D tensor shape (num_samples,)
    return waveform.squeeze(0)
