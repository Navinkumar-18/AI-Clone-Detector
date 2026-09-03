"""
build_real_asvspoof_dataset.py
==============================
Downloads real ASVspoof 2019 LA FLAC audio files and populates standard 
dataset protocols on disk under ./data/asvspoof2019LA/LA/
"""

import os
import io
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

DATA_ROOT = "./data/asvspoof2019LA/LA"
PROTOCOLS_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")
os.makedirs(PROTOCOLS_DIR, exist_ok=True)

SPLIT_MAPPING = [
    ("train", "train", "ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", 600),
    ("validation", "dev", "ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt", 200),
    ("test", "eval", "ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt", 200),
]

print("==================== DOWNLOADING REAL ASVSPOOF 2019 LA DATASET ====================")

for hf_split, local_split_name, folder_name, proto_filename, limit in SPLIT_MAPPING:
    print(f"\nProcessing real split '{local_split_name}' (target: {limit} real samples)...")
    
    audio_dir = os.path.join(DATA_ROOT, folder_name, "flac")
    os.makedirs(audio_dir, exist_ok=True)
    
    proto_path = os.path.join(PROTOCOLS_DIR, proto_filename)
    
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split=hf_split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    
    lines = []
    saved_count = 0
    
    for item in tqdm(ds, total=limit, desc=f"Saving {local_split_name} clips"):
        spk = item.get('speaker_id', 'LA_0001')
        fname = item.get('audio_file_name', item.get('path', ''))
        if fname.endswith('.flac'):
            fname_stem = fname[:-5]
        else:
            fname_stem = fname
            
        sys_id = item.get('system_id', '-')
        if sys_id is None or sys_id == '' or str(sys_id) == 'nan':
            sys_id = '-'
            
        key = item.get('key', 'spoof')
        
        # Save FLAC file to disk
        audio_bytes = item['audio']['bytes']
        out_flac_path = os.path.join(audio_dir, f"{fname_stem}.flac")
        
        with open(out_flac_path, "wb") as f:
            f.write(audio_bytes)
            
        # Format line: SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT SYSTEM_ID KEY
        # ASVspoof protocol format: LA_0079 LA_T_1138215 - - bonafide / LA_0079 LA_T_1271820 - A01 spoof
        line = f"{spk} {fname_stem} - {sys_id} {key}"
        lines.append(line)
        
        saved_count += 1
        if saved_count >= limit:
            break
            
    with open(proto_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f" Saved {saved_count} real FLAC audio files to: {audio_dir}")
    print(f" Wrote protocol file with {len(lines)} entries to: {proto_path}")

print("\n==================== DATASET SETUP COMPLETE! ====================")
