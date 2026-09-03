"""
build_balanced_real_asvspoof_dataset.py
=======================================
Downloads real ASVspoof 2019 LA FLAC audio files from HuggingFace, ensuring 
a balanced representation of both bonafide (genuine human speech) and spoof (AI-cloned) audio.
"""

import os
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

DATA_ROOT = "./data/asvspoof2019LA/LA"
PROTOCOLS_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")
os.makedirs(PROTOCOLS_DIR, exist_ok=True)

# target bonafide and spoof counts per split
CONFIGS = [
    ("train", "train", "ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", 150, 450),
    ("validation", "dev", "ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt", 50, 150),
    ("test", "eval", "ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt", 50, 150),
]

print("==================== DOWNLOADING BALANCED REAL ASVSPOOF 2019 LA DATASET ====================")

for hf_split, local_name, folder_name, proto_filename, target_bona, target_spoof in CONFIGS:
    print(f"\nProcessing real split '{local_name}' (Target: {target_bona} Bonafide + {target_spoof} Spoof)...")
    
    audio_dir = os.path.join(DATA_ROOT, folder_name, "flac")
    os.makedirs(audio_dir, exist_ok=True)
    proto_path = os.path.join(PROTOCOLS_DIR, proto_filename)
    
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split=hf_split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    
    lines = []
    bona_count = 0
    spoof_count = 0
    
    for item in tqdm(ds, desc=f"Filtering {local_name} clips"):
        raw_key = item.get('key', item.get('label', item.get('target', 0)))
        
        # Determine if bonafide or spoof
        if str(raw_key).strip().lower() in ['1', 'bonafide', 'true', 'target']:
            is_bona = True
        elif str(raw_key).strip().lower() in ['0', 'spoof', 'false', 'nontarget']:
            is_bona = False
        elif isinstance(raw_key, int):
            is_bona = (raw_key == 1)
        else:
            is_bona = False
            
        if is_bona and bona_count < target_bona:
            is_keep = True
            bona_count += 1
            key_str = 'bonafide'
        elif (not is_bona) and spoof_count < target_spoof:
            is_keep = True
            spoof_count += 1
            key_str = 'spoof'
        else:
            is_keep = False
            
        if not is_keep:
            if bona_count >= target_bona and spoof_count >= target_spoof:
                break
            continue
            
        spk = item.get('speaker_id', 'LA_0001')
        fname = item.get('audio_file_name', item.get('path', ''))
        fname_stem = fname[:-5] if fname.endswith('.flac') else fname
        sys_id = item.get('system_id', '-')
        if sys_id is None or str(sys_id).strip() in ['', 'nan', 'None']:
            sys_id = '-' if is_bona else 'A01'
            
        audio_bytes = item['audio']['bytes']
        out_flac_path = os.path.join(audio_dir, f"{fname_stem}.flac")
        with open(out_flac_path, "wb") as f:
            f.write(audio_bytes)
            
        line = f"{spk} {fname_stem} - {sys_id} {key_str}"
        lines.append(line)
        
        if bona_count >= target_bona and spoof_count >= target_spoof:
            break
            
    with open(proto_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f" Saved {len(lines)} real FLAC audio files ({bona_count} Bonafide, {spoof_count} Spoof) to: {audio_dir}")
    print(f" Wrote protocol file to: {proto_path}")

print("\n==================== BALANCED DATASET SETUP COMPLETE! ====================")
