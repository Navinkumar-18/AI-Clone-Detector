import os
from datasets import load_dataset, Audio
from tqdm import tqdm

DATA_ROOT = "./data/asvspoof2019LA/LA"
PROTOCOLS_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")

audio_dir = os.path.join(DATA_ROOT, "ASVspoof2019_LA_eval", "flac")
os.makedirs(audio_dir, exist_ok=True)
proto_path = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")

print("Downloading 50 Bonafide + 150 Spoof real samples for eval split...")
ds = load_dataset("Bisher/ASVspoof_2019_LA", split="validation", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

lines = []
target_bona = 50
target_spoof = 150
bona_count = 0
spoof_count = 0

for idx, item in enumerate(tqdm(ds, desc="Filtering eval clips")):
    if idx < 2600:
        continue
        
    raw_key = item.get('key', item.get('label', item.get('target', 0)))
    if str(raw_key).strip().lower() in ['1', 'bonafide', 'true', 'target']:
        is_bona = True
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

print(f"Saved {len(lines)} real FLAC files ({bona_count} Bonafide, {spoof_count} Spoof) to {audio_dir}")
print(f"Wrote protocol file to {proto_path}")
