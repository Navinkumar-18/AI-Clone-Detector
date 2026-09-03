import os
from datasets import load_dataset, Audio
from tqdm import tqdm

DATA_ROOT = "./data/asvspoof2019LA/LA"
PROTOCOLS_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")

audio_dir = os.path.join(DATA_ROOT, "ASVspoof2019_LA_eval", "flac")
os.makedirs(audio_dir, exist_ok=True)
proto_path = os.path.join(PROTOCOLS_DIR, "ASVspoof2019.LA.cm.eval.trl.txt")

print("Downloading 200 real samples for eval split from validation stream...")
ds = load_dataset("Bisher/ASVspoof_2019_LA", split="validation", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

lines = []
saved_count = 0

for idx, item in enumerate(tqdm(ds, desc="Saving eval clips")):
    # Skip first 100 items (used in dev split) to ensure distinct eval samples
    if idx < 100:
        continue
        
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
    
    audio_bytes = item['audio']['bytes']
    out_flac_path = os.path.join(audio_dir, f"{fname_stem}.flac")
    
    with open(out_flac_path, "wb") as f:
        f.write(audio_bytes)
        
    line = f"{spk} {fname_stem} - {sys_id} {key}"
    lines.append(line)
    
    saved_count += 1
    if saved_count >= 200:
        break

with open(proto_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Saved {saved_count} real FLAC files to {audio_dir}")
print(f"Wrote protocol file to {proto_path}")
