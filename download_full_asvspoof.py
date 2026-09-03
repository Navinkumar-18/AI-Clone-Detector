import os, time
from datasets import load_dataset, Audio
from tqdm import tqdm

DATA_ROOT = "./data/asvspoof2019LA/LA"
PROTOCOLS_DIR = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols")
os.makedirs(PROTOCOLS_DIR, exist_ok=True)

SPLITS = [
    ("train",      "train", "ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    ("validation", "dev",   "ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt"),
    ("test",       "eval",  "ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt"),
]
EXPECTED = {"train": 25380, "dev": 24844, "eval": 71237}

print("Starting full ASVspoof 2019 LA download (~7.5 GB, no caps)...")

for hf_split, local_name, folder_name, proto_filename in SPLITS:
    expected = EXPECTED[local_name]
    audio_dir = os.path.join(DATA_ROOT, folder_name, "flac")
    os.makedirs(audio_dir, exist_ok=True)
    proto_path = os.path.join(PROTOCOLS_DIR, proto_filename)
    existing_flac = [f for f in os.listdir(audio_dir) if f.endswith(".flac")]
    existing_set = set(f[:-5] for f in existing_flac)
    n_existing = len(existing_flac)
    if n_existing >= expected and os.path.exists(proto_path):
        print("Split " + local_name + ": complete (" + str(n_existing) + " files). Skipping.")
        continue
    print("Downloading " + local_name + " from HF split='" + hf_split + "', expected=" + str(expected))
    ds = load_dataset("Bisher/ASVspoof_2019_LA", split=hf_split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    lines_out = []
    saved = 0
    t0 = time.time()
    pbar = tqdm(desc=local_name, total=expected, unit="file")
    for item in ds:
        fname = item.get("audio_file_name", item.get("path", ""))
        stem = fname[:-5] if fname.endswith(".flac") else fname
        spk = item.get("speaker_id", "LA_0001")
        sid = item.get("system_id", "-")
        if sid is None or str(sid).strip() in ["", "nan", "None"]:
            sid = "-"
        key = str(item.get("key", "spoof")).strip().lower()
        if key not in ["bonafide", "spoof"]:
            key = "spoof"
        lines_out.append(spk + " " + stem + " - " + sid + " " + key)
        if stem not in existing_set:
            with open(os.path.join(audio_dir, stem + ".flac"), "wb") as fout:
                fout.write(item["audio"]["bytes"])
            existing_set.add(stem)
        saved += 1
        pbar.update(1)
        if saved % 1000 == 0:
            elapsed = time.time() - t0
            rate = saved / max(elapsed, 1)
            rem = (expected - saved) / max(rate, 0.01)
            print(str(saved) + "/" + str(expected) + " | " + str(round(rate, 1)) + " f/s | ~" + str(round(rem / 60)) + " min left")
            with open(proto_path + ".partial", "w", encoding="utf-8") as pf:
                pf.write("\n".join(lines_out) + "\n")
    pbar.close()
    with open(proto_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(lines_out) + "\n")
    if os.path.exists(proto_path + ".partial"):
        os.remove(proto_path + ".partial")
    elapsed = time.time() - t0
    print("Done " + local_name + ": " + str(saved) + " files in " + str(round(elapsed / 60, 1)) + " min")

print("ALL SPLITS DOWNLOADED.")
print("Next steps:")
print("  python extract_features.py")
print("  python train_classifier.py")