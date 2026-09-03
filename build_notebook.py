import json, textwrap

def cell(src, ctype='code'):
    lines = [l + '\n' for l in src.split('\n')]
    if lines and lines[-1] == '\n':
        lines[-1] = ''
    if ctype == 'code':
        return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
                'outputs': [], 'source': lines}
    else:
        return {'cell_type': 'markdown', 'metadata': {}, 'source': lines}

cells = []

# ---------- Cell 0: Title ----------
cells.append(cell(textwrap.dedent("""\
    # ASVspoof 2019 LA — Full Pipeline (GPU)
    ## Download → Feature Extraction (WavLM-base) → Train MLP → Eval EER
    **Split mapping (canonical):**
    - HF `train` → `ASVspoof2019_LA_train` (25,380 files, `LA_T_*`) → train
    - HF `validation` → `ASVspoof2019_LA_dev` (24,844 files, `LA_D_*`) → dev
    - HF `test` → `ASVspoof2019_LA_eval` (71,237 files, `LA_E_*`) → eval
    
    Run all cells top-to-bottom. GPU required for feature extraction (~30-60 min on T4)."""), 'markdown'))

# ---------- Cell 1: Install ----------
cells.append(cell(textwrap.dedent("""\
    !pip install -q datasets transformers torchaudio soundfile librosa scikit-learn joblib tqdm
    print('Installation complete')""")))

# ---------- Cell 2: GPU check ----------
cells.append(cell(textwrap.dedent("""\
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
    else:
        print('WARNING: No GPU detected. Enable GPU runtime: Runtime -> Change runtime type -> GPU')""")))

# ---------- Cell 3: Mount Drive ----------
cells.append(cell(textwrap.dedent("""\
    # Mount Google Drive to persist downloaded audio + cached embeddings
    from google.colab import drive
    drive.mount('/content/drive')
    
    import os
    WORK_DIR = '/content/drive/MyDrive/asvspoof2019_pipeline'
    os.makedirs(WORK_DIR, exist_ok=True)
    print('Working directory:', WORK_DIR)""")))

# ---------- Cell 4: Download full dataset ----------
cells.append(cell(textwrap.dedent("""\
    import os, time
    from datasets import load_dataset, Audio
    from tqdm.auto import tqdm
    
    LA_ROOT = os.path.join(WORK_DIR, 'data', 'asvspoof2019LA', 'LA')
    PROTO_DIR = os.path.join(LA_ROOT, 'ASVspoof2019_LA_cm_protocols')
    os.makedirs(PROTO_DIR, exist_ok=True)
    
    # CRITICAL: Correct HF split -> local split mapping
    # HF train      -> LA_train (LA_T_* files) -> used for training only
    # HF validation -> LA_dev   (LA_D_* files) -> used for dev/early-stopping only
    # HF test       -> LA_eval  (LA_E_* files) -> used for final eval only
    SPLITS = [
        ('train',      'train', 'ASVspoof2019_LA_train', 'ASVspoof2019.LA.cm.train.trn.txt', 25380),
        ('validation', 'dev',   'ASVspoof2019_LA_dev',   'ASVspoof2019.LA.cm.dev.trl.txt',   24844),
        ('test',       'eval',  'ASVspoof2019_LA_eval',  'ASVspoof2019.LA.cm.eval.trl.txt',  71237),
    ]
    
    for hf_split, local_name, folder, proto_file, expected in SPLITS:
        audio_dir = os.path.join(LA_ROOT, folder, 'flac')
        os.makedirs(audio_dir, exist_ok=True)
        proto_path = os.path.join(PROTO_DIR, proto_file)
    
        existing = [f for f in os.listdir(audio_dir) if f.endswith('.flac')]
        existing_set = set(f[:-5] for f in existing)
        print(f'Split {local_name}: {len(existing)}/{expected} files already on disk')
    
        if len(existing) >= expected and os.path.exists(proto_path):
            print(f'  -> Already complete. Skipping download.')
            continue
    
        print(f'Downloading {local_name} from HF split="{hf_split}" (expected {expected} files)...')
        ds = load_dataset('Bisher/ASVspoof_2019_LA', split=hf_split, streaming=True)
        ds = ds.cast_column('audio', Audio(decode=False))
    
        lines_out = []
        saved = len(existing)
        t0 = time.time()
        pbar = tqdm(desc=local_name, total=expected, initial=saved, unit='file')
    
        for item in ds:
            fname = item.get('audio_file_name', item.get('path', ''))
            stem = fname[:-5] if fname.endswith('.flac') else fname
            spk = item.get('speaker_id', 'LA_0001')
            sid = item.get('system_id', '-')
            if sid is None or str(sid).strip() in ['', 'nan', 'None']:
                sid = '-'
            key = str(item.get('key', 'spoof')).strip().lower()
            if key not in ['bonafide', 'spoof']:
                key = 'spoof'
    
            lines_out.append(spk + ' ' + stem + ' - ' + sid + ' ' + key)
    
            if stem not in existing_set:
                out_path = os.path.join(audio_dir, stem + '.flac')
                with open(out_path, 'wb') as fout:
                    fout.write(item['audio']['bytes'])
                existing_set.add(stem)
            saved += 1
            pbar.update(1)
    
            if saved % 2000 == 0:
                elapsed = time.time() - t0
                rate = saved / max(elapsed, 1)
                rem = (expected - saved) / max(rate, 0.01)
                print(f'  {saved}/{expected} | {rate:.0f} f/s | ~{rem/60:.0f} min left')
                with open(proto_path + '.partial', 'w') as pf:
                    pf.write('\\n'.join(lines_out) + '\\n')
    
        pbar.close()
        with open(proto_path, 'w', encoding='utf-8') as fout:
            fout.write('\\n'.join(lines_out) + '\\n')
        if os.path.exists(proto_path + '.partial'):
            os.remove(proto_path + '.partial')
        elapsed = time.time() - t0
        print(f'Done {local_name}: {saved} files in {elapsed/60:.1f} min')
    
    print('\\nAll splits downloaded.')""")))

# ---------- Cell 5: Verify file prefix distribution ----------
cells.append(cell(textwrap.dedent("""\
    # CRITICAL VERIFICATION: Print file-prefix distribution per folder
    # Confirms HF split mapping is correct before extracting any embeddings
    import os
    from collections import Counter
    
    SPLIT_FOLDERS = [
        ('train', 'ASVspoof2019_LA_train'),
        ('dev',   'ASVspoof2019_LA_dev'),
        ('eval',  'ASVspoof2019_LA_eval'),
    ]
    
    print('=== FILE PREFIX DISTRIBUTION (confirms correct HF split mapping) ===')
    print()
    for local_name, folder in SPLIT_FOLDERS:
        audio_dir = os.path.join(LA_ROOT, folder, 'flac')
        if not os.path.exists(audio_dir):
            print(f'{local_name}: directory not found')
            continue
        files = [f for f in os.listdir(audio_dir) if f.endswith('.flac')]
        prefix_counts = Counter(f[:4] for f in files)
        total = len(files)
        print(f'{local_name.upper()} folder ({total} files):')
        for prefix, count in sorted(prefix_counts.items()):
            expected_prefix = {'train': 'LA_T', 'dev': 'LA_D', 'eval': 'LA_E'}[local_name]
            status = 'OK' if prefix == expected_prefix else 'WRONG'
            print(f'  {prefix}_*: {count} files [{status}]')
        print()
    
    print('=== PROTOCOL ENTRY COUNTS ===')
    PROTO_DIR = os.path.join(LA_ROOT, 'ASVspoof2019_LA_cm_protocols')
    for proto_file, split_name in [
        ('ASVspoof2019.LA.cm.train.trn.txt', 'train'),
        ('ASVspoof2019.LA.cm.dev.trl.txt',   'dev'),
        ('ASVspoof2019.LA.cm.eval.trl.txt',  'eval'),
    ]:
        path = os.path.join(PROTO_DIR, proto_file)
        if os.path.exists(path):
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            bonafide = sum(1 for l in lines if l.split()[-1] == 'bonafide')
            spoof = sum(1 for l in lines if l.split()[-1] == 'spoof')
            print(f'{split_name}: {len(lines)} entries ({bonafide} bonafide + {spoof} spoof)')
        else:
            print(f'{split_name}: protocol file NOT FOUND')
    
    print()
    print('Expected: train=25380, dev=24844, eval=71237')""")))

# ---------- Cell 6: Extract features (GPU batched) ----------
cells.append(cell(textwrap.dedent("""\
    import os, glob, numpy as np, torch, gc
    from tqdm.auto import tqdm
    from transformers import AutoFeatureExtractor, WavLMModel
    import torchaudio
    
    MODEL_NAME = 'microsoft/wavlm-base'
    FEAT_DIR = os.path.join(WORK_DIR, 'data', 'features')
    os.makedirs(FEAT_DIR, exist_ok=True)
    
    def get_model_tag(name):
        return name.split('/')[-1].replace('-', '_').lower()
    
    def parse_protocol(path):
        entries = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                entries.append({
                    'speaker': parts[0], 'filename': parts[1],
                    'label': parts[4].lower(),
                    'target': 1 if parts[4].lower() == 'bonafide' else 0
                })
        return entries
    
    def load_audio_16k(path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        return waveform.squeeze(0)
    
    print('Loading WavLM-base model...')
    ssl_model = WavLMModel.from_pretrained(MODEL_NAME)
    ssl_model.eval()
    for p in ssl_model.parameters():
        p.requires_grad = False
    ssl_model = ssl_model.to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model_tag = get_model_tag(MODEL_NAME)
    print('Model tag:', model_tag)
    print('Model on device:', next(ssl_model.parameters()).device)
    
    PROTO_DIR = os.path.join(LA_ROOT, 'ASVspoof2019_LA_cm_protocols')
    
    SPLIT_CONFIG = [
        ('train', 'ASVspoof2019_LA_train', 'ASVspoof2019.LA.cm.train.trn.txt'),
        ('dev',   'ASVspoof2019_LA_dev',   'ASVspoof2019.LA.cm.dev.trl.txt'),
        ('eval',  'ASVspoof2019_LA_eval',  'ASVspoof2019.LA.cm.eval.trl.txt'),
    ]
    
    for split_name, folder, proto_file in SPLIT_CONFIG:
        x_path = os.path.join(FEAT_DIR, f'X_{split_name}_{model_tag}.npy')
        y_path = os.path.join(FEAT_DIR, f'y_{split_name}_{model_tag}.npy')
    
        if os.path.exists(x_path) and os.path.exists(y_path):
            arr = np.load(x_path)
            print(f'{split_name}: cached embeddings found -> shape {arr.shape}. Skipping.')
            continue
    
        proto_path = os.path.join(PROTO_DIR, proto_file)
        entries = parse_protocol(proto_path)
        audio_dir = os.path.join(LA_ROOT, folder, 'flac')
    
        # Resolve physical audio paths
        valid = []
        for e in entries:
            p_flac = os.path.join(audio_dir, e['filename'] + '.flac')
            p_wav  = os.path.join(audio_dir, e['filename'] + '.wav')
            if os.path.exists(p_flac):
                e['path'] = p_flac
                valid.append(e)
            elif os.path.exists(p_wav):
                e['path'] = p_wav
                valid.append(e)
        print(f'{split_name}: {len(valid)}/{len(entries)} audio files resolved')
    
        embeddings, targets = [], []
        errors = 0
        BATCH_SIZE = 8  # Tune based on VRAM; T4 16GB can handle 16-32
    
        for i in tqdm(range(0, len(valid), BATCH_SIZE), desc=f'Extracting {split_name}'):
            batch = valid[i:i+BATCH_SIZE]
            batch_waveforms = []
            batch_targets = []
    
            for entry in batch:
                try:
                    wav = load_audio_16k(entry['path']).numpy()
                    batch_waveforms.append(wav)
                    batch_targets.append(entry['target'])
                except Exception as ex:
                    errors += 1
                    continue
    
            if not batch_waveforms:
                continue
    
            try:
                inputs = feature_extractor(
                    batch_waveforms,
                    sampling_rate=16000,
                    return_tensors='pt',
                    padding=True
                ).input_values.to(device)
    
                with torch.no_grad():
                    hidden = ssl_model(inputs).last_hidden_state  # (B, T, 768)
                    pooled = hidden.mean(dim=1)  # (B, 768)
    
                embeddings.append(pooled.cpu().numpy())
                targets.extend(batch_targets[:len(batch_waveforms)])
            except Exception as ex:
                print(f'  Batch error at i={i}: {ex}')
                errors += 1
                torch.cuda.empty_cache()
                continue
    
        if len(embeddings) == 0:
            print(f'ERROR: No embeddings extracted for {split_name}')
            continue
    
        X = np.vstack(embeddings).astype(np.float32)
        y = np.array(targets, dtype=np.int64)
        np.save(x_path, X)
        np.save(y_path, y)
        print(f'{split_name}: saved X={X.shape} y={y.shape} | errors={errors}')
        torch.cuda.empty_cache()
        gc.collect()
    
    print('Feature extraction complete.')""")))

# ---------- Cell 7: Verify embedding shapes ----------
cells.append(cell(textwrap.dedent("""\
    import numpy as np, os
    model_tag = 'wavlm_base'
    print('=== EMBEDDING SHAPES (must match 25380/24844/71237) ===')
    for split in ['train', 'dev', 'eval']:
        xp = os.path.join(FEAT_DIR, f'X_{split}_{model_tag}.npy')
        yp = os.path.join(FEAT_DIR, f'y_{split}_{model_tag}.npy')
        if os.path.exists(xp):
            X = np.load(xp); y = np.load(yp)
            bona = int((y==1).sum()); spoof = int((y==0).sum())
            print(f'{split}: X={X.shape}, y={y.shape} | bonafide={bona}, spoof={spoof}')
        else:
            print(f'{split}: MISSING')""")))

# ---------- Cell 8: Train MLP ----------
cells.append(cell(textwrap.dedent("""\
    import numpy as np, torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve
    import os
    
    # ---- EER ----
    def compute_eer(y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        frr = 1.0 - tpr
        try:
            eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            threshold = float(interp1d(fpr, thresholds)(eer))
        except Exception:
            idx = np.nanargmin(np.abs(frr - fpr))
            eer = fpr[idx]; threshold = thresholds[idx]
        return eer * 100.0, threshold
    
    # ---- MLP ----
    class DeepfakeMLP(nn.Module):
        def __init__(self, input_dim=768, hidden=256, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden//2, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    model_tag = 'wavlm_base'
    X_train = np.load(os.path.join(FEAT_DIR, f'X_train_{model_tag}.npy'))
    y_train = np.load(os.path.join(FEAT_DIR, f'y_train_{model_tag}.npy'))
    X_dev   = np.load(os.path.join(FEAT_DIR, f'X_dev_{model_tag}.npy'))
    y_dev   = np.load(os.path.join(FEAT_DIR, f'y_dev_{model_tag}.npy'))
    X_eval  = np.load(os.path.join(FEAT_DIR, f'X_eval_{model_tag}.npy'))
    y_eval  = np.load(os.path.join(FEAT_DIR, f'y_eval_{model_tag}.npy'))
    
    print(f'Train: {X_train.shape} | bonafide={int((y_train==1).sum())} spoof={int((y_train==0).sum())}')
    print(f'Dev:   {X_dev.shape}   | bonafide={int((y_dev==1).sum())} spoof={int((y_dev==0).sum())}')
    print(f'Eval:  {X_eval.shape}  | bonafide={int((y_eval==1).sum())} spoof={int((y_eval==0).sum())}')
    
    # BCEWithLogitsLoss with class-weight for 9:1 imbalance
    n_spoof = int((y_train==0).sum())
    n_bona  = int((y_train==1).sum())
    pos_weight = torch.tensor([n_spoof / max(n_bona, 1)], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train.astype('float32')))
    dev_ds   = TensorDataset(torch.from_numpy(X_dev).float(),   torch.from_numpy(y_dev.astype('float32')))
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    
    clf = DeepfakeMLP(input_dim=768, hidden=256, dropout=0.3).to(device)
    optimizer = optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    EPOCHS = 50
    best_dev_eer = float('inf')
    best_weights = None
    history = []
    
    print('\\nTraining MLP (768->256->128->1, 50 epochs, pos_weight=', round(pos_weight.item(),2), ')...')
    print(f'{"Epoch":>6} {"TrainLoss":>10} {"DevEER":>8} {"BestEER":>8}')
    print('-' * 40)
    
    for epoch in range(1, EPOCHS + 1):
        clf.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(clf(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        train_loss = total_loss / len(train_ds)
        scheduler.step()
    
        clf.eval()
        dev_scores, dev_true = [], []
        with torch.no_grad():
            for bx, by in dev_loader:
                probs = torch.sigmoid(clf(bx.to(device))).cpu().numpy()
                dev_scores.extend(probs)
                dev_true.extend(by.numpy())
        dev_eer, dev_th = compute_eer(np.array(dev_true), np.array(dev_scores))
    
        history.append({'epoch': epoch, 'train_loss': train_loss, 'dev_eer': dev_eer})
    
        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            best_weights = {k: v.clone() for k, v in clf.state_dict().items()}
    
        if epoch % 5 == 0 or epoch == 1:
            print(f'{epoch:>6} {train_loss:>10.4f} {dev_eer:>8.2f}% {best_dev_eer:>8.2f}%')
    
    print(f'\\nBest Dev EER: {best_dev_eer:.2f}%')
    clf.load_state_dict(best_weights)""")))

# ---------- Cell 9: Evaluate on full eval set ----------
cells.append(cell(textwrap.dedent("""\
    import numpy as np, torch
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    clf.eval()
    eval_tensor = torch.from_numpy(X_eval).float()
    
    # Run inference in batches to avoid OOM
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(eval_tensor), 512):
            batch = eval_tensor[i:i+512].to(device)
            probs = torch.sigmoid(clf(batch)).cpu().numpy()
            all_probs.extend(probs)
    
    eval_probs = np.array(all_probs)
    eval_eer, eval_threshold = compute_eer(y_eval, eval_probs)
    
    y_pred = (eval_probs >= 0.5).astype(int)
    acc    = accuracy_score(y_eval, y_pred) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_eval, eval_probs) * 100.0
    except:
        auc = 50.0
    
    print('=' * 55)
    print('FINAL RESULTS ON REAL ASVspoof 2019 LA EVAL SET (71,237 files)')
    print('=' * 55)
    print(f'  Equal Error Rate (EER):  {eval_eer:.2f}%  (threshold={eval_threshold:.4f})')
    print(f'  Accuracy:                {acc:.2f}%')
    print(f'  Precision (Bonafide):    {prec*100:.2f}%')
    print(f'  Recall (Bonafide):       {rec*100:.2f}%')
    print(f'  F1-Score:                {f1*100:.2f}%')
    print(f'  ROC-AUC:                 {auc:.2f}%')
    print()
    print(f'  Eval bonafide:  {int((y_eval==1).sum())}  |  Eval spoof: {int((y_eval==0).sum())}')
    if eval_eer < 2.0:
        print()
        print('[!] WARNING: EER < 2% - check for speaker/file leakage across splits.')
    elif eval_eer > 45.0:
        print()
        print('[!] NOTE: EER > 45% - model is near-random on eval. Possible causes:')
        print('    - Eval contains unseen TTS/VC systems (A07-A19) not in train (A01-A06)')
        print('    - MLP head too shallow for this embedding space')
        print('    - Consider fine-tuning backbone or using larger MLP')""")))

# ---------- Cell 10: Save checkpoint to Drive ----------
cells.append(cell(textwrap.dedent("""\
    import torch, os, numpy as np
    
    MODELS_DIR = os.path.join(WORK_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    ckpt_path = os.path.join(MODELS_DIR, 'best_mlp_wavlm_base.pt')
    torch.save({
        'model_state_dict': clf.state_dict(),
        'input_dim': 768,
        'hidden_dim': 256,
        'model_name': 'microsoft/wavlm-base',
        'eer': eval_eer,
        'threshold': eval_threshold,
        'history': history,
    }, ckpt_path)
    print('Checkpoint saved to Drive:', ckpt_path)
    
    # Print final training curve
    print()
    print('\\nTraining curve (every 5 epochs):')
    print(f'{"Epoch":>6}  {"TrainLoss":>10}  {"DevEER":>8}')
    for h in history:
        if h['epoch'] % 5 == 0 or h['epoch'] == 1:
            print(f"{h['epoch']:>6}  {h['train_loss']:>10.4f}  {h['dev_eer']:>8.2f}%")
    
    print()
    print('Next step: download best_mlp_wavlm_base.pt from Drive and copy to ./models/')
    print('Then run:  python predict.py --test')""")))

notebook = {
    'nbformat': 4, 'nbformat_minor': 0,
    'metadata': {
        'colab': {'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU'
    },
    'cells': cells
}

with open('colab_asvspoof_full_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('Notebook written: colab_asvspoof_full_pipeline.ipynb')
print('Upload to Colab at: https://colab.research.google.com')
print('Remember to: Runtime -> Change runtime type -> GPU (T4)')
