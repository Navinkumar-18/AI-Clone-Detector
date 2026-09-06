# VoiceGuard Evaluation

## Purpose

Evaluate the **exact production model** (`garystafford/wav2vec2-deepfake-voice-detector`) with **identical preprocessing** to what `backend.py` uses.

## Scripts

### `run_production_evaluation.py`

Runs inference on the ASVspoof 2019 LA dataset and computes full metrics.

```bash
python evaluation/run_production_evaluation.py --data_dir ./data/asvspoof2019LA
python evaluation/run_production_evaluation.py --data_dir ./data/asvspoof2019LA --max_clips 50
```

### `generate_metrics_report.py`

Generates a formatted markdown report from evaluation results.

```bash
python evaluation/generate_metrics_report.py --results evaluation/results.json
```

## Required Dataset

ASVspoof 2019 LA (Logical Access) eval split.

Expected location: `./data/asvspoof2019LA/`

If the dataset is not available, the scripts report:

```
NOT AVAILABLE — dataset was not present at the expected path.
```

## Important Notes

- Do NOT fabricate evaluation output.
- Do NOT call softmax outputs "calibrated probabilities" unless calibration has been performed.
- Report only genuinely generated metrics.
- The committed threshold report (threshold_report.txt) records a 100% false-positive rate at the reviewed threshold. A complete threshold sweep must be regenerated before making claims about all thresholds.
