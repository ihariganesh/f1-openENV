# Round 1 Submission Report

## Deployment

- Space ID: iHariganesh/f1-race-stratergy
- Space URL: https://ihariganesh-f1-race-stratergy.hf.space
- Runtime status: live and responding on `POST /reset`

## Validator Output

Command:

```bash
source .venv/bin/activate && ./validate-submission.sh https://ihariganesh-f1-race-stratergy.hf.space .
```

Result summary:

- Step 1/3 ping `/reset`: PASSED
- Step 2/3 `docker build`: PASSED
- Step 3/3 `openenv validate`: PASSED
- Final: All checks passed.

## Live Inference Runs

Command used:

```bash
source .venv/bin/activate
for t in pit-window-easy safety-car-medium weather-shift-hard; do
  ENV_BASE_URL='https://ihariganesh-f1-race-stratergy.hf.space' RACE_TASK="$t" python inference.py
done
```

### Task: pit-window-easy

- END line: `[END] success=true steps=12 score=0.666 rewards=0.50,0.50,0.50,0.62,0.62,0.50,0.50,0.50,0.50,0.50,0.50,0.62`
- Log file: artifacts/inference_pit-window-easy.log

### Task: safety-car-medium

- END line: `[END] success=true steps=15 score=0.531 rewards=0.50,0.50,0.50,0.50,0.50,0.62,0.59,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.25`
- Log file: artifacts/inference_safety-car-medium.log

### Task: weather-shift-hard

- END line: `[END] success=false steps=16 score=0.267 rewards=0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.12`
- Log file: artifacts/inference_weather-shift-hard.log

## Notes

- Structured logging format is preserved with `[START]`, `[STEP]`, and `[END]` lines.
- The Space was initially in config error due to missing README metadata, then fixed and redeployed.
- Token should be rotated after submission process completes.
