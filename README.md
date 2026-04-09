---
title: F1 Race Strategy Optimizer
emoji: 🏎️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# 🏎️ F1 Race Strategy Optimizer — OpenEnv Hackathon

An **interactive F1 race strategy environment** where AI agents learn to make
real-time pit stop, tire compound, and pace-mode decisions across three
progressively harder races. Built for the
[Hugging Face OpenEnv Hackathon](https://huggingface.co/spaces/open-env/leaderboard).

## 🔗 Live Demo

**HF Space:** [ihariganesh/f1-race-stratergy](https://huggingface.co/spaces/ihariganesh/f1-race-stratergy)

---

## 🏁 Tasks

| Task | Difficulty | Laps | Key Challenge |
|------|-----------|------|---------------|
| `f1-sprint-dry` | Easy | 20 | Pit timing — soft tires cliff at lap 10 |
| `f1-feature-safetycar` | Medium | 50 | Exploit safety car window (laps 18-20) for discounted pit stops |
| `f1-chaos-weather` | Hard | 60 | Dynamic dry→wet transition at lap 30, requiring preemptive tire swaps |

OpenEnv scenario aliases available for stress testing:

- `easy-one-stop` (easy)
- `medium-strategy` (medium)
- `hard-safety-car` (hard)
- `ultra-chaos` (hard, overlapping weather + safety car)

## 🧪 Observation Space (15 fields)

| Field | Type | Description |
|-------|------|-------------|
| `current_lap` | int | Current lap number |
| `total_laps` | int | Total laps in race |
| `current_tire_compound` | str | SOFT / MEDIUM / HARD / INTER / WET |
| `tire_age_laps` | int | Laps since last pit |
| `tire_wear_percentage` | float | 0.0 → 1.0 (puncture at 1.0) |
| `fuel_kg` | float | Remaining fuel (starts at 110 kg) |
| `track_wetness` | float | 0.0 → 1.0 |
| `rain_forecast_next_5_laps` | float | Max wetness in next 5 laps |
| `safety_car_active` | bool | Safety car deployed this lap |
| `last_lap_time_seconds` | float | Previous lap time |
| `cumulative_race_time_seconds` | float | Total race time so far |
| `pit_stop_count` | int | Number of pit stops made |
| `drs_available` | bool | DRS zone available on this lap |
| `track_temperature_c` | float | Track surface temperature (drops in rain) |
| `tire_cliff_proximity` | float | 0.0 → 1.0, how close to the performance cliff |

## 🎮 Action Space

```json
{
  "pit_stop": true,
  "new_compound": "MEDIUM",
  "pace_mode": "BALANCED"
}
```

- **pit_stop**: `true` = enter pits (costs 20s, or 10s under safety car)
- **new_compound**: Required when pitting. One of: SOFT, MEDIUM, HARD, INTER, WET
- **pace_mode**: PUSH (-0.5s, +50% wear, +20% fuel) | BALANCED | CONSERVE (+1.0s, -50% wear, -20% fuel)

## 📊 Reward Signal

Per-step reward is clipped to **0.0–1.0** for stable learning and evaluator compliance,
while still providing dense partial-progress feedback every lap.

Core components:

- `time_reward`: starts near `1.0` for competitive laps and decreases with slower lap times
- `sc_pit_bonus`: +0.1 when pitting under safety car
- `wrong_tire_penalty`: applied every lap when on slicks in damp/wet conditions
- `puncture_penalty`: applied when tire wear reaches puncture range
- action/fuel penalties: invalid pit actions and fuel depletion are penalized
- terminal progress bonus: up to +0.2 at episode end, scaled by distance to optimal total time

Final per-step formula:

`reward = clip_0_1(raw_reward)`

This keeps rewards interpretable and bounded while preserving trajectory-level guidance.

## 🏆 Grading

Score = `max(0, 1 − (agent_time − optimal_time) / tolerance)`

| Task | Tolerance |
|------|-----------|
| Sprint | 12s |
| Feature | 22s |
| Chaos | 35s |

## 🧬 Tire Physics

Non-linear degradation model with compound-specific performance cliffs:

- **SOFT**: Fast (-1.2s offset), cliff at lap 10, exponential degradation after
- **MEDIUM**: Balanced (-0.5s), cliff at lap 25
- **HARD**: Durable (0.0s), cliff at lap 40
- **INTER**: Wet specialist (+0.2s), catastrophic in dry
- **WET**: Heavy wet (+0.5s), rapid wear if not sufficiently wet

## 🚀 Quick Start

```bash
git clone https://github.com/your-user/f1-race-strategy-optimizer
cd f1-race-strategy-optimizer
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### API Endpoints

```bash
# Health check
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "f1-sprint-dry", "seed": 7}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"pit_stop": false, "new_compound": null, "pace_mode": "BALANCED"}'

# Get current state
curl -X POST http://localhost:7860/state -H "Content-Type: application/json" -d '{}'
```

### Run Tests

```bash
PYTHONPATH=. python -m pytest -v
```

### Run Inference Agent

```bash
set -a
source .env.example
# Replace API_KEY with a valid key before running.
set +a

export API_BASE_URL=https://router.huggingface.co/v1
export API_KEY=<your_proxy_or_router_key>
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:7860

# Reproducible baseline across fixed seeds (default: 7,13,21)
python -u inference.py

# Optional overrides
# export INFERENCE_SEEDS=7,13,21,42
# export BASELINE_REPORT_PATH=artifacts/baseline_inference_report.json
```

The baseline runner writes a reproducible report JSON containing per-task/per-seed
scores and final averages.

### One-Command Local Integration Check (Canonical 3 Tasks)

This command starts the local FastAPI environment, runs `inference.py` against
the canonical easy/medium/hard tasks, writes a report, and then stops the server:

```bash
bash scripts/local_integration_check.sh
```

Optional overrides:

```bash
INFERENCE_SEEDS=7,13 MAX_STEPS=80 BASELINE_REPORT_PATH=artifacts/local_integration_report.json bash scripts/local_integration_check.sh
```

### Baseline Scores (Reproducible)

Use fixed seeds to keep scores reproducible:

```bash
INTEGRATION_FORCE_FALLBACK=1 INFERENCE_SEEDS=7,13,21 BASELINE_REPORT_PATH=artifacts/baseline_inference_report_3tasks.json bash scripts/local_integration_check.sh
```

This generates a JSON artifact with per-task scores for:

- `f1-sprint-dry` (easy)
- `f1-feature-safetycar` (medium)
- `f1-chaos-weather` (hard)

Latest generated result (`artifacts/baseline_inference_report_3tasks.json`):

| Task | Average Score (Seeds: 7, 13, 21) |
|------|----------------------------------|
| `f1-sprint-dry` | `1.000000` |
| `f1-feature-safetycar` | `0.939409` |
| `f1-chaos-weather` | `1.000000` |
| **Final average** | **`0.979803`** |

The inference runner uses the OpenAI Python client and reads credentials from `API_BASE_URL` and `API_KEY`.
For hackathon evaluation, this is required so calls are observed on the provided LiteLLM proxy key.

### Clean Container + HF Deploy Verification

Build and run the container locally:

```bash
docker build -t opp-eval-check .
cid=$(docker run -d -p 8786:7860 opp-eval-check)
sleep 5
curl -fsS http://127.0.0.1:8786/health
docker rm -f "$cid"
```

Deploy/update the Hugging Face Space:

```bash
export HF_SPACE_ID=ihariganesh/f1-race-stratergy
export HF_TOKEN=hf_xxx
./.venv/bin/python scripts/deploy_to_hf_space.py
curl -fsS https://ihariganesh-f1-race-stratergy.hf.space/health
```

### Functional + Non-Functional Compliance Matrix

- Real-world simulation: race strategy optimization with pit timing, weather adaptation, and fuel/tire trade-offs.
- OpenEnv spec: typed Pydantic models (`ObservationSpace`, `ActionSpace`, `Reward`), `reset`/`step`/`state`, and `openenv.yaml` metadata.
- 3 graded tasks: canonical easy/medium/hard tasks with deterministic programmatic grader in `tasks.py`.
- Meaningful reward: dense per-step reward with progress signals and penalties for wrong-tire, puncture risk, invalid actions, and fuel depletion.
- Baseline inference: OpenAI client (`openai` package), reads `API_BASE_URL` + `API_KEY`, writes reproducible score report JSON.
- HF Space deployment: Docker-based Space with metadata tag `openenv`.
- Containerized execution: validated via `docker build` and runtime health checks.
- Documentation: includes environment motivation, task definitions, action/observation spaces, setup, usage, and baseline scoring workflow.

Mode:

- Proxy mode (required for evaluation): uses `API_BASE_URL` and `API_KEY`.

### Required Submission Variables

The following variables are mandatory for Round 1 evaluation:

- `API_BASE_URL`
- `API_KEY`
- `MODEL_NAME`

For local runs, place them in `.env` (or export in shell).
For Hugging Face Space, set them in **Space Settings → Variables and secrets**.

You can also set them programmatically:

```bash
export HF_SPACE_ID=ihariganesh/f1-race-stratergy
export HF_TOKEN=hf_xxx
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python scripts/set_space_secrets.py
```

### Pre-Submission Validator

The validator requires the Space URL as an argument:

```bash
bash ./validate-submission.sh 'https://ihariganesh-f1-race-stratergy.hf.space' .
```

If you run it without arguments, it prints:

```text
Usage: ./validate-submission.sh <ping_url> [repo_dir]
```

## 📁 Project Structure

```
├── app/main.py        # FastAPI server + interactive UI
├── env.py             # Core environment logic & reward shaping
├── models.py          # Pydantic schemas (observation, action, reward)
├── tasks.py           # Task configs, tire physics, grading
├── inference.py       # LLM agent + expert fallback strategy
├── tests/             # Determinism, contract, grader tests
├── Dockerfile         # Production container
└── src/               # Alternate package implementation (not used by deployed app)
```

### Runtime Path Clarification

Submission runtime uses the root modules imported by `app/main.py`:

- `env.py`
- `models.py`
- `tasks.py`

The `src/race_strategy_optimizer/` package is kept for package-style development,
but it is not the entry path used by the deployed FastAPI app.

## 🏗️ Architecture

```
Agent ──▶ POST /reset ──▶ F1StrategyEnv.reset()
      ──▶ POST /step  ──▶ F1StrategyEnv.step(action)
      ◀── { observation, reward, done, info }
      ──▶ POST /state ──▶ Full internal state + grader score
```

---

**License:** MIT
