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

## 🧪 Observation Space (16 fields)

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

## 📊 Reward Signal (11 components)

| # | Component | Range | Description |
|---|-----------|-------|-------------|
| 1 | Pace reward | 0 – 0.15 | Better lap times → higher reward |
| 2 | Puncture penalty | -0.8 | Tire wear reaches 100% |
| 3 | Wrong tire penalty | -0.6 | Slicks in rain or wets on dry |
| 4 | Fuel depletion | -1.0 | Running out of fuel |
| 5 | SC pit bonus | +0.3 | Pitting under safety car |
| 6 | Weather preemption | +0.25 | Fitting rain tires before rain hits |
| 7 | Tire health bonus | +0.05 | Maintaining wear in 20-70% sweet spot |
| 8 | Over-extended stint | -0.1 | Running >85% wear without pitting |
| 9 | Excessive pitting | -0.15 | More than 3 pit stops |
| 10 | DRS push bonus | +0.05 | Pushing when DRS is available |
| 11 | SC conserve bonus | +0.08 | Conserving under safety car |

**Terminal bonus:** +2.0 (≥90% score), +1.0 (≥70%), +0.5 (≥50%)

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
pip install -r requirements.txt
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
pytest tests/ -v
```

### Run Inference Agent

```bash
export HF_TOKEN=hf_...
export ENV_BASE_URL=http://localhost:7860
python inference.py
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
└── requirements.txt
```

## 🏗️ Architecture

```
Agent ──▶ POST /reset ──▶ F1StrategyEnv.reset()
      ──▶ POST /step  ──▶ F1StrategyEnv.step(action)
      ◀── { observation, reward, done, info }
      ──▶ POST /state ──▶ Full internal state + grader score
```

---

**License:** MIT
