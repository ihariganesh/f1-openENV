# F1 Race Strategy Optimizer - Human Evaluation Rerun Report

Date: 2026-04-11
Target: https://ihariganesh-f1-race-stratergy.hf.space
Scope: API checks, reward signal checks, grader checks, browser UI checks, disqualification sanity checks

## Executive Summary

1. API tests: Pass
2. Reward checks: Pass (formatting note on `lap_reward_breakdown`)
3. Grader checks: Pass (with baseline-policy caveat documented)
4. Browser UI availability: Pass
5. Disqualification sanity checks: Pass with one caution (same as prior run)

## 1) API Checks

### 1.1 Health
- Request: `GET /health`
- Result: HTTP 200
- Body: `{"status":"ok","benchmark":"race_strategy_optimizer"}`
- Status: Pass

### 1.2 Reset (easy task)
- Request: `POST /reset` with `{"task":"f1-sprint-dry","seed":7}`
- Result: HTTP 200
- Observation field count: 15
- Required fields present: `fuel_kg`, `pit_stop_count`, `tire_cliff_proximity`, `cumulative_race_time_seconds`, `drs_available`, `track_temperature_c`
- Status: Pass

### 1.3 Step (hold action)
- Request: `POST /step` with `{"pit_stop":false,"new_compound":null,"pace_mode":"BALANCED"}`
- Result: HTTP 200
- Reward: `0.9790000000000001` (within [0.0, 1.0])
- `info.lap_reward_breakdown`: present
- Status: Pass

### 1.4 Step (invalid pit, no compound)
- Request: `POST /step` with `{"pit_stop":true,"new_compound":null,"pace_mode":"BALANCED"}`
- Result: HTTP 422 with validation error
- Server remained healthy after request
- Status: Pass

### 1.5 State
- Request: `POST /state` with `{}`
- Result: HTTP 200
- Keys present: `grader_score`, `optimal_total_time`, `pit_laps`, `track_wetness`
- Status: Pass

### 1.6 Tasks list
- Request: `GET /tasks`
- Result: HTTP 200
- Task count: 7
- Required tasks present:
  - `f1-sprint-dry` (easy)
  - `f1-feature-safetycar` (medium)
  - `f1-chaos-weather` (hard)
- Status: Pass

## 2) Reward Signal Checks

### 2.1 Good lap vs bad lap delta
- Good reward: `0.9790000000000001`
- Bad reward (chaos-weather, slicks into wet phase): `0.0`
- Delta: `0.9790000000000001` (> 0.1)
- Status: Pass

### 2.2 Safety car pit bonus
- Scenario: `f1-feature-safetycar`, pit during SC window with HARD
- Reward breakdown includes: `sc_pit_bonus=0.3`
- Status: Pass

### 2.3 Reward variation over 20 laps
- Min reward: `0.02497078489599991`
- Max reward: `0.9825`
- Range: `0.9575292151040001` (> 0.05)
- Status: Pass

## 3) Grader Checks

### 3.1 Baseline no-pit policy (BALANCED hold)
- `f1-sprint-dry`: `0.0001`
- `f1-feature-safetycar`: `0.387273`
- `f1-chaos-weather`: `0.0001`
- Distinct values: 2
- Note: This reproduces SCORE_EPS floor behavior for weak policy choices.

### 3.2 Determinism (same seed, same task)
- Task: `f1-feature-safetycar`, seed 7
- Run 1: `0.387273`
- Run 2: `0.387273`
- Exact match: true
- Status: Pass

### 3.3 Stronger policy distinctness
- `f1-sprint-dry`: `0.446667`
- `f1-feature-safetycar`: `0.5`
- `f1-chaos-weather`: `0.9999`
- Distinct values: 3
- Status: Pass

### 3.4 Hard-task challenge delta
- Good chaos strategy score: `0.9999`
- Bad chaos baseline score: `0.0001`
- Delta: `0.9998`
- Status: Pass

## 4) Browser UI Checks

- Opened root URL in real browser session (Playwright)
- Page title: `F1 Race Strategy Optimizer`
- UI sections present in rendered page text: `Episode Control`, `Step Action`, `Response`
- Console note: one non-blocking 404 for `/favicon.ico`
- Status: Pass

## 5) Disqualification Sanity Checks

1. Environment deploys and responds: Pass
2. Baseline inference script exists and has logging markers: Pass
   - [START] at line 41
   - [STEP] at line 47
   - [END] at line 55
3. Graders not always same score: Pass with caveat
   - Weak baseline policy can floor two tasks to `0.0001`
   - Task-aware policy demonstrates clear score separation across tasks

## Notes for Judges

- `lap_reward_breakdown` is currently returned as a string (for example: `time=0.828,sc_pit_bonus=0.3,...`) rather than a JSON object.
- This does not block validation because the signal is present and parseable.
- If desired for presentation quality, this can be converted to structured JSON in a future update.
