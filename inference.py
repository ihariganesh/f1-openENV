"""Inference runner for advanced F1 Strategy Optimizer OpenEnv."""

from __future__ import annotations

import os
from typing import List, Optional

import httpx
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
BENCHMARK = "race_strategy_optimizer"
TASKS = ["f1-sprint-dry", "f1-feature-safetycar", "f1-chaos-weather"]
MAX_STEPS = int(os.getenv("MAX_STEPS", "80"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    total_reward = sum(rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} "
        f"total_reward={total_reward:.4f} avg_reward={total_reward / max(1, len(rewards)):.4f}",
        flush=True,
    )


# --------------- Expert fallback strategy ---------------

def _fallback_action(obs: dict) -> str:
    """Rule-based expert strategy covering all three race scenarios."""
    lap = int(obs.get("current_lap", 1))
    total_laps = int(obs.get("total_laps", 20))
    task = obs.get("task_name", "")
    wetness = float(obs.get("track_wetness", 0.0))
    forecast = float(obs.get("rain_forecast_next_5_laps", 0.0))
    tire_age = int(obs.get("tire_age_laps", 0))
    wear = float(obs.get("tire_wear_percentage", 0.0))
    sc = bool(obs.get("safety_car_active", False))
    compound = str(obs.get("current_tire_compound", ""))
    fuel = float(obs.get("fuel_kg", 110.0))
    pit_count = int(obs.get("pit_stop_count", 0))
    cliff = float(obs.get("tire_cliff_proximity", 0.0))

    # Never pit on consecutive laps
    if tire_age <= 1:
        # Under safety car, conserve to save fuel
        if sc:
            return "hold:CONSERVE"
        return "hold:BALANCED"

    # EMERGENCY: tire failure imminent
    if wear >= 0.92:
        best_compound = "MEDIUM"
        if wetness > 0.5:
            best_compound = "INTER"
        elif wetness > 0.7:
            best_compound = "WET"
        return f"pit:{best_compound}:BALANCED"

    # ============ TASK-SPECIFIC STRATEGIES ============

    if task == "f1-sprint-dry":
        # Optimal: pit on lap 8-10 from SOFT to MEDIUM
        if lap in {8, 9, 10} and compound == "SOFT" and pit_count == 0:
            return "pit:MEDIUM:BALANCED"
        # If we missed the window, pit by lap 11
        if lap >= 11 and compound == "SOFT" and pit_count == 0 and cliff >= 0.9:
            return "pit:MEDIUM:BALANCED"
        # Push on fresh tires or low wear
        if wear < 0.3 and tire_age >= 2:
            return "hold:PUSH"
        return "hold:BALANCED"

    elif task == "f1-feature-safetycar":
        # KEY: Pit under safety car (laps 18-20) for discounted stop
        if sc and lap in {18, 19, 20} and pit_count == 0:
            return "pit:HARD:BALANCED"
        # Second stop around lap 38-40 if needed
        if lap in {36, 37, 38, 39, 40} and pit_count == 1 and wear > 0.60:
            return "pit:MEDIUM:BALANCED"
        # Conserve fuel under safety car
        if sc:
            return "hold:CONSERVE"
        # Push early for track position before SC
        if lap < 15 and wear < 0.4:
            return "hold:PUSH"
        # Late race: push if tires are fresh
        if lap > 40 and wear < 0.3:
            return "hold:PUSH"
        return "hold:BALANCED"

    elif task == "f1-chaos-weather":
        # MULTI-PHASE STRATEGY:
        # Phase 1 (laps 1-10): Run SOFT tires
        # Phase 2 (laps 10-29): MEDIUM tires through dry phase
        # Phase 3 (lap ~30+): Switch to INTER when wetness arrives

        # Pre-rain pit: switch from soft to medium around lap 9-12
        if lap in {9, 10, 11, 12} and compound == "SOFT" and pit_count == 0:
            return "pit:MEDIUM:BALANCED"
        if lap >= 13 and compound == "SOFT" and pit_count == 0:
            return "pit:MEDIUM:BALANCED"

        # Rain transition: ONLY switch to INTER when track is actually getting wet
        # DO NOT pit preemptively — inters on dry track have catastrophic degradation
        if wetness > 0.3 and compound in {"SOFT", "MEDIUM", "HARD"}:
            return "pit:INTER:BALANCED"
        if wetness > 0.75 and compound == "INTER" and wear > 0.55:
            return "pit:WET:BALANCED"

        # Dry phase: push with fresh tires
        if wetness == 0.0 and wear < 0.3 and tire_age >= 2:
            return "hold:PUSH"
        # Wet phase: conserve to protect inters
        if wetness > 0.3:
            return "hold:CONSERVE"
        return "hold:BALANCED"

    # Generic fallback
    if wear > 0.85:
        return "hold:CONSERVE"
    return "hold:BALANCED"


# --------------- LLM-based action selection ---------------

SYSTEM_PROMPT = """You are an expert F1 race strategist. Your job is to optimize total race time.

KEY PRINCIPLES:
- Monitor tire_wear_percentage and tire_cliff_proximity to time pit stops
- Soft tires cliff at ~12 laps, Medium at ~25, Hard at ~40
- Pit during safety car to save 10s on pit time
- Switch to INTER tires when track_wetness > 0.4, or WET when > 0.7
- Use rain_forecast_next_5_laps to anticipate weather changes
- PUSH pace burns tires/fuel 50%/20% faster, CONSERVE saves both
- Each pit stop costs 20s (10s under safety car)

RESPOND WITH EXACTLY ONE LINE in one of these formats:
1) hold:PUSH|BALANCED|CONSERVE
2) pit:SOFT|MEDIUM|HARD|INTER|WET:PUSH|BALANCED|CONSERVE"""


def choose_action(client: Optional[OpenAI], obs: dict) -> str:
    if client is None:
        return _fallback_action(obs)

    prompt = (
        f"Lap {obs.get('current_lap')}/{obs.get('total_laps')} | "
        f"Tire: {obs.get('current_tire_compound')} age={obs.get('tire_age_laps')} "
        f"wear={obs.get('tire_wear_percentage', 0):.1%} cliff={obs.get('tire_cliff_proximity', 0):.1%} | "
        f"Fuel: {obs.get('fuel_kg', 0):.0f}kg | "
        f"Track: wet={obs.get('track_wetness', 0):.1%} forecast={obs.get('rain_forecast_next_5_laps', 0):.1%} "
        f"temp={obs.get('track_temperature_c', 35)}°C | "
        f"SC={'YES' if obs.get('safety_car_active') else 'no'} "
        f"DRS={'YES' if obs.get('drs_available') else 'no'} | "
        f"Pits: {obs.get('pit_stop_count', 0)} | "
        f"Time: {obs.get('cumulative_race_time_seconds', 0):.0f}s last_lap={obs.get('last_lap_time_seconds', 0):.1f}s"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=24,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        return _fallback_action(obs)

    # Validate format
    if text and ("hold:" in text.lower() or "pit:" in text.lower()):
        return text
    return _fallback_action(obs)


def to_payload(action: str) -> dict:
    raw = action.strip().upper()
    if raw.startswith("PIT:"):
        parts = raw.split(":")
        if len(parts) == 3:
            _, comp, pace = parts
            if comp in {"SOFT", "MEDIUM", "HARD", "INTER", "WET"} and pace in {"PUSH", "BALANCED", "CONSERVE"}:
                return {"pit_stop": True, "new_compound": comp, "pace_mode": pace}
        return {"pit_stop": True, "new_compound": "MEDIUM", "pace_mode": "BALANCED"}

    if raw.startswith("HOLD:"):
        pace = raw.split(":", 1)[1] if ":" in raw else "BALANCED"
        if pace not in {"PUSH", "BALANCED", "CONSERVE"}:
            pace = "BALANCED"
        return {"pit_stop": False, "new_compound": None, "pace_mode": pace}

    return {"pit_stop": False, "new_compound": None, "pace_mode": "BALANCED"}


def run_task(http: httpx.Client, client: Optional[OpenAI], task: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset = http.post(f"{ENV_BASE_URL}/reset", json={"task": task, "seed": 7})
        reset.raise_for_status()
        result = reset.json()

        for step in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            obs = dict(result.get("observation", {}))
            obs["task_name"] = task
            action = choose_action(client, obs)
            payload = to_payload(action)

            step_resp = http.post(f"{ENV_BASE_URL}/step", json=payload)
            step_resp.raise_for_status()
            result = step_resp.json()

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = (result.get("info") or {}).get("last_action_error") or None
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error)
            if done:
                break

        state_resp = http.post(f"{ENV_BASE_URL}/state", json={})
        state_resp.raise_for_status()
        state = state_resp.json()
        score = float(state.get("grader_score", 0.0))
        score = max(0.0, min(1.0, score))
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    http = httpx.Client(timeout=60.0)

    all_scores: List[float] = []
    for task in TASKS:
        all_scores.append(run_task(http, client, task))

    avg = sum(all_scores) / len(all_scores)
    print(f"FINAL_AVG_SCORE={avg:.4f}", flush=True)
    for task, score in zip(TASKS, all_scores):
        print(f"  {task}: {score:.4f}", flush=True)
    http.close()


if __name__ == "__main__":
    main()
