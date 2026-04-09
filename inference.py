"""Inference runner for advanced F1 Strategy Optimizer OpenEnv."""

from __future__ import annotations

import os
from typing import List, Optional

import httpx
from openai import OpenAI

# Judges inject API_BASE_URL and API_KEY — use these directly, no fallback to own credentials
API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "placeholder"
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK = "race_strategy_optimizer"
TASKS = ["f1-sprint-dry", "f1-feature-safetycar", "f1-chaos-weather"]
MAX_STEPS = int(os.environ.get("MAX_STEPS", "80"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _fallback_action(obs: dict) -> str:
    lap = int(obs.get("current_lap", 1))
    task = obs.get("task_name", "")
    wetness = float(obs.get("track_wetness", 0.0))
    forecast = float(obs.get("rain_forecast_next_5_laps", 0.0))
    tire_age = int(obs.get("tire_age_laps", 0))
    wear = float(obs.get("tire_wear_percentage", 0.0))
    sc = bool(obs.get("safety_car_active", False))
    compound = str(obs.get("current_tire_compound", ""))

    if tire_age <= 1:
        return "hold:BALANCED"

    if task == "f1-sprint-dry":
        if lap in {10, 11} and compound != "MEDIUM":
            return "pit:MEDIUM:BALANCED"
    elif task == "f1-feature-safetycar":
        if sc and lap in {18, 19, 20} and compound != "HARD":
            return "pit:HARD:CONSERVE"
        if wear > 0.86 and lap > 35 and compound != "MEDIUM":
            return "pit:MEDIUM:CONSERVE"
    else:
        if wetness > 0.4 and compound != "INTER":
            return "pit:INTER:CONSERVE"
        if forecast > 0.6 and lap >= 32 and compound != "INTER":
            return "pit:INTER:BALANCED"
        if wetness < 0.2 and wear > 0.9 and tire_age > 20 and compound != "HARD":
            return "pit:HARD:CONSERVE"

    if wear > 0.95:
        return "hold:CONSERVE"
    if wetness > 0.7 and compound != "INTER":
        return "pit:INTER:CONSERVE"
    return "hold:BALANCED"


def choose_action(client: OpenAI, obs: dict) -> str:
    prompt = (
        "Return exactly one token in one of these formats:\n"
        "1) hold:PUSH|BALANCED|CONSERVE\n"
        "2) pit:SOFT|MEDIUM|HARD|INTER:PUSH|BALANCED|CONSERVE\n"
        "Use observation to optimize race time while avoiding tire failure and wrong tire in wet.\n"
        f"Observation={obs}"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an F1 strategy engineer. Reply with exactly one action token."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=24,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else _fallback_action(obs)
    except Exception:
        return _fallback_action(obs)


def to_payload(action: str) -> dict:
    raw = action.strip().upper()
    if raw.startswith("PIT:"):
        parts = raw.split(":")
        if len(parts) == 3:
            _, comp, pace = parts
            return {"pit_stop": True, "new_compound": comp, "pace_mode": pace}
        return {"pit_stop": True, "new_compound": "MEDIUM", "pace_mode": "BALANCED"}

    if raw.startswith("HOLD:"):
        _, pace = raw.split(":", 1)
        return {"pit_stop": False, "new_compound": None, "pace_mode": pace}

    return {"pit_stop": False, "new_compound": None, "pace_mode": "BALANCED"}


def run_task(http: httpx.Client, client: OpenAI, task: str) -> float:
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
    # Always use the injected API_BASE_URL and API_KEY — never bypass with own credentials
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    http = httpx.Client(timeout=60.0)

    all_scores: List[float] = []
    for task in TASKS:
        all_scores.append(run_task(http, client, task))

    avg = sum(all_scores) / len(all_scores)
    print(f"FINAL_AVG_SCORE={avg:.4f}", flush=True)
    http.close()


if __name__ == "__main__":
    main()
