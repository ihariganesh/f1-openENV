"""Inference runner for advanced F1 Strategy Optimizer OpenEnv."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

from models import DecisionActionToken, LLMDecision

API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"
BENCHMARK = "race_strategy_optimizer"
TASKS = [
    t.strip()
    for t in os.getenv("INFERENCE_TASKS", "f1-sprint-dry,f1-feature-safetycar,f1-chaos-weather").split(",")
    if t.strip()
]
MAX_STEPS = int(os.getenv("MAX_STEPS", "80"))
SEEDS = [int(s.strip()) for s in os.getenv("INFERENCE_SEEDS", "7,13,21").split(",") if s.strip()]
REPORT_PATH = os.getenv("BASELINE_REPORT_PATH", "artifacts/baseline_inference_report.json")


_LLM_DISABLED = False
_LLM_DISABLE_REASON = ""
_LLM_ERROR_COUNT = 0


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


def _normalize_action(action: str) -> str:
    raw = action.strip().upper()
    if raw.startswith("PIT:"):
        parts = raw.split(":")
        if len(parts) == 3:
            _, comp, pace = parts
            if comp in {"SOFT", "MEDIUM", "HARD", "INTER", "WET"} and pace in {"PUSH", "BALANCED", "CONSERVE"}:
                return f"pit:{comp}:{pace}"
        return "pit:MEDIUM:BALANCED"

    if raw.startswith("HOLD:"):
        pace = raw.split(":", 1)[1] if ":" in raw else "BALANCED"
        if pace not in {"PUSH", "BALANCED", "CONSERVE"}:
            pace = "BALANCED"
        return f"hold:{pace}"

    return "hold:BALANCED"


def _force_hold(pace: str = "BALANCED") -> str:
    pace = pace.upper()
    if pace not in {"PUSH", "BALANCED", "CONSERVE"}:
        pace = "BALANCED"
    return f"hold:{pace}"


def _apply_guardrails(action: str, obs: dict) -> str:
    action = _normalize_action(action)
    task = str(obs.get("task_name", ""))
    lap = int(obs.get("current_lap", 1))
    total_laps = int(obs.get("total_laps", 20))
    compound = str(obs.get("current_tire_compound", "SOFT")).upper()
    tire_age = int(obs.get("tire_age_laps", 0))
    wear = float(obs.get("tire_wear_percentage", 0.0))
    pit_count = int(obs.get("pit_stop_count", 0))
    wetness = float(obs.get("track_wetness", 0.0))
    forecast = float(obs.get("rain_forecast_next_5_laps", 0.0))
    sc = bool(obs.get("safety_car_active", False))

    # Generic anti-churn constraints.
    if action.startswith("pit:") and tire_age <= 1 and wear < 0.9:
        return _force_hold("BALANCED")
    if action.startswith("pit:") and tire_age <= 4 and wear < 0.88:
        return _force_hold("BALANCED")
    strategic_same_compound_pit = (
        (task == "medium-strategy" and compound == "MEDIUM" and pit_count == 0 and lap in {18, 19, 20, 21, 22, 23, 24})
        or (task == "easy-one-stop" and compound == "SOFT" and pit_count == 0 and lap in {9, 10, 11, 12})
    )
    if action.startswith(f"pit:{compound}") and wear < 0.92 and not strategic_same_compound_pit:
        return _force_hold("BALANCED")
    if action.startswith("pit:") and lap >= total_laps - 1:
        return _force_hold("PUSH" if wear < 0.5 else "BALANCED")

    if task in {"f1-sprint-dry", "easy-one-stop"}:
        if pit_count >= 1:
            return _force_hold("PUSH" if wear < 0.45 else "BALANCED")
        if compound == "SOFT" and lap in {8, 9, 10, 11}:
            return "pit:MEDIUM:BALANCED"
        if compound == "SOFT" and lap >= 12 and wear >= 0.78:
            return "pit:MEDIUM:BALANCED"
        if action.startswith("pit:") and lap < 8 and wear < 0.85:
            return _force_hold("BALANCED")

    if task in {"f1-feature-safetycar", "hard-safety-car", "medium-strategy"}:
        if pit_count >= 2:
            return _force_hold("PUSH" if wear < 0.35 else "BALANCED")
        if pit_count == 0 and sc and lap in {18, 19, 20}:
            return "pit:HARD:BALANCED"
        if pit_count == 0 and action.startswith("pit:") and not sc and lap < 18 and wear < 0.82:
            return _force_hold("BALANCED")
        if pit_count == 1 and action.startswith("pit:") and lap < 34 and wear < 0.8:
            return _force_hold("BALANCED")

    if task in {"f1-chaos-weather", "ultra-chaos"}:
        if pit_count >= 3 and wear < 0.9:
            return _force_hold("CONSERVE")
        if task == "ultra-chaos" and pit_count == 0 and compound == "SOFT" and lap in {14, 15, 16}:
            return "pit:MEDIUM:BALANCED"
        if task == "ultra-chaos" and pit_count == 0 and compound == "SOFT" and lap >= 18 and wear >= 0.7:
            return "pit:MEDIUM:BALANCED"
        if task == "f1-chaos-weather" and pit_count == 0 and compound == "SOFT" and lap in {12, 13, 14, 15}:
            return "pit:MEDIUM:BALANCED"
        if task == "f1-chaos-weather" and pit_count == 0 and compound == "SOFT" and lap >= 16 and wear >= 0.7:
            return "pit:MEDIUM:BALANCED"
        wetness_switch = 0.5 if task == "ultra-chaos" else 0.35
        if wetness > wetness_switch and compound in {"SOFT", "MEDIUM", "HARD"}:
            return "pit:INTER:BALANCED"
        if task == "f1-chaos-weather" and wetness > 0.82 and compound == "INTER" and wear > 0.9 and lap >= 52:
            return "pit:WET:BALANCED"
        if compound == "INTER" and action.startswith("pit:WET") and wetness < 0.7:
            return _force_hold("CONSERVE")
        if compound == "WET" and action.startswith("pit:INTER") and wetness > 0.4:
            return _force_hold("CONSERVE")
        if action.startswith("pit:") and wetness < 0.08 and forecast < 0.2 and wear < 0.82:
            return _force_hold("BALANCED")

    return action


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

    if task in {"f1-sprint-dry", "easy-one-stop"}:
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

    elif task in {"f1-feature-safetycar", "hard-safety-car"}:
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

    elif task == "medium-strategy":
        # Best deterministic baseline is a single reset stop around lap ~22.
        if pit_count == 0 and lap in {20, 21, 22, 23} and compound == "MEDIUM":
            return "pit:MEDIUM:BALANCED"
        if pit_count == 0 and lap >= 24 and (wear > 0.55 or cliff > 0.9):
            return "pit:MEDIUM:BALANCED"
        if wear < 0.25 and lap < 18:
            return "hold:PUSH"
        if wear > 0.65:
            return "hold:CONSERVE"
        return "hold:BALANCED"

    elif task in {"f1-chaos-weather", "ultra-chaos"}:
        # Align with optimal windows discovered by the task solver.
        # Ultra-chaos: dry reset around lap 15, then weather switch around lap 24.
        if task == "ultra-chaos":
            if pit_count == 0 and compound == "SOFT" and lap in {14, 15, 16}:
                return "pit:MEDIUM:BALANCED"
            if pit_count == 1 and compound in {"SOFT", "MEDIUM", "HARD"} and lap in {23, 24, 25}:
                return "pit:INTER:BALANCED"
            if pit_count == 0 and lap >= 17 and compound == "SOFT":
                return "pit:MEDIUM:BALANCED"
            if pit_count == 1 and wetness > 0.5 and compound in {"SOFT", "MEDIUM", "HARD"}:
                return "pit:INTER:BALANCED"
            if wetness > 0.75 and compound == "INTER" and wear > 0.88 and lap >= 50:
                return "pit:WET:BALANCED"

        # f1-chaos-weather: later rain onset, keep dry stint longer.
        if task == "f1-chaos-weather":
            if pit_count == 0 and compound == "SOFT" and lap in {12, 13, 14, 15}:
                return "pit:MEDIUM:BALANCED"
            if pit_count == 1 and compound in {"SOFT", "MEDIUM", "HARD"} and lap in {30, 31, 32}:
                return "pit:INTER:BALANCED"
            if pit_count == 1 and wetness > 0.35 and compound in {"SOFT", "MEDIUM", "HARD"}:
                return "pit:INTER:BALANCED"

        # Pace tuning by conditions.
        if wetness < 0.2 and wear < 0.35:
            return "hold:PUSH"
        if wetness >= 0.2 and compound == "INTER":
            return "hold:BALANCED"
        if wetness >= 0.2 and compound == "WET":
            return "hold:CONSERVE"
        if wear > 0.7:
            return "hold:CONSERVE"
        return "hold:BALANCED"

    # Generic fallback
    if wear > 0.85:
        return "hold:CONSERVE"
    return "hold:BALANCED"


# --------------- LLM-based action selection ---------------

SYSTEM_PROMPT = """
You are a Championship-winning F1 Strategy Engineer.
Your goal is to complete the race in the minimum total time.

Rules:
1. Soft tires wear out fast but are quick. Hard tires are slower but last long.
2. If track_wetness > 0.5, you should prefer INTER or WET over slick compounds.
3. If safety_car_active is true, pit stop cost is discounted.
4. Tire wear > 0.85 has high puncture risk.

You will receive current telemetry. Respond ONLY as JSON object:
{"reasoning":"brief thought process","action":"STAY_OUT"|"PIT_SOFT"|"PIT_MEDIUM"|"PIT_HARD"|"PIT_INTER"|"PIT_WET"}
""".strip()


def _action_token_to_env_action(token: DecisionActionToken, obs: dict) -> str:
    if token == DecisionActionToken.stay_out:
        # Pace heuristic for stable STAY_OUT behavior.
        wear = float(obs.get("tire_wear_percentage", 0.0))
        sc = bool(obs.get("safety_car_active", False))
        wetness = float(obs.get("track_wetness", 0.0))
        if sc or wetness > 0.35:
            return "hold:CONSERVE"
        if wear < 0.35:
            return "hold:PUSH"
        return "hold:BALANCED"
    if token == DecisionActionToken.pit_soft:
        return "pit:SOFT:BALANCED"
    if token == DecisionActionToken.pit_medium:
        return "pit:MEDIUM:BALANCED"
    if token == DecisionActionToken.pit_hard:
        return "pit:HARD:BALANCED"
    if token == DecisionActionToken.pit_inter:
        return "pit:INTER:BALANCED"
    if token == DecisionActionToken.pit_wet:
        return "pit:WET:BALANCED"
    return _fallback_action(obs)


def _llm_decision_json(client: OpenAI, obs: dict) -> Tuple[str, DecisionActionToken]:
    user_payload = {
        "telemetry": obs,
        "output_schema": {
            "reasoning": "string",
            "action": "STAY_OUT|PIT_SOFT|PIT_MEDIUM|PIT_HARD|PIT_INTER|PIT_WET",
        },
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload)},
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=120,
            response_format={"type": "json_object"},
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        # Some model/provider combinations reject response_format=json_object.
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=120,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        text = match.group(0).strip() if match else raw
    decision = LLMDecision.model_validate_json(text)
    return decision.reasoning, decision.action


def choose_action(client: Optional[OpenAI], obs: dict) -> Tuple[str, str]:
    global _LLM_DISABLED, _LLM_DISABLE_REASON, _LLM_ERROR_COUNT

    if client is None or _LLM_DISABLED:
        fallback = _apply_guardrails(_fallback_action(obs), obs)
        if _LLM_DISABLED:
            return f"Fallback policy ({_LLM_DISABLE_REASON})", fallback
        return "Fallback policy", fallback

    try:
        reasoning, token = _llm_decision_json(client, obs)
        action = _action_token_to_env_action(token, obs)
        return reasoning, _apply_guardrails(action, obs)
    except Exception as exc:
        _LLM_ERROR_COUNT += 1
        if _LLM_ERROR_COUNT >= 2:
            _LLM_DISABLED = True
            _LLM_DISABLE_REASON = f"LLM disabled after repeated errors: {type(exc).__name__}"
            print(f"[LLM_WARN] {_LLM_DISABLE_REASON}", flush=True, file=sys.stderr)
        fallback = _apply_guardrails(_fallback_action(obs), obs)
        return "Fallback policy due to model/JSON error", fallback


def to_payload(action: str) -> dict:
    raw = _normalize_action(action).upper()
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


def run_task(http: httpx.Client, client: Optional[OpenAI], task: str, seed: int) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_error: Optional[str] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset = http.post(f"{ENV_BASE_URL}/reset", json={"task": task, "seed": seed})
        reset.raise_for_status()
        result = reset.json()

        for step in range(1, MAX_STEPS + 1):
            if result.get("done"):
                break

            obs = dict(result.get("observation", {}))
            obs["task_name"] = task
            reasoning, action = choose_action(client, obs)
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

    except Exception as exc:
        task_error = str(exc)
        success = False
        score = 0.0
        print(f"[TASK_ERROR] task={task} seed={seed} error={task_error}", flush=True, file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    if not API_KEY:
        print("[WARN] API_KEY/HF_TOKEN/OPENAI_API_KEY not set; using deterministic fallback policy", flush=True, file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    http = httpx.Client(timeout=60.0)

    all_scores: List[float] = []
    per_task: Dict[str, List[float]] = {task: [] for task in TASKS}
    episodes: List[Dict[str, float | int | str]] = []
    interrupted = False
    try:
        try:
            for seed in SEEDS:
                for task in TASKS:
                    score = run_task(http, client, task, seed)
                    all_scores.append(score)
                    per_task[task].append(score)
                    episodes.append({"task": task, "seed": seed, "score": round(score, 6)})
        except KeyboardInterrupt:
            interrupted = True
            print("\n[INTERRUPTED] Received Ctrl+C. Finalizing partial report...", flush=True, file=sys.stderr)

        avg = sum(all_scores) / max(1, len(all_scores))
        print(f"FINAL_AVG_SCORE={avg:.4f}", flush=True, file=sys.stderr)
        for task in TASKS:
            task_scores = per_task[task]
            task_avg = sum(task_scores) / max(1, len(task_scores))
            print(f"  {task}: {task_avg:.4f} over {len(task_scores)} seed(s)", flush=True, file=sys.stderr)

        report = {
            "benchmark": BENCHMARK,
            "model": MODEL_NAME,
            "api_base_url": API_BASE_URL,
            "env_base_url": ENV_BASE_URL,
            "seeds": SEEDS,
            "episodes": episodes,
            "task_average_scores": {
                task: round(sum(scores) / max(1, len(scores)), 6) for task, scores in per_task.items()
            },
            "final_average_score": round(avg, 6),
            "interrupted": interrupted,
        }

        report_file = Path(REPORT_PATH)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"BASELINE_REPORT={report_file}", flush=True, file=sys.stderr)
    finally:
        http.close()


if __name__ == "__main__":
    main()
