from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx
from openai import OpenAI

TASKS = ["f1-sprint-dry", "f1-feature-safetycar", "f1-chaos-weather"]


@dataclass(frozen=True)
class PolicyParams:
    sprint_pit_lap: int
    sc_window_start: int
    sc_window_end: int
    late_wear_threshold: float
    rain_forecast_threshold: float
    wet_track_threshold: float
    dry_back_threshold: float


def _post_json_with_retry(
    http: httpx.Client,
    url: str,
    payload: Dict[str, Any],
    *,
    retries: int,
    tag: str,
) -> Dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = http.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            print(f"[RETRY] tag={tag} attempt={attempt}/{retries} error={exc}", flush=True)

    raise RuntimeError(f"request failed after {retries} attempts for {tag}: {last_exc}")


def _prompt_for(obs: Dict[str, Any], params: PolicyParams) -> str:
    return (
        "You are an F1 race strategy engineer.\n"
        "Return a single JSON object with exactly these keys: pit_stop, new_compound, pace_mode.\n"
        "Rules:\n"
        "- If pit_stop=false, set new_compound to null.\n"
        "- new_compound must be one of SOFT, MEDIUM, HARD, INTER if pit_stop=true.\n"
        "- pace_mode must be one of PUSH, BALANCED, CONSERVE.\n"
        "- Never add markdown, explanations, or extra keys.\n"
        "\n"
        "Environment-specific strategy priors (learned params):\n"
        f"- Sprint dry: consider one stop near lap {params.sprint_pit_lap} onto MEDIUM.\n"
        f"- Safety car discounted pit window: lap {params.sc_window_start}..{params.sc_window_end}.\n"
        f"- Late-race wear threshold for defensive pit: {params.late_wear_threshold:.2f}.\n"
        f"- Forecast threshold for INTER prep: {params.rain_forecast_threshold:.2f}.\n"
        f"- Wet-track threshold to require INTER: {params.wet_track_threshold:.2f}.\n"
        f"- Dry-back threshold to leave INTER: {params.dry_back_threshold:.2f}.\n"
        "\n"
        f"Observation: {json.dumps(obs, separators=(',', ':'))}"
    )


def _safe_action(content: str) -> Dict[str, Any]:
    fallback = {"pit_stop": False, "new_compound": None, "pace_mode": "BALANCED"}
    try:
        data = json.loads(content)
    except Exception:
        return fallback

    if not isinstance(data, dict):
        return fallback

    pit_stop = bool(data.get("pit_stop", False))
    new_compound = data.get("new_compound", None)
    pace_mode = str(data.get("pace_mode", "BALANCED")).upper()

    if pace_mode not in {"PUSH", "BALANCED", "CONSERVE"}:
        pace_mode = "BALANCED"

    if pit_stop:
        if new_compound not in {"SOFT", "MEDIUM", "HARD", "INTER"}:
            new_compound = "MEDIUM"
    else:
        new_compound = None

    return {"pit_stop": pit_stop, "new_compound": new_compound, "pace_mode": pace_mode}


def _candidate_params() -> Iterable[PolicyParams]:
    for sprint_pit_lap in (9, 10, 11, 12):
        for sc_window in ((18, 20), (17, 20), (18, 21)):
            for late_wear in (0.82, 0.86, 0.9):
                for rain_forecast in (0.45, 0.55, 0.65):
                    for wet_track in (0.35, 0.45, 0.55):
                        for dry_back in (0.1, 0.15, 0.2):
                            yield PolicyParams(
                                sprint_pit_lap=sprint_pit_lap,
                                sc_window_start=sc_window[0],
                                sc_window_end=sc_window[1],
                                late_wear_threshold=late_wear,
                                rain_forecast_threshold=rain_forecast,
                                wet_track_threshold=wet_track,
                                dry_back_threshold=dry_back,
                            )


def run_episode(
    model_client: OpenAI,
    model_name: str,
    env_base_url: str,
    task: str,
    seed: int,
    params: PolicyParams,
    max_steps: int = 80,
    http_timeout: float = 60.0,
    request_retries: int = 3,
) -> Dict[str, Any]:
    print(f"[EPISODE_START] task={task} seed={seed} env={env_base_url}", flush=True)
    with httpx.Client(timeout=http_timeout) as http:
        try:
            payload = _post_json_with_retry(
                http,
                f"{env_base_url}/reset",
                {"task": task, "seed": seed},
                retries=request_retries,
                tag=f"reset:{task}:{seed}",
            )

            steps = 0
            while not payload.get("done", False) and steps < max_steps:
                obs = payload.get("observation", {})
                msg = _prompt_for(obs, params)
                response = model_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": msg},
                    ],
                    temperature=0,
                    max_tokens=80,
                )
                text = (response.choices[0].message.content or "").strip()
                action = _safe_action(text)

                payload = _post_json_with_retry(
                    http,
                    f"{env_base_url}/step",
                    action,
                    retries=request_retries,
                    tag=f"step:{task}:{seed}:{steps + 1}",
                )
                steps += 1

                if steps % 10 == 0 or bool(payload.get("done", False)):
                    obs = payload.get("observation", {})
                    lap = obs.get("current_lap", "?")
                    total_laps = obs.get("total_laps", "?")
                    reward = payload.get("reward", 0.0)
                    print(
                        f"[EPISODE_PROGRESS] task={task} seed={seed} step={steps} lap={lap}/{total_laps} reward={reward}",
                        flush=True,
                    )

            state_data = _post_json_with_retry(
                http,
                f"{env_base_url}/state",
                {},
                retries=request_retries,
                tag=f"state:{task}:{seed}",
            )

            result = {
                "task": task,
                "seed": seed,
                "done": bool(payload.get("done", False)),
                "steps": steps,
                "grader_score": float(state_data.get("grader_score", 0.0)),
                "interim_progress_score": float(state_data.get("interim_progress_score", 0.0)),
                "optimal_total_time": float(state_data.get("optimal_total_time", 0.0)),
                "cumulative_race_time_seconds": float(state_data.get("cumulative_race_time_seconds", 0.0)),
                "error": "",
            }
        except Exception as exc:
            result = {
                "task": task,
                "seed": seed,
                "done": False,
                "steps": 0,
                "grader_score": 0.0,
                "interim_progress_score": 0.0,
                "optimal_total_time": 0.0,
                "cumulative_race_time_seconds": 0.0,
                "error": str(exc),
            }
            print(f"[EPISODE_ERROR] task={task} seed={seed} error={exc}", flush=True)

    print(
        f"[EPISODE_END] task={task} seed={seed} done={result['done']} steps={result['steps']} score={result['grader_score']:.4f}",
        flush=True,
    )
    return result


def evaluate_params(
    model_client: OpenAI,
    model_name: str,
    env_base_url: str,
    params: PolicyParams,
    tasks: List[str],
    seeds: List[int],
    max_steps: int,
    http_timeout: float,
    request_retries: int,
) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            print(f"[EVAL_RUN] task={task} seed={seed}", flush=True)
            episodes.append(
                run_episode(
                    model_client,
                    model_name,
                    env_base_url,
                    task,
                    seed,
                    params,
                    max_steps=max_steps,
                    http_timeout=http_timeout,
                    request_retries=request_retries,
                )
            )

    avg_score = sum(ep["grader_score"] for ep in episodes) / max(1, len(episodes))
    avg_progress = sum(ep["interim_progress_score"] for ep in episodes) / max(1, len(episodes))
    return {
        "params": asdict(params),
        "episodes": episodes,
        "avg_grader_score": round(avg_score, 6),
        "avg_interim_progress_score": round(avg_progress, 6),
    }


def train_params(
    model_client: OpenAI,
    model_name: str,
    train_env_base_url: str,
    train_tasks: List[str],
    train_seeds: List[int],
    trials: int,
    max_steps: int,
    http_timeout: float,
    request_retries: int,
) -> Dict[str, Any]:
    candidates = list(_candidate_params())
    random.shuffle(candidates)
    selected = candidates[: max(1, min(trials, len(candidates)))]

    best: Dict[str, Any] | None = None
    for idx, params in enumerate(selected, start=1):
        result = evaluate_params(
            model_client=model_client,
            model_name=model_name,
            env_base_url=train_env_base_url,
            params=params,
            tasks=train_tasks,
            seeds=train_seeds,
            max_steps=max_steps,
            http_timeout=http_timeout,
            request_retries=request_retries,
        )
        print(
            f"[TRAIN] trial={idx}/{len(selected)} avg_grader_score={result['avg_grader_score']:.4f} "
            f"params={result['params']}",
            flush=True,
        )
        if best is None or result["avg_grader_score"] > best["avg_grader_score"]:
            best = result

    assert best is not None
    return best


def _parse_int_list(raw: str) -> List[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(v) for v in values]


def _parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local Gemma policy params in OpenEnv, then test on real endpoint.")
    parser.add_argument("--model", default="gemma4:e2b", help="Model name served by local OpenAI-compatible endpoint.")
    parser.add_argument("--api-base", default="http://localhost:11434/v1", help="OpenAI-compatible base URL (Ollama default).")
    parser.add_argument("--api-key", default="ollama", help="API key placeholder for OpenAI SDK.")
    parser.add_argument("--train-env", default="http://localhost:8000", help="Training environment base URL.")
    parser.add_argument(
        "--eval-env",
        default="https://ihariganesh-f1-race-stratergy.hf.space",
        help="Real environment URL for post-training evaluation.",
    )
    parser.add_argument("--train-tasks", default=",".join(TASKS), help="Comma-separated task names.")
    parser.add_argument("--eval-tasks", default=",".join(TASKS), help="Comma-separated task names.")
    parser.add_argument("--train-seeds", default="7,13", help="Comma-separated train seeds.")
    parser.add_argument("--eval-seeds", default="7,13,21", help="Comma-separated eval seeds.")
    parser.add_argument("--trials", type=int, default=16, help="Number of policy-parameter candidates to evaluate.")
    parser.add_argument("--max-steps", type=int, default=80, help="Maximum steps per episode.")
    parser.add_argument("--http-timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--request-retries", type=int, default=3, help="Retry count for reset/step/state HTTP calls.")
    parser.add_argument(
        "--out",
        default="artifacts/local_gemma_training_report.json",
        help="Path to write training/evaluation report.",
    )
    args = parser.parse_args()

    train_tasks = _parse_str_list(args.train_tasks)
    eval_tasks = _parse_str_list(args.eval_tasks)
    train_seeds = _parse_int_list(args.train_seeds)
    eval_seeds = _parse_int_list(args.eval_seeds)

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    best = train_params(
        model_client=client,
        model_name=args.model,
        train_env_base_url=args.train_env,
        train_tasks=train_tasks,
        train_seeds=train_seeds,
        trials=args.trials,
        max_steps=args.max_steps,
        http_timeout=args.http_timeout,
        request_retries=args.request_retries,
    )

    best_params = PolicyParams(**best["params"])
    eval_result = evaluate_params(
        model_client=client,
        model_name=args.model,
        env_base_url=args.eval_env,
        params=best_params,
        tasks=eval_tasks,
        seeds=eval_seeds,
        max_steps=args.max_steps,
        http_timeout=args.http_timeout,
        request_retries=args.request_retries,
    )

    report = {
        "model": args.model,
        "api_base": args.api_base,
        "train_env": args.train_env,
        "eval_env": args.eval_env,
        "best_train_result": best,
        "real_eval_result": eval_result,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[DONE] Best training avg grader score:", best["avg_grader_score"], flush=True)
    print("[DONE] Real eval avg grader score:", eval_result["avg_grader_score"], flush=True)
    print(f"[DONE] Report written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
