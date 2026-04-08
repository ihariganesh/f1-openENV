from __future__ import annotations

from typing import Dict, List

from .models import GraderResult


def _norm(v: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    x = (v - low) / (high - low)
    return max(0.0, min(1.0, x))


def grade_easy(pit_laps: List[int], finish_position: int, total_time_loss: float) -> GraderResult:
    target_lap = 6
    if not pit_laps:
        pit_quality = 0.0
    else:
        pit_quality = max(0.0, 1.0 - abs(pit_laps[0] - target_lap) / 6.0)
    position_score = _norm(12 - finish_position, 0, 11)
    efficiency = _norm(30 - total_time_loss, 0, 30)
    score = 0.5 * pit_quality + 0.3 * position_score + 0.2 * efficiency
    return GraderResult(score=max(0.0, min(1.0, score)), subscores={"pit_quality": pit_quality, "position": position_score, "efficiency": efficiency})


def grade_medium(pit_laps: List[int], safety_car_laps: List[int], finish_position: int, total_time_loss: float) -> GraderResult:
    sc_window = set(safety_car_laps)
    sc_pit = 1.0 if any(l in sc_window for l in pit_laps) else 0.0
    position_score = _norm(14 - finish_position, 0, 13)
    efficiency = _norm(35 - total_time_loss, 0, 35)
    score = 0.45 * sc_pit + 0.35 * position_score + 0.2 * efficiency
    return GraderResult(score=max(0.0, min(1.0, score)), subscores={"safety_car_call": sc_pit, "position": position_score, "efficiency": efficiency})


def grade_hard(
    pit_laps: List[int],
    rain_start_lap: int,
    wet_compound_pits: int,
    finish_position: int,
    total_time_loss: float,
) -> GraderResult:
    multi_stop = 1.0 if len(pit_laps) >= 2 else 0.0
    rain_adapt = 1.0 if wet_compound_pits >= 1 and any(l >= rain_start_lap for l in pit_laps) else 0.0
    position_score = _norm(16 - finish_position, 0, 15)
    efficiency = _norm(45 - total_time_loss, 0, 45)
    score = 0.25 * multi_stop + 0.35 * rain_adapt + 0.25 * position_score + 0.15 * efficiency
    return GraderResult(score=max(0.0, min(1.0, score)), subscores={"multi_stop": multi_stop, "rain_adapt": rain_adapt, "position": position_score, "efficiency": efficiency})


def grade_task(task_name: str, metrics: Dict[str, object]) -> GraderResult:
    if task_name == "pit-window-easy":
        return grade_easy(
            pit_laps=list(metrics.get("pit_laps", [])),
            finish_position=int(metrics.get("finish_position", 20)),
            total_time_loss=float(metrics.get("total_time_loss", 99.0)),
        )
    if task_name == "safety-car-medium":
        return grade_medium(
            pit_laps=list(metrics.get("pit_laps", [])),
            safety_car_laps=list(metrics.get("safety_car_laps", [])),
            finish_position=int(metrics.get("finish_position", 20)),
            total_time_loss=float(metrics.get("total_time_loss", 99.0)),
        )

    return grade_hard(
        pit_laps=list(metrics.get("pit_laps", [])),
        rain_start_lap=int(metrics.get("rain_start_lap", 99)),
        wet_compound_pits=int(metrics.get("wet_compound_pits", 0)),
        finish_position=int(metrics.get("finish_position", 20)),
        total_time_loss=float(metrics.get("total_time_loss", 99.0)),
    )
