from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Dict, Iterable, List, Sequence

from models import GraderResult, TaskSpec, TireCompound


@dataclass(frozen=True)
class TaskConfig:
    name: str
    difficulty: str
    description: str
    total_laps: int
    start_compound: TireCompound
    safety_car_laps: Sequence[int]
    pit_penalty_normal: float
    pit_penalty_safety_car: float
    tolerance_seconds: float
    track_temp_base: float        # Base track temperature (°C)
    drs_laps: Sequence[int]       # Laps where DRS is available
    rain_start_lap: int | None = None
    rain_ramp_laps: int = 0
    rain_peak_wetness: float = 0.0


TASKS: Dict[str, TaskConfig] = {
    "f1-sprint-dry": TaskConfig(
        name="f1-sprint-dry",
        difficulty="easy",
        description=(
            "20 lap sprint race on a dry circuit. Soft tires degrade rapidly after ~12 laps "
            "with a severe cliff. A well-timed single pit stop to Medium tires around lap 10 "
            "is the expected optimum. Fuel starts at 110 kg and burns ~1.5 kg/lap."
        ),
        total_laps=20,
        start_compound="SOFT",
        safety_car_laps=(),
        pit_penalty_normal=20.0,
        pit_penalty_safety_car=20.0,
        tolerance_seconds=12.0,     # Tightened from 16s
        track_temp_base=38.0,
        drs_laps=tuple(range(3, 21)),  # DRS available from lap 3 onward
    ),
    "f1-feature-safetycar": TaskConfig(
        name="f1-feature-safetycar",
        difficulty="medium",
        description=(
            "50 lap feature race. A Safety Car is deployed on laps 18-20, cutting pit stop "
            "time by 10s. The agent must recognize this window and pit under Safety Car to "
            "save time. Track starts dry. Two-stop strategies with clever tire choices score best."
        ),
        total_laps=50,
        start_compound="MEDIUM",
        safety_car_laps=(18, 19, 20),
        pit_penalty_normal=20.0,
        pit_penalty_safety_car=10.0,
        tolerance_seconds=22.0,     # Tightened from 30s
        track_temp_base=35.0,
        drs_laps=tuple(range(5, 51)),
    ),
    "f1-chaos-weather": TaskConfig(
        name="f1-chaos-weather",
        difficulty="hard",
        description=(
            "60 lap race with dynamic weather transition. Track is dry until lap 30, then "
            "wetness ramps linearly to 90% by lap 35 and stays wet. Agent must pre-empt the "
            "weather change using forecast data and switch to INTER or WET tires. Running "
            "slicks on >50% wetness causes catastrophic pace loss. Track temperature drops "
            "from 35°C to 22°C during rain."
        ),
        total_laps=60,
        start_compound="SOFT",
        safety_car_laps=(),
        pit_penalty_normal=20.0,
        pit_penalty_safety_car=20.0,
        tolerance_seconds=35.0,     # Weather changes require more margin
        track_temp_base=35.0,
        drs_laps=tuple(range(5, 30)),  # No DRS in wet conditions
        rain_start_lap=30,
        rain_ramp_laps=5,
        rain_peak_wetness=0.9,
    ),
    # Scenario aliases for the LLM Race Engineer mode.
    "easy-one-stop": TaskConfig(
        name="easy-one-stop",
        difficulty="easy",
        description=(
            "20 lap dry race. Single stop strategy expected around lap 8-11. "
            "No weather change and no safety car interruptions."
        ),
        total_laps=20,
        start_compound="SOFT",
        safety_car_laps=(),
        pit_penalty_normal=20.0,
        pit_penalty_safety_car=20.0,
        tolerance_seconds=12.0,
        track_temp_base=38.0,
        drs_laps=tuple(range(3, 21)),
    ),
    "medium-strategy": TaskConfig(
        name="medium-strategy",
        difficulty="medium",
        description=(
            "50 lap dry race. The agent must balance one-stop versus two-stop strategy "
            "using tire wear and pace trade-offs."
        ),
        total_laps=50,
        start_compound="MEDIUM",
        safety_car_laps=(),
        pit_penalty_normal=20.0,
        pit_penalty_safety_car=20.0,
        tolerance_seconds=24.0,
        track_temp_base=35.0,
        drs_laps=tuple(range(5, 51)),
    ),
    "hard-safety-car": TaskConfig(
        name="hard-safety-car",
        difficulty="hard",
        description=(
            "50 lap race with a deterministic safety car on lap 15. "
            "Pitting during this lap should be strongly beneficial."
        ),
        total_laps=50,
        start_compound="MEDIUM",
        safety_car_laps=(15,),
        pit_penalty_normal=20.0,
        pit_penalty_safety_car=10.0,
        tolerance_seconds=24.0,
        track_temp_base=35.0,
        drs_laps=tuple(range(5, 51)),
    ),
    "ultra-chaos": TaskConfig(
        name="ultra-chaos",
        difficulty="hard",
        description=(
            "60 lap race with rain ramp starting lap 20 and a safety car on lap 40. "
            "Agent must handle overlapping tire/weather/safety-car decisions."
        ),
        total_laps=60,
        start_compound="SOFT",
        safety_car_laps=(40,),
        pit_penalty_normal=20.0,
        pit_penalty_safety_car=10.0,
        tolerance_seconds=36.0,
        track_temp_base=35.0,
        drs_laps=tuple(range(5, 60)),
        rain_start_lap=20,
        rain_ramp_laps=5,
        rain_peak_wetness=0.9,
    ),
}


def list_task_specs() -> List[TaskSpec]:
    return [
        TaskSpec(name=t.name, difficulty=t.difficulty, description=t.description, total_laps=t.total_laps)
        for t in TASKS.values()
    ]


# --------------- Weather model ---------------

def track_wetness(task_name: str, lap: int) -> float:
    task = TASKS.get(task_name)
    if task is None or task.rain_start_lap is None or task.rain_peak_wetness <= 0.0:
        return 0.0

    assert task.rain_start_lap is not None
    if lap < task.rain_start_lap:
        return 0.0
    if task.rain_ramp_laps <= 0:
        return round(task.rain_peak_wetness, 4)

    full_wet_lap = task.rain_start_lap + task.rain_ramp_laps
    if lap >= full_wet_lap:
        return round(task.rain_peak_wetness, 4)

    return round((lap - task.rain_start_lap) * (task.rain_peak_wetness / task.rain_ramp_laps), 4)


def rain_forecast_next_5(task_name: str, lap: int) -> float:
    values = [track_wetness(task_name, lap + i) for i in range(1, 6)]
    if not values:
        return 0.0
    return max(values)


def track_temperature(task_name: str, lap: int) -> float:
    """Track temperature varies with weather. Rain cools the track."""
    task = TASKS.get(task_name)
    if task is None:
        return 35.0
    base = task.track_temp_base
    wet = track_wetness(task_name, lap)
    # Temperature drops proportionally with wetness
    return round(base - wet * 14.0, 1)


def is_drs_available(task_name: str, lap: int) -> bool:
    task = TASKS.get(task_name)
    if task is None:
        return False
    wet = track_wetness(task_name, lap)
    # No DRS in wet conditions
    if wet > 0.3:
        return False
    return lap in task.drs_laps


# --------------- Tire physics model ---------------

def _compound_pace_offset(compound: TireCompound) -> float:
    """Base lap time offset per compound. Softer = faster but degrades faster."""
    return {
        "SOFT": -1.2,
        "MEDIUM": -0.5,
        "HARD": 0.0,
        "INTER": 0.2,
        "WET": 0.5,
    }[compound]


def _degradation_penalty(compound: TireCompound, tire_age_laps: int, wetness: float) -> float:
    """Non-linear degradation model with performance cliffs."""
    if compound == "SOFT":
        # Soft tires have a sharp cliff after ~10 laps
        if tire_age_laps <= 10:
            return 0.04 * tire_age_laps
        cliff_laps = tire_age_laps - 10
        return 0.04 * 10 + 0.15 * (1.6 ** cliff_laps - 1.0)

    if compound == "MEDIUM":
        # Medium tires are consistent but slow after ~25 laps
        if tire_age_laps <= 25:
            return 0.02 * tire_age_laps
        return 0.02 * 25 + 0.04 * (tire_age_laps - 25)

    if compound == "HARD":
        # Hard tires degrade slowly, cliff at ~40 laps
        if tire_age_laps <= 40:
            return 0.01 * tire_age_laps
        return 0.01 * 40 + 0.03 * (tire_age_laps - 40)

    if compound == "INTER":
        # Inters are great in wet, terrible in dry
        if wetness > 0.4:
            return 0.018 * tire_age_laps
        return 0.35 * tire_age_laps  # Massive penalty on dry

    # WET compound
    if wetness > 0.6:
        return 0.022 * tire_age_laps
    if wetness > 0.3:
        return 0.10 * tire_age_laps  # Moderate penalty in light rain
    return 0.50 * tire_age_laps  # Catastrophic on dry


def tire_cliff_proximity(compound: TireCompound, tire_age_laps: int) -> float:
    """Returns 0.0 to 1.0 indicating how close we are to the performance cliff."""
    cliff_map = {"SOFT": 10, "MEDIUM": 25, "HARD": 40, "INTER": 30, "WET": 25}
    cliff_lap = cliff_map[compound]
    if tire_age_laps >= cliff_lap:
        return 1.0
    return round(tire_age_laps / cliff_lap, 4)


def _wear_increment(compound: TireCompound, tire_age_laps: int, wetness: float) -> float:
    if compound == "SOFT":
        base = 0.04 if tire_age_laps <= 10 else 0.07 + 0.015 * (tire_age_laps - 10)
        return min(0.4, base)
    if compound == "MEDIUM":
        return 0.02
    if compound == "HARD":
        return 0.012
    if compound == "INTER":
        if wetness > 0.4:
            return 0.025
        return 0.34  # Rapid wear on dry
    # WET
    if wetness > 0.6:
        return 0.028
    return 0.40  # Very rapid wear if not wet


def _pace_effects(pace_mode: str) -> tuple[float, float, float]:
    """Returns (time_delta, wear_multiplier, fuel_multiplier)"""
    if pace_mode == "PUSH":
        return (-0.5, 1.5, 1.2)
    if pace_mode == "CONSERVE":
        return (1.0, 0.5, 0.8)
    return (0.0, 1.0, 1.0)


def compute_lap_time(
    task: TaskConfig,
    lap: int,
    compound: TireCompound,
    tire_age_laps: int,
    fuel_kg: float,
    pace_mode: str,
    pit_stop: bool,
) -> tuple[float, float, float]:
    wetness = track_wetness(task.name, lap)
    fuel_penalty = (fuel_kg / 10.0) * 0.3
    deg_penalty = _degradation_penalty(compound, tire_age_laps, wetness)

    # Weather penalty for wrong tire choice
    weather_multiplier = 0.0
    if wetness > 0.5 and compound in {"SOFT", "MEDIUM", "HARD"}:
        weather_multiplier += 15.0 * wetness  # Catastrophic
    elif wetness > 0.0 and compound in {"SOFT", "MEDIUM", "HARD"}:
        weather_multiplier += 3.0 * wetness   # Noticeable
    # Wet tires on dry track penalty
    if wetness < 0.3 and compound == "WET":
        weather_multiplier += 8.0 * (1.0 - wetness)

    # DRS benefit (only on dry, adds a small pace advantage)
    drs_bonus = 0.0
    if is_drs_available(task.name, lap) and compound in {"SOFT", "MEDIUM", "HARD"}:
        drs_bonus = -0.3

    base_time = 90.0 + _compound_pace_offset(compound) + fuel_penalty + deg_penalty + weather_multiplier + drs_bonus

    pace_time_delta, wear_mult, fuel_mult = _pace_effects(pace_mode)
    lap_time = base_time + pace_time_delta

    # Pit stop costs
    pit_penalty = 0.0
    if pit_stop:
        pit_penalty = task.pit_penalty_safety_car if lap in task.safety_car_laps else task.pit_penalty_normal
    if lap in task.safety_car_laps:
        lap_time += 5.0  # Safety car slows the field

    lap_time += pit_penalty

    wear = _wear_increment(compound, tire_age_laps, wetness) * wear_mult
    fuel_burn = 1.5 * fuel_mult
    return lap_time, wear, fuel_burn


# --------------- Strategy solver (for optimal baselines) ---------------

def _simulate_strategy(task: TaskConfig, pit_plan: Dict[int, TireCompound], pace_mode: str = "BALANCED") -> float:
    compound: TireCompound = task.start_compound
    tire_age = 0
    tire_wear = 0.0
    fuel_kg = 110.0
    total_time = 0.0

    for lap in range(1, task.total_laps + 1):
        pit_stop = lap in pit_plan
        if pit_stop:
            compound = pit_plan[lap]
            tire_age = 0
            tire_wear = 0.0

        lap_time, wear_inc, fuel_burn = compute_lap_time(task, lap, compound, tire_age, fuel_kg, pace_mode, pit_stop)

        tire_wear = min(1.0, tire_wear + wear_inc)
        if tire_wear >= 1.0:
            lap_time += 6.0

        fuel_kg = max(0.0, fuel_kg - fuel_burn)
        tire_age += 1
        total_time += lap_time

    return total_time


@lru_cache(maxsize=8)
def solve_optimal_total_time(task_name: str) -> float:
    task = TASKS[task_name]
    best = float("inf")

    if task_name in {"f1-sprint-dry", "easy-one-stop"}:
        lap_options = [[9, 10, 11, 12]]
        stop_counts = [1]
    elif task_name in {"f1-feature-safetycar", "hard-safety-car"}:
        lap_options = [[17, 18, 19, 20, 21], [32, 36, 40]]
        stop_counts = [1, 2]
    elif task_name == "medium-strategy":
        lap_options = [[15, 18, 22], [34, 38, 42]]
        stop_counts = [1, 2]
    else:
        lap_options = [[12, 15, 18], [20, 22, 24, 26], [44, 48, 52]]
        stop_counts = [2, 3]

    best = min(best, _simulate_strategy(task, {}))

    for stops in stop_counts:
        selected_windows = lap_options[:stops]
        for laps in product(*selected_windows):
            if tuple(sorted(laps)) != tuple(laps):
                continue

            compound_choices: List[List[TireCompound]] = []
            for _ in laps:
                if task_name in {"f1-chaos-weather", "ultra-chaos"}:
                    compound_choices.append(["MEDIUM", "HARD", "INTER", "WET"])
                else:
                    compound_choices.append(["SOFT", "MEDIUM", "HARD"])

            for compounds_for_stops in product(*compound_choices):
                pit_plan = {lap: comp for lap, comp in zip(laps, compounds_for_stops)}
                total = _simulate_strategy(task, pit_plan)
                if total < best:
                    best = total

    return best


def grade_episode(task_name: str, agent_total_time: float) -> GraderResult:
    task = TASKS[task_name]
    optimal = solve_optimal_total_time(task_name)
    delta = agent_total_time - optimal
    raw_score = 1.0 - (delta / task.tolerance_seconds)
    # Hackathon validator expects strict open interval (0, 1), not inclusive bounds.
    eps = 1e-4
    score = max(eps, min(1.0 - eps, raw_score))
    return GraderResult(
        score=score,
        agent_total_time=agent_total_time,
        optimal_total_time=optimal,
        tolerance_seconds=task.tolerance_seconds,
        details={
            "task": task_name,
            "delta_seconds": round(delta, 4),
            "delta_percentage": round((delta / max(1e-6, optimal)) * 100, 4),
        },
    )
