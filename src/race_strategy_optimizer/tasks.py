from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .models import TaskSpec, TireCompound, WeatherType


@dataclass(frozen=True)
class TaskConfig:
    name: str
    difficulty: str
    objective: str
    total_laps: int
    start_compound: TireCompound
    start_fuel: float
    start_position: int
    safety_car_laps: tuple[int, ...]
    weather_timeline: Dict[int, WeatherType]


def _build_weather(total_laps: int, first: WeatherType, second_from: int | None = None, second: WeatherType | None = None) -> Dict[int, WeatherType]:
    data: Dict[int, WeatherType] = {}
    for lap in range(1, total_laps + 1):
        if second_from is not None and second is not None and lap >= second_from:
            data[lap] = second
        else:
            data[lap] = first
    return data


TASKS: Dict[str, TaskConfig] = {
    "pit-window-easy": TaskConfig(
        name="pit-window-easy",
        difficulty="easy",
        objective="Find a near-optimal single pit lap using tire wear and fuel trend.",
        total_laps=12,
        start_compound=TireCompound.soft,
        start_fuel=1.0,
        start_position=5,
        safety_car_laps=(),
        weather_timeline=_build_weather(12, WeatherType.dry),
    ),
    "safety-car-medium": TaskConfig(
        name="safety-car-medium",
        difficulty="medium",
        objective="React to a safety car deployment and choose whether to pit for track-position gain.",
        total_laps=16,
        start_compound=TireCompound.medium,
        start_fuel=1.0,
        start_position=8,
        safety_car_laps=(7, 8),
        weather_timeline=_build_weather(16, WeatherType.dry),
    ),
    "weather-shift-hard": TaskConfig(
        name="weather-shift-hard",
        difficulty="hard",
        objective="Execute a multi-stop strategy across dry-to-rain transition while protecting position.",
        total_laps=20,
        start_compound=TireCompound.medium,
        start_fuel=1.0,
        start_position=9,
        safety_car_laps=(10,),
        weather_timeline=_build_weather(20, WeatherType.dry, second_from=13, second=WeatherType.rain),
    ),
}


def list_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(
            name=t.name,
            difficulty=t.difficulty,
            objective=t.objective,
            total_laps=t.total_laps,
        )
        for t in TASKS.values()
    ]
