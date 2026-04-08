from __future__ import annotations

from .models import ActionType, RaceAction, TireCompound, WeatherType


def clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def step_reward(
    action: RaceAction,
    lap: int,
    position: int,
    prev_position: int,
    tire_wear: float,
    weather: WeatherType,
    safety_car: bool,
    rain_eta_laps: int,
    pit_stops: int,
    last_error: str | None,
) -> float:
    reward = 0.2

    if position < prev_position:
        reward += 0.25
    if position > prev_position:
        reward -= 0.2

    if 0.35 <= tire_wear <= 0.75:
        reward += 0.2
    elif tire_wear > 0.9:
        reward -= 0.25

    if action.action_type == ActionType.pit:
        if weather == WeatherType.rain and action.pit_compound in {TireCompound.intermediate, TireCompound.wet}:
            reward += 0.35
        if weather == WeatherType.dry and action.pit_compound in {TireCompound.soft, TireCompound.medium, TireCompound.hard}:
            reward += 0.2
        if safety_car:
            reward += 0.15
        # Reward proactive weather calls shortly before rain, but discourage over-pitting.
        if 0 <= rain_eta_laps <= 2 and action.pit_compound in {TireCompound.intermediate, TireCompound.wet}:
            reward += 0.2
        if pit_stops >= 3:
            reward -= 0.12

    if last_error:
        reward -= 0.4

    # Normalize to [0, 1] for dense, stable learning signal.
    return clamp_01((reward + 0.6) / 1.6)
