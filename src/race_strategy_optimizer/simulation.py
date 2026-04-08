from __future__ import annotations

from dataclasses import dataclass

from .models import ActionType, RaceAction, TireCompound, WeatherType


@dataclass
class SimulationStep:
    time_delta: float
    wear_delta: float
    fuel_delta: float
    position_delta: int
    invalid_reason: str | None = None


WEAR_RATE = {
    TireCompound.soft: 0.09,
    TireCompound.medium: 0.065,
    TireCompound.hard: 0.05,
    TireCompound.intermediate: 0.07,
    TireCompound.wet: 0.08,
}

BASE_PACE_PENALTY = {
    TireCompound.soft: 0.0,
    TireCompound.medium: 0.3,
    TireCompound.hard: 0.6,
    TireCompound.intermediate: 0.4,
    TireCompound.wet: 0.5,
}


def weather_penalty(compound: TireCompound, weather: WeatherType) -> float:
    if weather == WeatherType.dry and compound in {TireCompound.intermediate, TireCompound.wet}:
        return 1.3
    if weather == WeatherType.rain and compound in {TireCompound.soft, TireCompound.medium, TireCompound.hard}:
        return 1.8
    return 0.0


def run_step(
    action: RaceAction,
    tire_compound: TireCompound,
    tire_wear: float,
    fuel_level: float,
    weather: WeatherType,
    safety_car_active: bool,
) -> SimulationStep:
    if action.action_type == ActionType.pit and action.pit_compound is None:
        return SimulationStep(0.0, 0.0, 0.0, 0, invalid_reason="pit action requires pit_compound")

    base_delta = 1.0
    if action.action_type == ActionType.push:
        base_delta -= 0.35
    if action.action_type == ActionType.conserve:
        base_delta += 0.25

    safety_factor = 0.55 if safety_car_active else 1.0
    wear = WEAR_RATE[tire_compound] * (1.15 if action.action_type == ActionType.push else 0.85 if action.action_type == ActionType.conserve else 1.0)
    fuel = 0.08 * (1.08 if action.action_type == ActionType.push else 0.92 if action.action_type == ActionType.conserve else 1.0)

    pace_penalty = BASE_PACE_PENALTY[tire_compound]
    pace_penalty += weather_penalty(tire_compound, weather)
    pace_penalty += max(0.0, (tire_wear - 0.65) * 4.0)

    position_delta = 0
    if action.action_type == ActionType.push and pace_penalty < 1.2 and not safety_car_active:
        position_delta = -1
    if action.action_type == ActionType.conserve and pace_penalty > 2.0 and not safety_car_active:
        position_delta = 1

    time_delta = (base_delta + pace_penalty) * safety_factor

    if action.action_type == ActionType.pit:
        time_delta += 9.0 * safety_factor
        wear = -tire_wear
        fuel = 0.02
        position_delta = 2 if not safety_car_active else 1

    if fuel_level - fuel < 0.0:
        return SimulationStep(time_delta + 6.0, wear, fuel, position_delta + 2, invalid_reason="fuel depleted")

    return SimulationStep(time_delta=time_delta, wear_delta=wear, fuel_delta=fuel, position_delta=position_delta)
