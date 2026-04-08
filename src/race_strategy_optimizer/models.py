from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TireCompound(str, Enum):
    soft = "soft"
    medium = "medium"
    hard = "hard"
    intermediate = "intermediate"
    wet = "wet"


class WeatherType(str, Enum):
    dry = "dry"
    mixed = "mixed"
    rain = "rain"


class ActionType(str, Enum):
    push = "push"
    conserve = "conserve"
    pit = "pit"
    hold = "hold"


class RaceAction(BaseModel):
    action_type: ActionType
    pit_compound: Optional[TireCompound] = None


class RivalSnapshot(BaseModel):
    rival_id: str
    gap_seconds: float


class RaceObservation(BaseModel):
    task_name: str
    lap: int
    total_laps: int
    tire_compound: TireCompound
    tire_wear: float = Field(ge=0.0, le=1.0)
    fuel_level: float = Field(ge=0.0, le=1.0)
    weather_now: WeatherType
    weather_next_3_laps: List[WeatherType]
    safety_car_active: bool
    pit_stops: int
    rain_eta_laps: int
    undercut_risk: float = Field(ge=0.0, le=1.0)
    position: int
    gap_to_car_ahead: float
    gap_to_car_behind: float
    rivals: List[RivalSnapshot]
    legal_actions: List[ActionType]
    notes: str


class RaceState(BaseModel):
    task_name: str
    seed: int
    lap: int
    total_laps: int
    tire_compound: TireCompound
    tire_wear: float
    fuel_level: float
    weather_timeline: Dict[int, WeatherType]
    safety_car_laps: List[int]
    position: int
    cumulative_time_loss: float
    pit_stops: int
    last_action_error: Optional[str] = None
    done: bool = False


class OpenEnvResetResponse(BaseModel):
    observation: RaceObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, float | str | int | bool] = Field(default_factory=dict)


class OpenEnvStepResponse(BaseModel):
    observation: RaceObservation
    reward: float
    done: bool
    info: Dict[str, float | str | int | bool] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    name: str
    difficulty: str
    objective: str
    total_laps: int


class GraderResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    subscores: Dict[str, float] = Field(default_factory=dict)
