from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

TireCompound = Literal["SOFT", "MEDIUM", "HARD", "INTER", "WET"]
PaceMode = Literal["PUSH", "BALANCED", "CONSERVE"]


class ObservationSpace(BaseModel):
    """Full observation available to the agent each lap."""

    current_lap: int = Field(ge=1)
    total_laps: int = Field(ge=1)
    current_tire_compound: TireCompound
    tire_age_laps: int = Field(ge=0)
    tire_wear_percentage: float = Field(ge=0.0, le=1.0)
    fuel_kg: float = Field(ge=0.0, description="Remaining fuel in kilograms")
    track_wetness: float = Field(ge=0.0, le=1.0)
    rain_forecast_next_5_laps: float = Field(ge=0.0, le=1.0)
    safety_car_active: bool
    last_lap_time_seconds: float = Field(ge=0.0)
    cumulative_race_time_seconds: float = Field(ge=0.0)
    pit_stop_count: int = Field(ge=0, description="Number of pit stops made so far")
    drs_available: bool = Field(default=False, description="DRS zone available this lap")
    track_temperature_c: float = Field(ge=10.0, le=60.0, default=35.0)
    tire_cliff_proximity: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="How close the current tire is to its performance cliff (1.0 = at cliff)",
    )


class ActionSpace(BaseModel):
    pit_stop: bool
    new_compound: Optional[TireCompound] = None
    pace_mode: PaceMode = "BALANCED"

    @model_validator(mode="after")
    def validate_pit_action(self) -> "ActionSpace":
        if self.pit_stop and self.new_compound is None:
            raise ValueError("new_compound must be provided when pit_stop is true")
        if not self.pit_stop and self.new_compound is not None:
            raise ValueError("new_compound must be null when pit_stop is false")
        return self


class Reward(BaseModel):
    value: float


class ResetResponse(BaseModel):
    observation: ObservationSpace
    reward: float = 0.0
    done: bool = False
    info: Dict[str, float | int | str | bool] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: ObservationSpace
    reward: float
    done: bool
    info: Dict[str, float | int | str | bool] = Field(default_factory=dict)


class GraderResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    agent_total_time: float
    optimal_total_time: float
    tolerance_seconds: float
    details: Dict[str, float | int | str | bool] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    total_laps: int


class TaskSummary(BaseModel):
    benchmark: str = "race_strategy_optimizer"
    tasks: List[TaskSpec]
