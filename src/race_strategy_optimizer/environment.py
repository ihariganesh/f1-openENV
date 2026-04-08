from __future__ import annotations

from typing import Dict, List

from .graders import grade_task
from .models import (
    ActionType,
    OpenEnvResetResponse,
    OpenEnvStepResponse,
    RaceAction,
    RaceObservation,
    RaceState,
    TireCompound,
    WeatherType,
)
from .rewards import step_reward
from .simulation import run_step
from .tasks import TASKS, TaskConfig


class RaceStrategyEnvironment:
    def __init__(self) -> None:
        self._state: RaceState | None = None
        self._task: TaskConfig | None = None
        self._pit_laps: List[int] = []
        self._wet_compound_pits = 0

    def reset(self, task_name: str, seed: int = 7) -> OpenEnvResetResponse:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'")
        self._task = TASKS[task_name]
        self._pit_laps = []
        self._wet_compound_pits = 0

        self._state = RaceState(
            task_name=task_name,
            seed=seed,
            lap=1,
            total_laps=self._task.total_laps,
            tire_compound=self._task.start_compound,
            tire_wear=0.05,
            fuel_level=self._task.start_fuel,
            weather_timeline=self._task.weather_timeline,
            safety_car_laps=list(self._task.safety_car_laps),
            position=self._task.start_position,
            cumulative_time_loss=0.0,
            pit_stops=0,
        )
        return OpenEnvResetResponse(observation=self._observation())

    def state(self) -> Dict[str, object]:
        if self._state is None:
            return {"initialized": False}
        return self._state.model_dump()

    def step(self, action: RaceAction) -> OpenEnvStepResponse:
        if self._state is None or self._task is None:
            raise RuntimeError("Environment not initialized. Call reset first.")

        if self._state.done:
            return OpenEnvStepResponse(observation=self._observation(), reward=0.0, done=True, info={"message": "episode finished"})

        current_weather = self._state.weather_timeline.get(self._state.lap, WeatherType.dry)
        safety_car = self._state.lap in self._state.safety_car_laps

        sim = run_step(
            action=action,
            tire_compound=self._state.tire_compound,
            tire_wear=self._state.tire_wear,
            fuel_level=self._state.fuel_level,
            weather=current_weather,
            safety_car_active=safety_car,
        )

        prev_position = self._state.position
        self._state.last_action_error = sim.invalid_reason
        self._state.cumulative_time_loss += sim.time_delta
        self._state.tire_wear = max(0.0, min(1.0, self._state.tire_wear + sim.wear_delta))
        self._state.fuel_level = max(0.0, min(1.0, self._state.fuel_level - sim.fuel_delta))
        self._state.position = max(1, min(20, self._state.position + sim.position_delta))

        if action.action_type == ActionType.pit and action.pit_compound is not None:
            self._state.pit_stops += 1
            self._pit_laps.append(self._state.lap)
            self._state.tire_compound = action.pit_compound
            self._state.tire_wear = 0.05
            if action.pit_compound in {TireCompound.intermediate, TireCompound.wet}:
                self._wet_compound_pits += 1

        reward = step_reward(
            action=action,
            lap=self._state.lap,
            position=self._state.position,
            prev_position=prev_position,
            tire_wear=self._state.tire_wear,
            weather=current_weather,
            safety_car=safety_car,
            rain_eta_laps=self._rain_eta_laps(self._state.lap),
            pit_stops=self._state.pit_stops,
            last_error=self._state.last_action_error,
        )

        self._state.lap += 1
        done = self._state.lap > self._state.total_laps or self._state.fuel_level <= 0.0
        self._state.done = done

        metrics = {
            "pit_laps": list(self._pit_laps),
            "finish_position": self._state.position,
            "total_time_loss": self._state.cumulative_time_loss,
            "safety_car_laps": list(self._state.safety_car_laps),
            "rain_start_lap": self._rain_start_lap(),
            "wet_compound_pits": self._wet_compound_pits,
        }
        grade = grade_task(self._state.task_name, metrics)

        return OpenEnvStepResponse(
            observation=self._observation(),
            reward=reward,
            done=done,
            info={
                "last_action_error": self._state.last_action_error or "",
                "grader_score": grade.score,
                "pit_stops": self._state.pit_stops,
            },
        )

    def final_score(self) -> float:
        if self._state is None:
            return 0.0
        metrics = {
            "pit_laps": list(self._pit_laps),
            "finish_position": self._state.position,
            "total_time_loss": self._state.cumulative_time_loss,
            "safety_car_laps": list(self._state.safety_car_laps),
            "rain_start_lap": self._rain_start_lap(),
            "wet_compound_pits": self._wet_compound_pits,
        }
        return grade_task(self._state.task_name, metrics).score

    def _rain_start_lap(self) -> int:
        if self._state is None:
            return 999
        for lap, weather in self._state.weather_timeline.items():
            if weather == WeatherType.rain:
                return lap
        return 999

    def _rain_eta_laps(self, lap: int) -> int:
        rain_start = self._rain_start_lap()
        if rain_start == 999:
            return 999
        return max(0, rain_start - lap)

    def _observation(self) -> RaceObservation:
        if self._state is None:
            raise RuntimeError("Environment not initialized")

        lap = self._state.lap
        total = self._state.total_laps
        weather_now = self._state.weather_timeline.get(lap, WeatherType.dry)
        forecast = [self._state.weather_timeline.get(min(total, lap + i), weather_now) for i in range(1, 4)]
        safety_car = lap in self._state.safety_car_laps
        rain_eta = self._rain_eta_laps(lap)
        undercut_risk = max(0.0, min(1.0, 1.0 - self._state.tire_wear + (0.15 if self._state.position > 8 else 0.0)))

        return RaceObservation(
            task_name=self._state.task_name,
            lap=lap,
            total_laps=total,
            tire_compound=self._state.tire_compound,
            tire_wear=round(self._state.tire_wear, 3),
            fuel_level=round(self._state.fuel_level, 3),
            weather_now=weather_now,
            weather_next_3_laps=forecast,
            safety_car_active=safety_car,
            pit_stops=self._state.pit_stops,
            rain_eta_laps=rain_eta,
            undercut_risk=round(undercut_risk, 3),
            position=self._state.position,
            gap_to_car_ahead=max(0.2, round(0.8 + self._state.position * 0.1, 2)),
            gap_to_car_behind=max(0.2, round(0.5 + (21 - self._state.position) * 0.08, 2)),
            rivals=[
                {"rival_id": "R1", "gap_seconds": 1.4},
                {"rival_id": "R2", "gap_seconds": 2.1},
            ],
            legal_actions=[ActionType.push, ActionType.conserve, ActionType.pit, ActionType.hold],
            notes=self._task.objective if self._task else "",
        )
