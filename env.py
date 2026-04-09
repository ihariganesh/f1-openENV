from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from models import ActionSpace, ObservationSpace, ResetResponse, Reward, StepResponse, TireCompound
from tasks import (
    TASKS,
    compute_lap_time,
    grade_episode,
    is_drs_available,
    rain_forecast_next_5,
    tire_cliff_proximity,
    track_temperature,
    track_wetness,
)


@dataclass
class RaceState:
    task_name: str
    current_lap: int
    total_laps: int
    current_tire_compound: TireCompound
    tire_age_laps: int
    tire_wear_percentage: float
    fuel_kg: float
    last_lap_time_seconds: float
    safety_car_active: bool
    cumulative_race_time_seconds: float
    cumulative_reward: float
    done: bool
    seed: int
    terminal_bonus_awarded: bool
    last_action_error: Optional[str]
    pit_stop_count: int = 0
    pit_laps: list[int] = field(default_factory=list)


class F1StrategyEnv:
    def __init__(self) -> None:
        self._state: Optional[RaceState] = None

    def reset(self, task_name: str = "f1-sprint-dry", seed: int = 7) -> ResetResponse:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")
        task = TASKS[task_name]

        self._state = RaceState(
            task_name=task.name,
            current_lap=1,
            total_laps=task.total_laps,
            current_tire_compound=task.start_compound,
            tire_age_laps=0,
            tire_wear_percentage=0.0,
            fuel_kg=110.0,
            last_lap_time_seconds=0.0,
            safety_car_active=1 in task.safety_car_laps,
            cumulative_race_time_seconds=0.0,
            cumulative_reward=0.0,
            done=False,
            seed=seed,
            terminal_bonus_awarded=False,
            last_action_error=None,
            pit_stop_count=0,
            pit_laps=[],
        )

        return ResetResponse(
            observation=self._observation(),
            reward=0.0,
            done=False,
            info={"task": task_name, "seed": seed},
        )

    def step(self, action: ActionSpace) -> StepResponse:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset().")
        if self._state.done:
            return StepResponse(
                observation=self._observation(), reward=0.0, done=True, info={"message": "episode finished"}
            )

        task = TASKS[self._state.task_name]
        st = self._state
        st.last_action_error = None

        # --- Apply pit stop before lap simulation ---
        pit_stop = action.pit_stop
        if pit_stop:
            if action.new_compound is None:
                st.last_action_error = "pit_stop=true requires new_compound"
            else:
                st.current_tire_compound = action.new_compound
                st.tire_age_laps = 0
                st.tire_wear_percentage = 0.0
                st.pit_stop_count += 1
                st.pit_laps.append(st.current_lap)

        lap = st.current_lap
        st.safety_car_active = lap in task.safety_car_laps

        lap_time, wear_inc, fuel_burn = compute_lap_time(
            task=task,
            lap=lap,
            compound=st.current_tire_compound,
            tire_age_laps=st.tire_age_laps,
            fuel_kg=st.fuel_kg,
            pace_mode=action.pace_mode,
            pit_stop=pit_stop,
        )

        st.tire_wear_percentage = min(1.0, st.tire_wear_percentage + wear_inc)
        st.fuel_kg = max(0.0, st.fuel_kg - fuel_burn)
        st.tire_age_laps += 1

        # --- Penalty and reward calculation ---
        puncture_penalty = 0.0
        wrong_tire_penalty = 0.0

        # 1. Tire failure (puncture) penalty
        puncture_flag = False
        if st.tire_wear_percentage >= 1.0:
            puncture_penalty = 0.8
            puncture_flag = True
            lap_time += 6.0

        # 2. Wrong tire compound penalty (evaluated every lap)
        wet = track_wetness(st.task_name, lap)
        wrong_tire_flag = False
        if wet > 0.5 and st.current_tire_compound in {"SOFT", "MEDIUM", "HARD"}:
            wrong_tire_penalty = 1.5 * wet
            wrong_tire_flag = True
        elif wet > 0.1 and st.current_tire_compound in {"SOFT", "MEDIUM", "HARD"}:
            wrong_tire_penalty = 0.3 * wet
            wrong_tire_flag = True

        # 3. Fuel depletion flag
        fuel_critical = False
        if st.fuel_kg <= 0.0:
            fuel_critical = True

        st.last_lap_time_seconds = lap_time
        st.cumulative_race_time_seconds += lap_time

        # Keep per-step reward in a stable 0..1 range while preserving dense progress signals.
        time_reward = 1.0 - max(0.0, (lap_time - 90.0)) * 0.01
        sc_pit_bonus = 0.3 if (pit_stop and st.safety_car_active) else 0.0
        reward_value_raw = time_reward + sc_pit_bonus - wrong_tire_penalty - puncture_penalty

        if st.last_action_error:
            reward_value_raw -= 0.3
        if fuel_critical:
            reward_value_raw -= 1.0

        # --- Terminal reward ---
        st.current_lap += 1
        st.done = st.current_lap > st.total_laps

        grade = grade_episode(st.task_name, st.cumulative_race_time_seconds)
        if st.done and not st.terminal_bonus_awarded:
            delta = st.cumulative_race_time_seconds - grade.optimal_total_time
            terminal_bonus = 0.2 * max(0.0, 1.0 - delta / grade.tolerance_seconds)
            reward_value_raw += terminal_bonus
            st.terminal_bonus_awarded = True

        reward_value = min(1.0, max(0.0, reward_value_raw))

        reward_model = Reward(value=reward_value)
        st.cumulative_reward += reward_model.value

        progress_ratio = min(1.0, max(0.0, st.cumulative_race_time_seconds / max(1e-6, grade.optimal_total_time)))
        interim_progress_score = max(0.0, 1.0 - progress_ratio)

        info: Dict[str, float | int | str | bool] = {
            "grader_score": grade.score if st.done else 0.0,
            "interim_progress_score": round(interim_progress_score, 6),
            "agent_total_time": round(grade.agent_total_time, 4),
            "optimal_total_time": round(grade.optimal_total_time, 4),
            "tolerance_seconds": grade.tolerance_seconds,
            "last_action_error": st.last_action_error or "",
            "puncture": puncture_flag,
            "wrong_tire": wrong_tire_flag,
            "fuel_critical": fuel_critical,
            "pit_stop_count": st.pit_stop_count,
            "lap_reward_breakdown": (
                f"time={round(time_reward, 3)},sc_pit_bonus={round(sc_pit_bonus, 3)},"
                f"wrong_tire_penalty={round(wrong_tire_penalty, 3)},puncture_penalty={round(puncture_penalty, 3)},"
                f"raw={round(reward_value_raw, 3)},clipped={round(reward_value, 3)}"
            ),
        }

        return StepResponse(observation=self._observation(), reward=reward_model.value, done=st.done, info=info)

    def state(self) -> Dict[str, object]:
        if self._state is None:
            return {"initialized": False}
        st = self._state
        grade = grade_episode(st.task_name, st.cumulative_race_time_seconds)
        lap_for_weather = min(st.current_lap, st.total_laps)
        return {
            "initialized": True,
            "task_name": st.task_name,
            "current_lap": st.current_lap,
            "total_laps": st.total_laps,
            "current_tire_compound": st.current_tire_compound,
            "tire_age_laps": st.tire_age_laps,
            "tire_wear_percentage": round(st.tire_wear_percentage, 4),
            "fuel_kg": round(st.fuel_kg, 4),
            "track_wetness": track_wetness(st.task_name, lap_for_weather),
            "track_temperature_c": track_temperature(st.task_name, lap_for_weather),
            "rain_forecast_next_5_laps": rain_forecast_next_5(st.task_name, lap_for_weather),
            "safety_car_active": st.safety_car_active,
            "drs_available": is_drs_available(st.task_name, lap_for_weather),
            "last_lap_time_seconds": round(st.last_lap_time_seconds, 4),
            "cumulative_race_time_seconds": round(st.cumulative_race_time_seconds, 4),
            "cumulative_reward": round(st.cumulative_reward, 4),
            "pit_stop_count": st.pit_stop_count,
            "pit_laps": list(st.pit_laps),
            "done": st.done,
            "grader_score": round(grade.score, 6) if st.done else 0.0,
            "interim_progress_score": round(
                max(
                    0.0,
                    1.0 - min(1.0, st.cumulative_race_time_seconds / max(1e-6, grade.optimal_total_time)),
                ),
                6,
            ),
            "optimal_total_time": round(grade.optimal_total_time, 4),
            "seed": st.seed,
        }

    def _observation(self) -> ObservationSpace:
        if self._state is None:
            raise RuntimeError("Environment is not initialized")
        st = self._state
        lap_for_weather = min(st.current_lap, st.total_laps)
        return ObservationSpace(
            current_lap=st.current_lap,
            total_laps=st.total_laps,
            current_tire_compound=st.current_tire_compound,
            tire_age_laps=st.tire_age_laps,
            tire_wear_percentage=round(st.tire_wear_percentage, 4),
            fuel_kg=round(st.fuel_kg, 4),
            track_wetness=round(track_wetness(st.task_name, lap_for_weather), 4),
            rain_forecast_next_5_laps=round(rain_forecast_next_5(st.task_name, lap_for_weather), 4),
            safety_car_active=st.safety_car_active,
            last_lap_time_seconds=round(st.last_lap_time_seconds, 4),
            cumulative_race_time_seconds=round(st.cumulative_race_time_seconds, 4),
            pit_stop_count=st.pit_stop_count,
            drs_available=False,
            track_temperature_c=35.0,
            tire_cliff_proximity=round(min(1.0, st.tire_wear_percentage / 0.85), 4),
        )
