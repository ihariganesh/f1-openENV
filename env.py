from __future__ import annotations

from dataclasses import dataclass
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
    pit_stop_count: int
    pit_laps: list[int]


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
        lap_reward = 0.0
        penalty = 0.0
        bonus = 0.0

        # 1. Dense pace reward: better lap times yield more reward
        optimal_base = 90.0 - 1.2  # Best theoretical base (SOFT, no fuel)
        pace_score = max(0.0, 1.0 - ((lap_time - optimal_base) / 25.0))
        lap_reward += pace_score * 0.15

        # 2. Tire failure (puncture) penalty
        puncture_flag = False
        if st.tire_wear_percentage >= 1.0:
            penalty += 0.8
            puncture_flag = True
            lap_time += 6.0

        # 3. Wrong tire compound penalty
        wet = track_wetness(st.task_name, lap)
        wrong_tire_flag = False
        if wet > 0.5 and st.current_tire_compound in {"SOFT", "MEDIUM", "HARD"}:
            penalty += 0.6  # Running slicks in heavy rain
            wrong_tire_flag = True
        elif wet < 0.2 and st.current_tire_compound == "WET":
            penalty += 0.4  # Running wets on dry track
            wrong_tire_flag = True

        # 4. Fuel depletion penalty
        fuel_critical = False
        if st.fuel_kg <= 0.0:
            penalty += 1.0
            fuel_critical = True

        # 5. Safety car pit reward: pitting under SC saves time
        if pit_stop and st.safety_car_active:
            bonus += 0.3

        # 6. Proactive weather adaptation reward
        forecast = rain_forecast_next_5(st.task_name, lap)
        if pit_stop and forecast > 0.5 and action.new_compound in {"INTER", "WET"} and wet <= 0.3:
            bonus += 0.25  # Smart preemptive pit

        # 7. Tire health management reward
        if 0.2 <= st.tire_wear_percentage <= 0.7:
            bonus += 0.05  # In the sweet spot
        elif st.tire_wear_percentage > 0.85 and not pit_stop:
            penalty += 0.1  # Dangerously extended stint

        # 8. Excessive pit stop penalty (more than 3 stops is almost always suboptimal)
        if st.pit_stop_count > 3:
            penalty += 0.15

        # 9. Invalid action penalty
        if st.last_action_error:
            penalty += 0.3

        # 10. PUSH mode reward when DRS is available (realistic race behavior)
        if action.pace_mode == "PUSH" and is_drs_available(st.task_name, lap) and st.tire_wear_percentage < 0.6:
            bonus += 0.05

        # 11. CONSERVE under safety car bonus
        if action.pace_mode == "CONSERVE" and st.safety_car_active and not pit_stop:
            bonus += 0.08

        st.last_lap_time_seconds = lap_time
        st.cumulative_race_time_seconds += lap_time

        reward_value = lap_reward + bonus - penalty

        # --- Terminal reward ---
        st.current_lap += 1
        st.done = st.current_lap > st.total_laps

        if st.done and not st.terminal_bonus_awarded:
            grade = grade_episode(st.task_name, st.cumulative_race_time_seconds)
            if grade.score >= 0.9:
                reward_value += 2.0
            elif grade.score >= 0.7:
                reward_value += 1.0
            elif grade.score >= 0.5:
                reward_value += 0.5
            st.terminal_bonus_awarded = True
        else:
            grade = grade_episode(st.task_name, st.cumulative_race_time_seconds)

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
            "lap_reward_breakdown": f"pace={round(lap_reward, 3)},bonus={round(bonus, 3)},penalty={round(penalty, 3)}",
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
            fuel_kg=round(st.fuel_kg, 2),
            track_wetness=round(track_wetness(st.task_name, lap_for_weather), 4),
            rain_forecast_next_5_laps=round(rain_forecast_next_5(st.task_name, lap_for_weather), 4),
            safety_car_active=st.safety_car_active,
            last_lap_time_seconds=round(st.last_lap_time_seconds, 4),
            cumulative_race_time_seconds=round(st.cumulative_race_time_seconds, 4),
            pit_stop_count=st.pit_stop_count,
            drs_available=is_drs_available(st.task_name, lap_for_weather),
            track_temperature_c=track_temperature(st.task_name, lap_for_weather),
            tire_cliff_proximity=tire_cliff_proximity(st.current_tire_compound, st.tire_age_laps),
        )
