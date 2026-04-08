from __future__ import annotations

from race_strategy_optimizer.environment import RaceStrategyEnvironment
from race_strategy_optimizer.models import RaceAction

TASKS = ["pit-window-easy", "safety-car-medium", "weather-shift-hard"]


def policy(task: str, lap: int, weather: str, safety_car: bool, pit_stops: int) -> RaceAction:
    if task == "pit-window-easy" and lap == 6 and pit_stops == 0:
        return RaceAction(action_type="pit", pit_compound="medium")
    if task == "safety-car-medium" and safety_car and lap in {7, 8} and pit_stops == 0:
        return RaceAction(action_type="pit", pit_compound="hard")
    if task == "weather-shift-hard":
        if lap == 6 and pit_stops == 0:
            return RaceAction(action_type="pit", pit_compound="hard")
        if pit_stops == 1 and weather == "rain":
            return RaceAction(action_type="pit", pit_compound="intermediate")
    return RaceAction(action_type="conserve")


def run_task(task: str) -> float:
    env = RaceStrategyEnvironment()
    step = env.reset(task_name=task, seed=7)
    done = False
    while not done:
        obs = step.observation
        action = policy(task, obs.lap, obs.weather_now.value, obs.safety_car_active, obs.pit_stops)
        step = env.step(action)
        done = step.done
    return env.final_score()


def main() -> None:
    for task in TASKS:
        score = run_task(task)
        print(f"{task}: score={score:.3f}")


if __name__ == "__main__":
    main()
