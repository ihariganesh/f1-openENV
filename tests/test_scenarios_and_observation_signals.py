from env import F1StrategyEnv
from models import ActionSpace
from tasks import TASKS, grade_episode, solve_optimal_total_time


def test_openenv_alias_tasks_reset_step() -> None:
    env = F1StrategyEnv()
    for task_name in ["easy-one-stop", "medium-strategy", "hard-safety-car", "ultra-chaos"]:
        reset = env.reset(task_name=task_name, seed=7)
        assert reset.observation.current_lap == 1
        step = env.step(ActionSpace(pit_stop=False, new_compound=None, pace_mode="BALANCED"))
        assert 0.0 <= step.reward <= 1.0


def test_all_task_graders_are_bounded() -> None:
    for task_name in TASKS.keys():
        optimum = solve_optimal_total_time(task_name)
        result = grade_episode(task_name, optimum)
        assert 0.0 <= result.score <= 1.0


def test_observation_dynamic_signals_change_in_weather_task() -> None:
    env = F1StrategyEnv()
    env.reset(task_name="f1-chaos-weather", seed=7)

    first_obs = env.step(ActionSpace(pit_stop=False, new_compound=None, pace_mode="BALANCED")).observation
    for _ in range(34):
        obs = env.step(ActionSpace(pit_stop=False, new_compound=None, pace_mode="BALANCED")).observation

    assert obs.track_wetness >= first_obs.track_wetness
    assert obs.track_temperature_c <= first_obs.track_temperature_c