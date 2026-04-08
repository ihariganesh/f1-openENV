from env import F1StrategyEnv
from models import ActionSpace


def test_reset_step_state_contract() -> None:
    env = F1StrategyEnv()
    reset = env.reset(task_name="f1-sprint-dry", seed=7)
    assert reset.done is False
    assert reset.observation.current_lap == 1
    assert reset.observation.fuel_kg > 0

    step = env.step(ActionSpace(pit_stop=False, new_compound=None, pace_mode="BALANCED"))
    assert isinstance(step.reward, float)
    assert step.observation.pit_stop_count == 0

    st = env.state()
    assert st["task_name"] == "f1-sprint-dry"
    assert st["initialized"] is True


def test_pit_stop_increments_count() -> None:
    env = F1StrategyEnv()
    env.reset(task_name="f1-sprint-dry", seed=7)
    env.step(ActionSpace(pit_stop=False, new_compound=None, pace_mode="BALANCED"))
    res = env.step(ActionSpace(pit_stop=True, new_compound="MEDIUM", pace_mode="BALANCED"))
    assert res.observation.pit_stop_count == 1
    assert res.observation.tire_wear_percentage < 0.1  # Fresh tires, minimal wear


def test_fuel_decreases() -> None:
    env = F1StrategyEnv()
    env.reset(task_name="f1-sprint-dry", seed=7)
    initial_fuel = env.state()["fuel_kg"]
    env.step(ActionSpace(pit_stop=False, new_compound=None, pace_mode="BALANCED"))
    after_fuel = env.state()["fuel_kg"]
    assert after_fuel < initial_fuel


def test_cumulative_time_increases() -> None:
    env = F1StrategyEnv()
    env.reset(task_name="f1-sprint-dry", seed=7)
    env.step(ActionSpace(pit_stop=False, new_compound=None, pace_mode="BALANCED"))
    st = env.state()
    assert st["cumulative_race_time_seconds"] > 0
