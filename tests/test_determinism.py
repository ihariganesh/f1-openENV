from env import F1StrategyEnv
from models import ActionSpace


def run_episode() -> tuple[float, int]:
    env = F1StrategyEnv()
    env.reset("f1-sprint-dry", seed=7)
    done = False
    reward_sum = 0.0
    steps = 0
    while not done:
        res = env.step(ActionSpace(pit_stop=False, new_compound=None, pace_mode="BALANCED"))
        reward_sum += res.reward
        done = res.done
        steps += 1
    state = env.state()
    return round(reward_sum, 5), int(state["current_lap"])


def test_seed_determinism() -> None:
    a = run_episode()
    b = run_episode()
    assert a == b
