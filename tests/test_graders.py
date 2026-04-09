from tasks import grade_episode, solve_optimal_total_time


def test_grader_bounds() -> None:
    for task in ["f1-sprint-dry", "f1-feature-safetycar", "f1-chaos-weather"]:
        optimum = solve_optimal_total_time(task)
        score = grade_episode(task, optimum).score
        assert 0.0 <= score <= 1.0


def test_grader_not_constant() -> None:
    optimum = solve_optimal_total_time("f1-sprint-dry")
    low = grade_episode("f1-sprint-dry", optimum + 80.0).score
    high = grade_episode("f1-sprint-dry", optimum + 5.0).score
    assert high > low


def test_optimal_is_perfect_score() -> None:
    for task in ["f1-sprint-dry", "f1-feature-safetycar", "f1-chaos-weather"]:
        optimum = solve_optimal_total_time(task)
        result = grade_episode(task, optimum)
        assert 0.0 < result.score < 1.0
        assert result.score > 0.99


def test_bad_time_low_score() -> None:
    optimum = solve_optimal_total_time("f1-sprint-dry")
    result = grade_episode("f1-sprint-dry", optimum + 100.0)
    assert 0.0 < result.score < 1.0
    assert result.score < 0.01


def test_weather_task_optimal_under_60_laps() -> None:
    optimum = solve_optimal_total_time("f1-chaos-weather")
    # 60 laps * ~90s/lap = ~5400s baseline
    assert optimum < 6000.0, f"Optimal time {optimum}s seems too high"
    assert optimum > 3000.0, f"Optimal time {optimum}s seems too low"
