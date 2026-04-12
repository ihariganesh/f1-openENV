from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Optional

from env import F1StrategyEnv
from models import ActionSpace, TireCompound
from tasks import TASKS, TaskConfig, solve_optimal_total_time


def setup_abu_dhabi_task():
    """
    Register Abu Dhabi (Yas Marina) task in the global TASKS registry.
    This configuration mimics the 2021 season finale characteristics.
    """
    abu_dhabi = TaskConfig(
        name="f1-abu-dhabi",
        difficulty="medium",
        description=(
            "58 lap full-length race at the Yas Marina Circuit, Abu Dhabi. "
            "Track temp starts high and drops as night falls. "
            "High pit lane time loss (~23s). Safety car possibility late in the race."
        ),
        total_laps=58,
        start_compound="MEDIUM",
        safety_car_laps=(53, 54, 55),  # Reference to the famous late SC
        pit_penalty_normal=23.0,
        pit_penalty_safety_car=13.0,
        tolerance_seconds=30.0,
        track_temp_base=32.0,
        drs_laps=tuple(range(3, 59)),
    )
    TASKS["f1-abu-dhabi"] = abu_dhabi
    return abu_dhabi


class StrategyEngineer:
    """
    A rule-based expert agent that simulates race strategy decisions.
    """
    def __init__(self, task: TaskConfig):
        self.task = task
        self.planned_pit_lap = 24
        self.has_pitted_sc = False

    def decide(self, obs) -> ActionSpace:
        lap = obs.current_lap
        compound = obs.current_tire_compound
        wear = obs.tire_wear_percentage
        sc = obs.safety_car_active
        fuel = obs.fuel_kg

        # 1. Pit Stop Logic
        if obs.pit_stop_count == 0:
            # Standard window for M -> H
            if lap >= self.planned_pit_lap:
                if sc or wear > 0.40 or lap >= 28:
                    return ActionSpace(pit_stop=True, new_compound="HARD", pace_mode="BALANCED")
        
        # 2. Late Race Safety Car Logic (The 'Gamble')
        if sc and lap >= 53 and obs.pit_stop_count == 1 and not self.has_pitted_sc:
            self.has_pitted_sc = True
            return ActionSpace(pit_stop=True, new_compound="SOFT", pace_mode="PUSH")

        # 3. Pace Mode Logic
        pace = "BALANCED"
        
        # Fuel management: if fuel is getting low, conserve
        fuel_per_lap = fuel / max(1, (self.task.total_laps - lap + 1))
        if fuel_per_lap < 1.4:
            pace = "CONSERVE"
        # Push on fresh tires or end of race
        elif (obs.tire_age_laps < 3 and wear < 0.15) or (lap >= self.task.total_laps - 1):
            pace = "PUSH"
        # Conserve if tires are critical
        elif wear > 0.75:
            pace = "CONSERVE"
        # Push if we have a lot of fuel and tires are okay
        elif fuel > 40 and wear < 0.4:
            pace = "PUSH"

        return ActionSpace(pit_stop=False, new_compound=None, pace_mode=pace)


def run_full_telemetry_sim():
    task_cfg = setup_abu_dhabi_task()
    env = F1StrategyEnv()
    
    print("\n" + "=" * 80)
    print(f" RACE SIMULATION: {task_cfg.name.upper()} ".center(80, "="))
    print("=" * 80)
    print(f"Circuit: Yas Marina")
    print(f"Distance: {task_cfg.total_laps} Laps")
    print(f"Initial Compound: {task_cfg.start_compound}")
    print("-" * 80)

    res = env.reset(task_name="f1-abu-dhabi", seed=7)
    engineer = StrategyEngineer(task_cfg)
    
    telemetry = []
    done = False
    
    header = f"{'LAP':<4} | {'Tire':<10} | {'Age':<3} | {'Wear':<6} | {'Fuel':<6} | {'LapTime':<8} | {'Pace':<9} | {'Status'}"
    print(header)
    print("-" * len(header))

    while not done:
        obs = res.observation
        action = engineer.decide(obs)
        
        # Store telemetry
        lap_data = {
            "lap": obs.current_lap,
            "tire": obs.current_tire_compound,
            "age": obs.tire_age_laps,
            "wear": f"{obs.tire_wear_percentage*100:.1f}%",
            "fuel": f"{obs.fuel_kg:.1f}kg",
            "pace": action.pace_mode,
            "sc": obs.safety_car_active,
            "pit": action.pit_stop
        }
        
        res = env.step(action)
        done = res.done
        
        # Logging
        status = ""
        if action.pit_stop: status += "BOX "
        if obs.safety_car_active: status += "SC "
        if obs.drs_available: status += "DRS "
        
        if obs.current_lap % 2 == 0 or action.pit_stop or done:
            print(f"{obs.current_lap:<4} | {obs.current_tire_compound:<10} | {obs.tire_age_laps:<3} | "
                  f"{obs.tire_wear_percentage*100:>5.1f}% | {obs.fuel_kg:>5.1f}kg | "
                  f"{obs.last_lap_time_seconds:>7.3f}s | {action.pace_mode:<9} | {status}")

    print("-" * len(header))
    state = env.state()
    print(f"FINISH TIME: {state['cumulative_race_time_seconds']:.3f}s")
    print(f"GRADER SCORE: {state['grader_score']:.6f}")
    print(f"PIT STOPS: {state['pit_stop_count']}")
    print(f"FINAL FUEL: {state['fuel_kg']:.2f}kg")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_telemetry_sim()
