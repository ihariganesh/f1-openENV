from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env import F1StrategyEnv
from models import ActionSpace, ResetResponse, StepResponse
from tasks import list_task_specs

app = FastAPI(title="Race Strategy Optimizer OpenEnv")
env = F1StrategyEnv()


UI_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>F1 Race Strategy Optimizer</title>
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --ink: #e5e7eb;
            --muted: #9ca3af;
            --accent: #22c55e;
            --warn: #f59e0b;
            --danger: #ef4444;
            --info: #3b82f6;
            --line: #1f2937;
        }
        * { box-sizing: border-box; margin: 0; }
        body {
            font-family: 'Inter', ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
            background: radial-gradient(circle at top right, #1f2937 0%, var(--bg) 50%);
            color: var(--ink);
            min-height: 100vh;
        }
        .wrap { max-width: 1080px; margin: 32px auto; padding: 0 16px; }
        .head { margin-bottom: 24px; }
        .title { font-size: 30px; font-weight: 700; margin: 0 0 6px; }
        .sub { color: var(--muted); margin: 0; font-size: 14px; }
        .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
        .card {
            background: color-mix(in srgb, var(--panel) 85%, black 15%);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 16px;
        }
        .card h3 { margin: 0 0 12px; font-size: 16px; }
        label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 4px; }
        input, select {
            width: 100%;
            background: #0b1220;
            border: 1px solid #243244;
            color: var(--ink);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 8px;
            font-size: 14px;
        }
        button {
            border: 0;
            border-radius: 10px;
            padding: 10px 14px;
            background: var(--accent);
            color: #052e16;
            font-weight: 700;
            cursor: pointer;
            margin-right: 8px;
            font-size: 14px;
            transition: filter 0.2s;
        }
        button:hover { filter: brightness(1.15); }
        button.alt { background: var(--warn); color: #422006; }
        button.info-btn { background: var(--info); color: white; }
        .status-bar {
            display: flex; gap: 12px; flex-wrap: wrap;
            margin-bottom: 14px;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-green { background: #16a34a22; color: #22c55e; border: 1px solid #16a34a44; }
        .badge-amber { background: #b4530044; color: #f59e0b; border: 1px solid #b4530066; }
        .badge-red { background: #dc262644; color: #ef4444; border: 1px solid #dc262666; }
        pre {
            background: #020617;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 14px;
            color: #cbd5e1;
            overflow: auto;
            min-height: 240px;
            font-size: 13px;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="head">
            <h1 class="title">🏎️ F1 Race Strategy Optimizer</h1>
            <p class="sub">Interactive UI for manual environment testing. OpenEnv-compatible API at POST /reset, /step, /state.</p>
        </div>
        <div class="status-bar" id="statusBar"></div>
        <div class="grid">
            <div class="card">
                <h3>Episode Control</h3>
                <label>Task</label>
                <select id="task"></select>
                <label>Seed</label>
                <input id="seed" type="number" value="7" />
                <button onclick="resetEnv()">Reset</button>
                <button class="alt" onclick="fetchState()">State</button>
                <button class="info-btn" onclick="autoRun()">Auto-Run</button>
            </div>
            <div class="card">
                <h3>Step Action</h3>
                <label>Action Type</label>
                <select id="actionType">
                    <option value="HOLD" selected>HOLD</option>
                    <option value="PIT">PIT</option>
                </select>
                <label>Pit Compound</label>
                <select id="pitCompound">
                    <option value="">(none)</option>
                    <option value="SOFT">SOFT</option>
                    <option value="MEDIUM">MEDIUM</option>
                    <option value="HARD">HARD</option>
                    <option value="INTER">INTER</option>
                    <option value="WET">WET</option>
                </select>
                <label>Pace Mode</label>
                <select id="paceMode">
                    <option value="PUSH">PUSH</option>
                    <option value="BALANCED" selected>BALANCED</option>
                    <option value="CONSERVE">CONSERVE</option>
                </select>
                <button onclick="stepEnv()">Step</button>
            </div>
            <div class="card" style="grid-column: 1 / -1;">
                <h3>Response</h3>
                <pre id="out">Loading tasks...</pre>
            </div>
        </div>
    </div>
    <script>
        let currentLap = 0;
        let totalLaps = 0;
        let cumulativeReward = 0;

        function updateStatus(data) {
            const bar = document.getElementById('statusBar');
            if (!data || !data.data) { bar.innerHTML = ''; return; }
            const d = data.data;
            const obs = d.observation || {};
            currentLap = obs.current_lap || 0;
            totalLaps = obs.total_laps || 0;
            cumulativeReward += (d.reward || 0);

            let badges = [];
            badges.push(`<span class="badge badge-green">Lap ${currentLap}/${totalLaps}</span>`);
            if (obs.current_tire_compound) badges.push(`<span class="badge badge-amber">🏁 ${obs.current_tire_compound} (age ${obs.tire_age_laps})</span>`);
            const wear = obs.tire_wear_percentage || 0;
            const wearClass = wear > 0.8 ? 'badge-red' : wear > 0.5 ? 'badge-amber' : 'badge-green';
            badges.push(`<span class="badge ${wearClass}">Wear: ${(wear*100).toFixed(0)}%</span>`);
            if (obs.fuel_kg != null) badges.push(`<span class="badge badge-green">⛽ ${obs.fuel_kg.toFixed(0)}kg</span>`);
            if (obs.safety_car_active) badges.push(`<span class="badge badge-amber">🚗 Safety Car</span>`);
            if (obs.drs_available) badges.push(`<span class="badge badge-green">DRS</span>`);
            if (d.done) badges.push(`<span class="badge badge-red">FINISHED</span>`);
            bar.innerHTML = badges.join('');
        }

        async function api(path, payload) {
            const res = await fetch(path, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload || {})
            });
            const data = await res.json();
            return { ok: res.ok, status: res.status, data };
        }

        function show(data) {
            document.getElementById('out').textContent = JSON.stringify(data, null, 2);
            updateStatus(data);
        }

        async function loadTasks() {
            const res = await fetch('/tasks');
            const tasks = await res.json();
            const sel = document.getElementById('task');
            sel.innerHTML = '';
            for (const t of tasks) {
                const o = document.createElement('option');
                o.value = t.name;
                o.textContent = `${t.name} (${t.difficulty})`;
                sel.appendChild(o);
            }
            show({ status: 'ready', tasks });
        }

        async function resetEnv() {
            cumulativeReward = 0;
            const task = document.getElementById('task').value;
            const seed = parseInt(document.getElementById('seed').value || '7', 10);
            show(await api('/reset', { task, seed }));
        }

        async function stepEnv() {
            const action_type = document.getElementById('actionType').value;
            const c = document.getElementById('pitCompound').value;
            const pace_mode = document.getElementById('paceMode').value;
            const payload = {
                pit_stop: action_type === 'PIT',
                new_compound: action_type === 'PIT' ? (c || null) : null,
                pace_mode,
            };
            show(await api('/step', payload));
        }

        async function fetchState() {
            show(await api('/state', {}));
        }

        async function autoRun() {
            cumulativeReward = 0;
            const task = document.getElementById('task').value;
            const seed = parseInt(document.getElementById('seed').value || '7', 10);
            let result = await api('/reset', { task, seed });
            show(result);

            const maxSteps = 80;
            for (let i = 0; i < maxSteps; i++) {
                if (result.data?.done) break;
                await new Promise(r => setTimeout(r, 100));
                result = await api('/step', { pit_stop: false, new_compound: null, pace_mode: 'BALANCED' });
                show(result);
            }
            show(await api('/state', {}));
        }

        loadTasks();
    </script>
</body>
</html>
"""


class ResetPayload(BaseModel):
    task: str = "f1-sprint-dry"
    seed: int = 7


class StatePayload(BaseModel):
    include_internal: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "benchmark": "race_strategy_optimizer"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return UI_HTML


@app.get("/tasks")
def tasks() -> list[dict[str, object]]:
    return [t.model_dump() for t in list_task_specs()]


@app.post("/reset", response_model=ResetResponse)
def reset(payload: ResetPayload | None = None) -> ResetResponse:
    payload = payload or ResetPayload()
    return env.reset(task_name=payload.task, seed=payload.seed)


@app.post("/step", response_model=StepResponse)
def step(action: ActionSpace) -> StepResponse:
    return env.step(action)


@app.post("/state")
def state(_: StatePayload | None = None) -> dict[str, object]:
    return env.state()
