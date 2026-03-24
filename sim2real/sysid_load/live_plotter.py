"""Interactive playback viewer for load_sim2real NPZ recordings.

Features:
- Multi-panel dashboard with position, velocity, torque, and parameters
- Sliding window playback (default 10 seconds)
- Play/pause, speed control, and seek functionality
- Live RMSE statistics comparing sim vs real
- Interactive zoom, pan, hover tooltips
- Annotation support for marking events

Usage:
    python live_plotter.py data/20240324_120000_load_sysid.npz
    python live_plotter.py --window 30  # 30 second window
    python live_plotter.py --list       # List available data files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

# ── Constants ───────────────────────────────────────────────────────────────────
DEFAULT_WINDOW = 10.0  # seconds
UPDATE_INTERVAL = 50  # ms (20 Hz UI update)
SPEED_OPTIONS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# ── Global data storage ─────────────────────────────────────────────────────────
DATA = {}


def load_npz_data(filepath: Path) -> dict:
    """Load NPZ recording and return as dict with metadata."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data_raw = np.load(str(filepath))

    # Extract all arrays
    data = {
        "t": data_raw["t"],
        "target": data_raw["target"],
        "q_sim": data_raw["q_sim"],
        "dq_sim": data_raw["dq_sim"],
        "tau_sim": data_raw["tau_sim"],
        "q_real": data_raw["q_real"],
        "dq_real": data_raw["dq_real"],
        "tau_real": data_raw["tau_real"],
        "kp": data_raw["kp"],
        "kd": data_raw["kd"],
        "damping": data_raw["damping"],
        "armature": data_raw["armature"],
        "frictionloss": data_raw["frictionloss"],
        "control_hz": float(data_raw["control_hz"][0]),
    }

    # Metadata
    data["duration"] = data["t"][-1] if len(data["t"]) > 0 else 0.0
    data["num_samples"] = len(data["t"])
    data["filepath"] = filepath

    return data


def compute_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray = None
) -> float:
    """Compute RMSE between two signals with optional mask."""
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)

    if not np.any(mask):
        return 0.0

    diff = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(diff**2)))


def create_dashboard_layout(window_size: float) -> dash.Dash:
    """Create Dash app with multi-panel layout."""
    app = dash.Dash(__name__, title="Load Sim2Real Viewer")

    app.layout = html.Div(
        [
            # Header
            html.Div(
                [
                    html.H1(
                        "Load Sim2Real Interactive Viewer", style={"margin": "10px"}
                    ),
                    html.Div(
                        id="file-info", style={"margin": "10px", "fontSize": "14px"}
                    ),
                ],
                style={"backgroundColor": "#f0f0f0", "padding": "10px"},
            ),
            # Control panel
            html.Div(
                [
                    html.Button(
                        "⏯ Play/Pause",
                        id="btn-play",
                        n_clicks=0,
                        style={"margin": "5px", "fontSize": "16px"},
                    ),
                    html.Button(
                        "⏮ Reset",
                        id="btn-reset",
                        n_clicks=0,
                        style={"margin": "5px", "fontSize": "16px"},
                    ),
                    html.Label("Speed:", style={"margin": "5px 10px 5px 20px"}),
                    dcc.Dropdown(
                        id="speed-dropdown",
                        options=[{"label": f"{s}x", "value": s} for s in SPEED_OPTIONS],
                        value=1.0,
                        clearable=False,
                        style={"width": "100px", "display": "inline-block"},
                    ),
                    html.Label("Window:", style={"margin": "5px 10px 5px 20px"}),
                    dcc.Input(
                        id="window-input",
                        type="number",
                        value=window_size,
                        min=1,
                        max=60,
                        step=1,
                        style={"width": "80px", "margin": "5px"},
                    ),
                    html.Label("s", style={"margin": "5px"}),
                ],
                style={"backgroundColor": "#e0e0e0", "padding": "10px"},
            ),
            # Playback progress
            html.Div(
                [
                    html.Div(id="time-display", style={"marginBottom": "5px", "fontSize": "13px", "fontFamily": "monospace"}),
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=100,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                        updatemode="mouseup",
                    ),
                ],
                style={"margin": "20px"},
            ),
            # Statistics panel
            html.Div(
                id="stats-panel",
                style={
                    "backgroundColor": "#fff3cd",
                    "padding": "10px",
                    "margin": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "fontFamily": "monospace",
                },
            ),
            # Main graph
            dcc.Graph(
                id="main-graph",
                style={"height": "85vh"},
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            ),
            # Hidden state stores
            dcc.Store(
                id="playback-state",
                data={
                    "playing": False,
                    "current_time": 0.0,
                    "speed": 1.0,
                    "window": window_size,
                },
            ),
            dcc.Store(id="seek-store", data={"value": 0.0, "version": 0}),
            # Update interval
            dcc.Interval(id="interval-update", interval=UPDATE_INTERVAL, n_intervals=0),
        ]
    )

    return app


def create_plots(data: dict, current_time: float, window: float):
    """Create multi-panel plot for current time window."""
    # Time window
    t_min = max(0, current_time - window)
    t_max = current_time

    # Find indices in window
    mask = (data["t"] >= t_min) & (data["t"] <= t_max)

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Position Tracking",
            "Velocity Comparison",
            "Torque Comparison",
            "Physics Parameters",
        ),
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
    )

    # ── Panel 1: Position ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["target"][mask],
            name="target",
            line=dict(color="#f5a623", dash="dashdot", width=2.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["q_sim"][mask],
            name="q_sim",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["q_real"][mask],
            name="q_real",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=1,
    )

    # ── Panel 2: Velocity ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["dq_sim"][mask],
            name="dq_sim",
            line=dict(color="cyan", width=2),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["dq_real"][mask],
            name="dq_real",
            line=dict(color="orange", width=2),
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    # ── Panel 3: Torque ─────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["tau_sim"][mask],
            name="tau_sim",
            line=dict(color="green", width=2),
            showlegend=True,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["tau_real"][mask],
            name="tau_real",
            line=dict(color="magenta", width=2),
            showlegend=True,
        ),
        row=3,
        col=1,
    )

    # ── Panel 4: Parameters ─────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["kp"][mask],
            name="kp",
            line=dict(color="purple", width=2),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["kd"][mask],
            name="kd",
            line=dict(color="brown", width=2),
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["damping"][mask],
            name="damping",
            line=dict(color="pink", width=2),
        ),
        row=4,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["armature"][mask],
            name="armature",
            line=dict(color="olive", width=2),
        ),
        row=4,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=data["t"][mask],
            y=data["frictionloss"][mask],
            name="frictionloss",
            line=dict(color="teal", width=2),
        ),
        row=4,
        col=1,
        secondary_y=True,
    )

    # ── Axis labels ─────────────────────────────────────────────────────────────
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Position (rad)", row=1, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (rad/s)", row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Torque (Nm)", row=3, col=1)

    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Gains (kp, kd)", row=4, col=1)
    fig.update_yaxes(
        title_text="Params (damping, etc.)", row=4, col=1, secondary_y=True
    )

    # ── Layout ──────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )

    # Add vertical line at current time
    for row in range(1, 5):
        fig.add_vline(
            x=current_time,
            line_dash="dash",
            line_color="black",
            opacity=0.5,
            row=row,
            col=1,
        )

    return fig


def compute_statistics(data: dict, current_time: float, window: float) -> str:
    """Compute and format statistics for current window."""
    t_min = max(0, current_time - window)
    mask = (data["t"] >= t_min) & (data["t"] <= current_time)

    if not np.any(mask):
        return "No data in window"

    # Compute RMSE values
    q_rmse = compute_rmse(data["q_real"], data["q_sim"], mask)
    dq_rmse = compute_rmse(data["dq_real"], data["dq_sim"], mask)
    tau_rmse = compute_rmse(data["tau_real"], data["tau_sim"], mask)

    # Mean values in window
    q_real_mean = np.mean(data["q_real"][mask])
    q_sim_mean = np.mean(data["q_sim"][mask])

    # Format statistics
    stats = f"""
📊 STATISTICS (Window: {t_min:.2f}s - {current_time:.2f}s)
├─ Position RMSE:  {q_rmse:.4f} rad  (Real mean: {q_real_mean:+.3f}, Sim mean: {q_sim_mean:+.3f})
├─ Velocity RMSE:  {dq_rmse:.4f} rad/s
├─ Torque RMSE:    {tau_rmse:.4f} Nm
└─ Samples in window: {np.sum(mask)} / {len(data["t"])} total
    """.strip()

    return stats


def setup_callbacks(app: dash.Dash):
    """Setup all Dash callbacks for interactivity."""

    @app.callback(
        Output("file-info", "children"), Input("playback-state", "data")
    )
    def update_file_info(_):
        if not DATA:
            return "No data loaded"
        return (
            f"📁 {DATA['filepath'].name} | "
            f"Duration: {DATA['duration']:.2f}s | "
            f"Samples: {DATA['num_samples']} | "
            f"Rate: {DATA['control_hz']:.1f} Hz"
        )

    @app.callback(
        Output("time-slider", "max"),
        Output("time-slider", "marks"),
        Input("interval-update", "n_intervals"),
    )
    def update_slider_range(_):
        if not DATA:
            return 100, {}

        duration = DATA["duration"]
        marks = {i * duration / 10: f"{i * duration / 10:.1f}s" for i in range(11)}
        return duration, marks

    @app.callback(
        Output("seek-store", "data"),
        Input("time-slider", "value"),
        State("seek-store", "data"),
        prevent_initial_call=True,
    )
    def record_seek(slider_value, seek):
        """User dragged the slider — record a seek request."""
        return {"value": slider_value, "version": seek["version"] + 1}

    @app.callback(
        Output("playback-state", "data"),
        Input("btn-play", "n_clicks"),
        Input("btn-reset", "n_clicks"),
        Input("speed-dropdown", "value"),
        Input("window-input", "value"),
        Input("interval-update", "n_intervals"),
        Input("seek-store", "data"),
        State("playback-state", "data"),
        prevent_initial_call=True,
    )
    def update_playback_state(
        play_clicks, reset_clicks, speed, window, n_intervals, seek, state
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            return state

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Play/Pause button
        if trigger_id == "btn-play":
            state["playing"] = not state["playing"]

        # Reset button
        elif trigger_id == "btn-reset":
            state["current_time"] = 0.0
            state["playing"] = False

        # Speed change
        elif trigger_id == "speed-dropdown":
            state["speed"] = speed

        # Window size change
        elif trigger_id == "window-input":
            state["window"] = window

        # User seeked via slider
        elif trigger_id == "seek-store":
            state["current_time"] = seek["value"]
            state["playing"] = False

        # Interval tick (playback advance)
        elif trigger_id == "interval-update" and state["playing"]:
            if DATA:
                dt = (UPDATE_INTERVAL / 1000.0) * state["speed"]
                state["current_time"] += dt
                if state["current_time"] > DATA["duration"]:
                    state["current_time"] = 0.0

        return state

    @app.callback(
        Output("time-display", "children"),
        Input("playback-state", "data"),
        prevent_initial_call=True,
    )
    def update_time_display(state):
        if not DATA:
            return ""
        t = state["current_time"]
        dur = DATA["duration"]
        status = "▶ Playing" if state["playing"] else "⏸ Paused"
        return f"{status}  |  {t:.2f}s / {dur:.2f}s  ({t/dur*100:.1f}%)"

    @app.callback(
        Output("main-graph", "figure"),
        Output("stats-panel", "children"),
        Input("playback-state", "data"),
        prevent_initial_call=True,
    )
    def update_plots(state):
        if not DATA:
            return {}, "No data loaded"

        current_time = state["current_time"]
        window = state["window"]

        fig = create_plots(DATA, current_time, window)
        stats = compute_statistics(DATA, current_time, window)

        return fig, stats


def list_data_files(data_dir: Path):
    """List all NPZ files in data directory."""
    npz_files = sorted(data_dir.glob("*.npz"))

    if not npz_files:
        print(f"No NPZ files found in {data_dir}")
        return

    print(f"\n📂 Available data files in {data_dir}:\n")
    for i, f in enumerate(npz_files, 1):
        try:
            data = np.load(str(f))
            duration = data["t"][-1]
            samples = len(data["t"])
            rate = data["control_hz"][0]
            print(f"  {i}. {f.name}")
            print(
                f"     Duration: {duration:.2f}s, Samples: {samples}, Rate: {rate:.1f} Hz"
            )
        except Exception as e:
            print(f"  {i}. {f.name} (error: {e})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for load_sim2real recordings"
    )
    parser.add_argument("file", nargs="?", help="Path to NPZ file")
    parser.add_argument(
        "--window",
        "-w",
        type=float,
        default=DEFAULT_WINDOW,
        help=f"Sliding window size in seconds (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8050,
        help="Port for Dash server (default: 8050)",
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available data files"
    )

    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"

    # List mode
    if args.list:
        list_data_files(data_dir)
        return

    # Check for file
    if not args.file:
        print("Error: No file specified. Use --list to see available files.")
        list_data_files(data_dir)
        sys.exit(1)

    # Load data
    filepath = Path(args.file)
    if not filepath.is_absolute():
        filepath = data_dir / filepath

    try:
        global DATA
        DATA = load_npz_data(filepath)
        print(f"✓ Loaded {filepath.name}")
        print(f"  Duration: {DATA['duration']:.2f}s")
        print(f"  Samples: {DATA['num_samples']}")
        print(f"  Rate: {DATA['control_hz']:.1f} Hz")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Create and run app
    app = create_dashboard_layout(args.window)
    setup_callbacks(app)

    print(f"\n🚀 Starting interactive viewer on http://localhost:{args.port}")
    print("   Press Ctrl+C to stop")

    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
