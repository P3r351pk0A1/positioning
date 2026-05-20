"""Small Tkinter visualizations for demonstration scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from .geometry import Vec3


@dataclass(frozen=True)
class TrajectoryEstimate:
    point: Vec3
    timestamp: float
    confidence: float
    status: str


@dataclass(frozen=True)
class TruthTrajectoryPoint:
    point: Vec3
    timestamp: float | None = None


def show_trajectory_window(
    microphones: Sequence[Vec3],
    truth_points: Sequence[Vec3],
    estimates: Sequence[TrajectoryEstimate],
    *,
    title: str = "Потоковая обработка: исходная и распознанная траектория",
) -> None:
    """Open a blocking popup window with a top-down trajectory view."""
    try:
        import tkinter as tk
    except ImportError as exc:  # pragma: no cover - depends on local Python build.
        raise RuntimeError("Tkinter недоступен в текущем Python-окружении") from exc

    root = tk.Tk()
    root.title(title)
    root.geometry("980x720")
    root.minsize(760, 560)

    header = tk.Label(
        root,
        text=title,
        anchor="w",
        font=("Arial", 14, "bold"),
        padx=14,
        pady=8,
    )
    header.pack(fill="x")

    ordered_estimates = sorted(estimates, key=lambda item: item.timestamp)

    info = tk.Label(
        root,
        text=(
            f"Микрофонов: {len(microphones)}    "
            f"Исходных точек: {len(truth_points)}    "
            f"Распознанных точек: {len(ordered_estimates)}"
        ),
        anchor="w",
        font=("Arial", 10),
        padx=14,
    )
    info.pack(fill="x")

    canvas = tk.Canvas(root, bg="#f8fafc", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=14, pady=10)

    footer = tk.Label(
        root,
        text=(
            "Синий - микрофоны; зеленый - исходная траектория/точка из metadata; "
            "красный - распознанные оценки по окнам; пунктир - связь соответствующих точек."
        ),
        anchor="w",
        font=("Arial", 9),
        padx=14,
        pady=8,
    )
    footer.pack(fill="x")

    all_points = [*microphones, *truth_points, *(estimate.point for estimate in ordered_estimates)]
    if not all_points:
        all_points = [Vec3(0.0, 0.0, 0.0)]
    min_x, max_x, min_y, max_y = _bounds_xy(all_points)

    def sx(x: float) -> float:
        width = max(1, canvas.winfo_width())
        margin = 70.0
        return margin + (x - min_x) / max(1e-9, max_x - min_x) * max(1.0, width - 2.0 * margin)

    def sy(y: float) -> float:
        height = max(1, canvas.winfo_height())
        margin = 70.0
        return height - margin - (y - min_y) / max(1e-9, max_y - min_y) * max(1.0, height - 2.0 * margin)

    def redraw(_event: object | None = None) -> None:
        canvas.delete("all")
        _draw_grid(canvas, min_x, max_x, min_y, max_y, sx, sy)

        if len(microphones) >= 2:
            mic_xy = [(sx(point.x), sy(point.y)) for point in microphones]
            for i in range(len(mic_xy)):
                x1, y1 = mic_xy[i]
                x2, y2 = mic_xy[(i + 1) % len(mic_xy)]
                canvas.create_line(x1, y1, x2, y2, fill="#2563eb", width=2)

        for index, point in enumerate(microphones):
            x, y = sx(point.x), sy(point.y)
            canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill="#2563eb", outline="")
            canvas.create_text(x + 14, y - 12, text=f"M{index}", fill="#1e3a8a", anchor="w", font=("Arial", 9, "bold"))

        _draw_polyline(canvas, truth_points, sx, sy, fill="#16a34a", width=3, smooth=False)
        for index, point in enumerate(truth_points):
            x, y = sx(point.x), sy(point.y)
            radius = 6 if len(truth_points) == 1 else 4
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#16a34a", outline="")
            if index == 0:
                canvas.create_text(x + 12, y + 12, text="исходная", fill="#166534", anchor="w", font=("Arial", 9, "bold"))

        estimate_points = [item.point for item in ordered_estimates]
        _draw_polyline(canvas, estimate_points, sx, sy, fill="#dc2626", width=2, smooth=False)
        connect_by_index = len(truth_points) == len(ordered_estimates) and len(truth_points) > 1
        for index, estimate in enumerate(ordered_estimates):
            x, y = sx(estimate.point.x), sy(estimate.point.y)
            radius = 4 + 4 * max(0.0, min(1.0, estimate.confidence))
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#dc2626", outline="")
            if len(truth_points) == 1:
                truth = truth_points[0]
                canvas.create_line(sx(truth.x), sy(truth.y), x, y, fill="#94a3b8", dash=(4, 4))
            elif connect_by_index:
                truth = truth_points[min(index, len(truth_points) - 1)]
                canvas.create_line(sx(truth.x), sy(truth.y), x, y, fill="#94a3b8", dash=(4, 4))
            if index == 0:
                canvas.create_text(x + 12, y - 12, text="распознано", fill="#991b1b", anchor="w", font=("Arial", 9, "bold"))

        _draw_legend(canvas)

    canvas.bind("<Configure>", redraw)
    root.after(50, redraw)
    root.mainloop()


def metadata_truth_points(metadata: dict[str, object] | None) -> list[Vec3]:
    return [item.point for item in metadata_truth_samples(metadata)]


def metadata_truth_samples(metadata: dict[str, object] | None) -> list[TruthTrajectoryPoint]:
    if not metadata:
        return []
    trajectory = metadata.get("trajectory")
    if isinstance(trajectory, list):
        points = []
        for item in trajectory:
            if not isinstance(item, dict):
                continue
            point = _vec_from_mapping(item)
            if point is None:
                continue
            timestamp = _float_or_none(item.get("timestamp"))
            points.append(TruthTrajectoryPoint(point, timestamp))
        return points
    target = metadata.get("target")
    if isinstance(target, dict):
        point = _vec_from_mapping(target)
        return [TruthTrajectoryPoint(point)] if point is not None else []
    return []


def _vec_from_mapping(value: dict[object, object]) -> Vec3 | None:
    try:
        x = float(value["x"])
        y = float(value["y"])
        z = float(value.get("z", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(isfinite(item) for item in (x, y, z)):
        return None
    return Vec3(x, y, z)


def _float_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _bounds_xy(points: Sequence[Vec3]) -> tuple[float, float, float, float]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 0.5)
    pad = max(0.15, span * 0.2)
    return min_x - pad, max_x + pad, min_y - pad, max_y + pad


def _draw_grid(canvas, min_x, max_x, min_y, max_y, sx, sy) -> None:
    for index in range(6):
        tx = min_x + (max_x - min_x) * index / 5.0
        ty = min_y + (max_y - min_y) * index / 5.0
        x = sx(tx)
        y = sy(ty)
        canvas.create_line(x, sy(min_y), x, sy(max_y), fill="#e2e8f0")
        canvas.create_line(sx(min_x), y, sx(max_x), y, fill="#e2e8f0")
        canvas.create_text(x, sy(min_y) + 18, text=f"{tx:.2f}", fill="#64748b", font=("Arial", 8))
        canvas.create_text(sx(min_x) - 25, y, text=f"{ty:.2f}", fill="#64748b", font=("Arial", 8))
    canvas.create_text(sx(max_x), sy(min_y) + 40, text="X, м", fill="#334155", anchor="e", font=("Arial", 9, "bold"))
    canvas.create_text(sx(min_x) - 45, sy(max_y), text="Y, м", fill="#334155", anchor="w", font=("Arial", 9, "bold"))


def _draw_polyline(canvas, points: Sequence[Vec3], sx, sy, *, fill: str, width: int, smooth: bool = False) -> None:
    if len(points) < 2:
        return
    coords = []
    for point in points:
        coords.extend([sx(point.x), sy(point.y)])
    canvas.create_line(*coords, fill=fill, width=width, smooth=smooth)


def _draw_legend(canvas) -> None:
    x0, y0 = 18, 16
    canvas.create_rectangle(x0, y0, x0 + 290, y0 + 78, fill="#ffffff", outline="#cbd5e1")
    items = [("#2563eb", "микрофоны"), ("#16a34a", "исходная траектория"), ("#dc2626", "распознанная траектория")]
    for index, (color, label) in enumerate(items):
        y = y0 + 18 + index * 20
        canvas.create_oval(x0 + 12, y - 5, x0 + 22, y + 5, fill=color, outline="")
        canvas.create_text(x0 + 32, y, text=label, anchor="w", fill="#334155", font=("Arial", 9))
