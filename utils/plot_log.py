import csv
import html
import math
import sys
from pathlib import Path


def latest_log() -> Path:
    logs = sorted(Path("logs").glob("*.csv"))
    if not logs:
        raise FileNotFoundError("no logs/*.csv files found")
    return logs[-1]


log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_log()
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.with_suffix(".svg")

steps = []
losses = []
with open(log_path) as f:
    for row in csv.DictReader(f):
        steps.append(float(row["step"]))
        losses.append(float(row["loss"]))

if not steps:
    raise ValueError(f"{log_path} has no rows")

width = 900
height = 500
left = 75
right = 25
top = 50
bottom = 65
plot_width = width - left - right
plot_height = height - top - bottom

min_step, max_step = min(steps), max(steps)
min_loss, max_loss = min(losses), max(losses)
step_span = max(max_step - min_step, 1)
loss_span = max(max_loss - min_loss, 1e-9)


def x_scale(step: float) -> float:
    return left + (step - min_step) / step_span * plot_width


def y_scale(loss: float) -> float:
    return top + (max_loss - loss) / loss_span * plot_height


def ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if min_value == max_value:
        return [min_value]
    step = (max_value - min_value) / (count - 1)
    return [min_value + i * step for i in range(count)]


points = []
for step, loss in zip(steps, losses):
    points.append(f"{x_scale(step):.2f},{y_scale(loss):.2f}")

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text { font-family: sans-serif; fill: #222; } .grid { stroke: #ddd; } .axis { stroke: #222; } .tick { stroke: #222; }</style>',
    f'<text x="{left}" y="28" font-size="18" font-weight="700">training loss</text>',
    f'<text x="{width - right}" y="28" font-size="12" text-anchor="end">{html.escape(str(log_path))}</text>',
]

for loss in ticks(min_loss, max_loss):
    y = y_scale(loss)
    parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
    parts.append(f'<line class="tick" x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}"/>')
    parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{loss:.4g}</text>')

for step in ticks(min_step, max_step):
    x = x_scale(step)
    parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}"/>')
    parts.append(f'<line class="tick" x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" y2="{height - bottom + 5}"/>')
    parts.append(f'<text x="{x:.2f}" y="{height - bottom + 22}" font-size="12" text-anchor="middle">{step:.0f}</text>')

parts.extend([
    f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
    f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
    f'<polyline points="{" ".join(points)}" fill="none" stroke="steelblue" stroke-width="2"/>',
    f'<circle cx="{x_scale(steps[-1]):.2f}" cy="{y_scale(losses[-1]):.2f}" r="4" fill="steelblue"/>',
    f'<text x="{width / 2}" y="{height - 18}" font-size="14" text-anchor="middle">step</text>',
    f'<text x="18" y="{height / 2}" font-size="14" text-anchor="middle" transform="rotate(-90 18 {height / 2})">loss</text>',
    f'<text x="{width - right}" y="{height - bottom - 10}" font-size="12" text-anchor="end">last: step {steps[-1]:.0f}, loss {losses[-1]:.4g}</text>',
    "</svg>",
])

svg = "\n".join(parts)

out_path.write_text(svg)
print(f"Wrote {out_path}")
