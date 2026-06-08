import csv
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
pad = 50

min_step, max_step = min(steps), max(steps)
min_loss, max_loss = min(losses), max(losses)
step_span = max(max_step - min_step, 1)
loss_span = max(max_loss - min_loss, 1e-9)

points = []
for step, loss in zip(steps, losses):
    x = pad + (step - min_step) / step_span * (width - 2 * pad)
    y = height - pad - (loss - min_loss) / loss_span * (height - 2 * pad)
    points.append(f"{x:.2f},{y:.2f}")

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="black"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="black"/>
  <polyline points="{" ".join(points)}" fill="none" stroke="steelblue" stroke-width="2"/>
  <text x="{pad}" y="25" font-family="sans-serif" font-size="16">{log_path}</text>
  <text x="{pad}" y="{height - 15}" font-family="sans-serif" font-size="12">step {min_step:.0f} - {max_step:.0f}</text>
  <text x="{width - 180}" y="{height - 15}" font-family="sans-serif" font-size="12">loss {min_loss:.4f} - {max_loss:.4f}</text>
</svg>
"""

out_path.write_text(svg)
print(f"Wrote {out_path}")
