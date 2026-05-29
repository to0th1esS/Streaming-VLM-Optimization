import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_line_svg(path: Path, rows: List[Dict], x_key: str, y_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    width, height = 720, 420
    pad = 60
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 1.0

    def px(x):
        return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)

    def py(y):
        return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in zip(xs, ys))
    circles = "\n".join(
        f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="4" fill="#2563eb" />'
        for x, y in zip(xs, ys)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#111827"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#111827"/>
  <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>
  {circles}
  <text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13">{x_key}</text>
  <text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="Arial" font-size="13">{y_key}</text>
  <text x="{pad}" y="{height - pad + 22}" text-anchor="middle" font-family="Arial" font-size="11">{x_min:.2f}</text>
  <text x="{width - pad}" y="{height - pad + 22}" text-anchor="middle" font-family="Arial" font-size="11">{x_max:.2f}</text>
  <text x="{pad - 8}" y="{py(y_min):.2f}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.3f}</text>
  <text x="{pad - 8}" y="{py(y_max):.2f}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.3f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_bar_svg(path: Path, rows: List[Dict], label_key: str, value_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    width, height = 720, 420
    pad = 60
    labels = [str(row[label_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        max_value = 1.0

    bar_area_width = width - 2 * pad
    bar_width = bar_area_width / max(1, len(rows)) * 0.65
    gap = bar_area_width / max(1, len(rows))

    bars = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = pad + idx * gap + (gap - bar_width) / 2
        bar_height = value / max_value * (height - 2 * pad)
        y = height - pad - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#2563eb" />'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - pad + 18}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#111827"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#111827"/>
  {"".join(bars)}
  <text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="Arial" font-size="13">{value_key}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
