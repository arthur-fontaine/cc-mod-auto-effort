"""Draw the benchmark as a chart: right level against latency, one point per model.

    uv run python pipeline/plot.py     # -> ../docs/benchmark.svg

Reads results/bench-*.json, like report.py. The SVG follows the viewer's light or dark mode.
"""
import json
import math
import re
from pathlib import Path

from report import BENCH, ROOT

OUT = ROOT.parent / "docs" / "benchmark.svg"
W, H = 760, 470
LEFT, RIGHT, TOP, BOTTOM = 72, 28, 56, 64
X_MIN, X_MAX = 100, 4000  # ms, log scale
Y_MIN, Y_MAX = 40, 80  # % right level
# The zone to aim for: under the Doherty threshold (people stay engaged when a system answers
# within 400 ms; Doherty and Thadani, IBM, 1982), and clearly above keeping the default.
GOAL_MS, GOAL_PCT = 400, 60
DEFAULT_PCT = 52.9


def x(ms):
    return LEFT + (math.log10(ms) - math.log10(X_MIN)) / (math.log10(X_MAX) - math.log10(X_MIN)) * (W - LEFT - RIGHT)


def y(pct):
    return TOP + (Y_MAX - pct) / (Y_MAX - Y_MIN) * (H - TOP - BOTTOM)


def points():
    for name, label in BENCH:
        path = ROOT / "results" / f"bench-{name}.json"
        if path.exists():
            r = json.loads(path.read_text())
            where = r["bench"]["where"]
            yield {"label": re.sub(r"\*\*|\s*\(.*?\)", "", label) + f" ({where})", "local": where == "local",
                   "pct": 100 * r["level_actual_model"]["accuracy"], **r["latency_ms"]}


def svg():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">',
           """<style>
  .bg { fill: #ffffff; } text { fill: #1f2328; font-size: 13px; } .muted { fill: #656d76; font-size: 12px; }
  .grid { stroke: #d8dee4; } .default { stroke: #656d76; }
  .zone { fill: #dafbe1; stroke: #1a7f37; } text.goal { fill: #1a7f37; }
  .local { fill: #0969da; stroke: #0969da; } .cloud { fill: #ffffff; stroke: #8250df; }
  .local-tail { stroke: #0969da; } .cloud-tail { stroke: #8250df; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #0d1117; } text { fill: #e6edf3; } .muted { fill: #8d96a0; }
    .grid { stroke: #30363d; } .default { stroke: #8d96a0; }
    .zone { fill: #12261e; stroke: #3fb950; } text.goal { fill: #3fb950; }
    .local { fill: #4493f8; stroke: #4493f8; } .cloud { fill: #0d1117; stroke: #ab7df8; }
    .local-tail { stroke: #4493f8; } .cloud-tail { stroke: #ab7df8; }
  }
</style>""",
           f'<rect class="bg" width="{W}" height="{H}"/>']

    # The goal zone, behind everything else.
    gx, gy = x(GOAL_MS), y(GOAL_PCT)
    out.append(f'<rect x="{LEFT}" y="{TOP}" width="{gx - LEFT:.1f}" height="{gy - TOP:.1f}" class="zone" '
               'stroke-dasharray="4 3"/>')
    out.append(f'<text x="{LEFT + 10}" y="{TOP + 20}" class="goal" style="font-weight: 600">Where we want to be</text>')
    out.append(f'<text x="{LEFT + 10}" y="{TOP + 37}" class="goal" style="font-size: 12px">fast and accurate</text>')
    out.append(f'<text x="{gx - 6:.1f}" y="{gy - 8:.1f}" class="goal" text-anchor="end" style="font-size: 11px">'
               f'Doherty threshold, {GOAL_MS} ms</text>')

    for ms in (100, 200, 500, 1000, 2000):
        out.append(f'<line class="grid" x1="{x(ms):.1f}" x2="{x(ms):.1f}" y1="{TOP}" y2="{H - BOTTOM}"/>')
        out.append(f'<text class="muted" x="{x(ms):.1f}" y="{H - BOTTOM + 18}" text-anchor="middle">'
                   f'{ms if ms < 1000 else f"{ms // 1000} s"}{" ms" if ms < 1000 else ""}</text>')
    for pct in range(Y_MIN, Y_MAX + 1, 10):
        out.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(pct):.1f}" y2="{y(pct):.1f}"/>')
        out.append(f'<text class="muted" x="{LEFT - 8}" y="{y(pct) + 4:.1f}" text-anchor="end">{pct}%</text>')

    out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y(DEFAULT_PCT):.1f}" y2="{y(DEFAULT_PCT):.1f}" '
               'class="default" stroke-dasharray="6 4"/>')
    out.append(f'<text class="muted" x="{W - RIGHT - 6}" y="{y(DEFAULT_PCT) + 16:.1f}" text-anchor="end">'
               f'always the model\'s default ({DEFAULT_PCT}%)</text>')

    for p in points():
        kind = "local" if p["local"] else "cloud"
        cx, cy = x(p["p50"]), y(p["pct"])
        out.append(f'<line x1="{cx:.1f}" x2="{x(p["p95"]):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" class="{kind}-tail" '
                   'stroke-width="2" stroke-opacity="0.35"/>')
        out.append(f'<circle class="{kind}" cx="{cx:.1f}" cy="{cy:.1f}" r="6" stroke-width="2.5"/>')
        # Above the point, unless that would sit on the baseline.
        ly = cy - 12 if abs(cy - 16 - y(DEFAULT_PCT)) > 10 else cy + 24
        out.append(f'<text x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle" style="font-weight: 600">{p["label"]}</text>')

    out.append(f'<text x="{LEFT}" y="26" style="font-size: 16px; font-weight: 600">'
               'Right effort level against latency</text>')
    out.append(f'<text x="{(LEFT + W - RIGHT) / 2:.0f}" y="{H - 18}" text-anchor="middle" class="muted">'
               'Latency per prompt: median, line to p95 (log scale)</text>')
    out.append(f'<text transform="translate(18 {(TOP + H - BOTTOM) / 2:.0f}) rotate(-90)" text-anchor="middle" '
               'class="muted">Right level, 136 held-out turns</text>')

    lx = W - RIGHT - 230
    out.append(f'<circle class="local" cx="{lx}" cy="21" r="5" stroke-width="2"/>')
    out.append(f'<text x="{lx + 10}" y="25" class="muted">Local (M4 Pro, llama.cpp)</text>')
    out.append(f'<circle class="cloud" cx="{lx + 170}" cy="21" r="5" stroke-width="2"/>')
    out.append(f'<text x="{lx + 180}" y="25" class="muted">Cloud</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(svg())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
