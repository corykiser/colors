"""Fixed probe contexts for sample grids (handoff G3): neutrals, vivid, near-grays, unusual pairs, n+m in {3,4,5}."""
from __future__ import annotations

PROBES: list[dict] = [
    {"name": "sage+brown → 2", "colors": ["#9caf88", "#6b4a2b"], "m": 2},
    {"name": "sage+brown → 3", "colors": ["#9caf88", "#6b4a2b"], "m": 3},
    {"name": "navy → 3", "colors": ["#1f2a44"], "m": 3},
    {"name": "navy → 4", "colors": ["#1f2a44"], "m": 4},
    {"name": "black+white → 3", "colors": ["#000000", "#ffffff"], "m": 3},
    {"name": "near-grays → 2", "colors": ["#8a8a8a", "#b3b3b0"], "m": 2},
    {"name": "warm gray → 4", "colors": ["#a89f91"], "m": 4},
    {"name": "magenta+lime → 3", "colors": ["#e91e8c", "#9be51b"], "m": 3},
    {"name": "teal+coral → 2", "colors": ["#1b9aaa", "#ef767a"], "m": 2},
    {"name": "teal+coral → 3", "colors": ["#1b9aaa", "#ef767a"], "m": 3},
    {"name": "mustard → 2", "colors": ["#d4a017"], "m": 2},
    {"name": "burgundy+cream+olive → 2", "colors": ["#6d1a36", "#f3e9d2", "#6b7a3a"], "m": 2},
    {"name": "pastels → 2", "colors": ["#f9c1ce", "#c1e1f9", "#e0f9c1"], "m": 2},
    {"name": "pure red → 4", "colors": ["#ff0000"], "m": 4},
    {"name": "pure blue+yellow → 3", "colors": ["#0000ff", "#ffff00"], "m": 3},
    {"name": "brown+purple (unusual) → 3", "colors": ["#7a4b1e", "#7d3c98"], "m": 3},
    {"name": "orange+cyan (unusual) → 2", "colors": ["#ff7f0e", "#17becf"], "m": 2},
    {"name": "dark forest → 4", "colors": ["#0b3d2e"], "m": 4},
    {"name": "four earth tones → 1", "colors": ["#8b5a2b", "#c9a66b", "#5c6b3a", "#3e2a1d"], "m": 1},
    {"name": "empty → 5", "colors": [], "m": 5},
    {"name": "empty → 3", "colors": [], "m": 3},
    {"name": "peach+sky → 2", "colors": ["#ffcba4", "#87ceeb"], "m": 2},
]
