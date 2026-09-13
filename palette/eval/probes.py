"""Fixed probe contexts (handoff G3), balanced by context size so that 3- and 4-color completions get equal weight.

Layout: 6 contexts each with n = 1, 2, 3 fixed colors, and 2 with n = 4, at target counts giving n+m ∈ {3,4,5}.
Each n-group mixes neutrals, vivid colors, near-grays and unusual pairs. Two unconditional contexts at the end.
"""
from __future__ import annotations

PROBES: list[dict] = [
    # n = 1
    {"name": "navy → 2", "colors": ["#1f2a44"], "m": 2},
    {"name": "navy → 4", "colors": ["#1f2a44"], "m": 4},
    {"name": "mustard → 3", "colors": ["#d4a017"], "m": 3},
    {"name": "warm gray → 2", "colors": ["#a89f91"], "m": 2},
    {"name": "pure red → 3", "colors": ["#ff0000"], "m": 3},
    {"name": "dark forest → 4", "colors": ["#0b3d2e"], "m": 4},
    # n = 2
    {"name": "sage+brown → 1", "colors": ["#9caf88", "#6b4a2b"], "m": 1},
    {"name": "sage+brown → 2", "colors": ["#9caf88", "#6b4a2b"], "m": 2},
    {"name": "sage+brown → 3", "colors": ["#9caf88", "#6b4a2b"], "m": 3},
    {"name": "black+white → 2", "colors": ["#000000", "#ffffff"], "m": 2},
    {"name": "near-grays → 2", "colors": ["#8a8a8a", "#b3b3b0"], "m": 2},
    {"name": "magenta+lime → 3", "colors": ["#e91e8c", "#9be51b"], "m": 3},
    {"name": "teal+coral → 1", "colors": ["#1b9aaa", "#ef767a"], "m": 1},
    {"name": "brown+purple (unusual) → 2", "colors": ["#7a4b1e", "#7d3c98"], "m": 2},
    {"name": "orange+cyan (unusual) → 3", "colors": ["#ff7f0e", "#17becf"], "m": 3},
    {"name": "peach+sky → 2", "colors": ["#ffcba4", "#87ceeb"], "m": 2},
    # n = 3
    {"name": "burgundy+cream+olive → 1", "colors": ["#6d1a36", "#f3e9d2", "#6b7a3a"], "m": 1},
    {"name": "burgundy+cream+olive → 2", "colors": ["#6d1a36", "#f3e9d2", "#6b7a3a"], "m": 2},
    {"name": "pastels → 1", "colors": ["#f9c1ce", "#c1e1f9", "#e0f9c1"], "m": 1},
    {"name": "pastels → 2", "colors": ["#f9c1ce", "#c1e1f9", "#e0f9c1"], "m": 2},
    {"name": "charcoal+rust+ivory → 1", "colors": ["#36393b", "#b7410e", "#fffff0"], "m": 1},
    {"name": "three grays → 2", "colors": ["#3a3a3a", "#8a8a8a", "#d0d0d0"], "m": 2},
    {"name": "blue+yellow+red (primaries) → 2", "colors": ["#0000ff", "#ffff00", "#ff0000"], "m": 2},
    {"name": "ocean trio → 1", "colors": ["#0a2342", "#2ca6a4", "#84bcda"], "m": 1},
    # n = 4
    {"name": "four earth tones → 1", "colors": ["#8b5a2b", "#c9a66b", "#5c6b3a", "#3e2a1d"], "m": 1},
    {"name": "four vivid → 1", "colors": ["#e63946", "#f1fa8c", "#06d6a0", "#118ab2"], "m": 1},
    # unconditional
    {"name": "empty → 3", "colors": [], "m": 3},
    {"name": "empty → 5", "colors": [], "m": 5},
]
