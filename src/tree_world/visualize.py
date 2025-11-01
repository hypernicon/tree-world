from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import math
import torch

from matplotlib.font_manager import FontProperties
import os

def get_emoji_fontprops():
    candidates = [
        "/System/Library/Fonts/NotoEmoji-Regular.ttf",
        "/Library/Fonts/NotoColorEmoji.ttf",
        # "/opt/homebrew/Caskroom/font-noto-color-emoji/latest/NotoColorEmoji.ttf",
        "/opt/homebrew/Caskroom/font-noto-emoji/latest/NotoEmoji[wght].ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return FontProperties(fname=p)
    # Fallback if not found
    return FontProperties(family="DejaVu Sans")


def visualize_treeworld_run(
    tree_locations,
    tree_names,
    tree_is_poisonous,
    agent_positions,
    agent_healths,
    max_health,
    title="TreeWorld run",
    figsize=(8, 8),
    icon_fontsize=18,
    padding_ratio=0.08,
    save_path=None,
    show=True,
):
    """
    Visualize a 2D TreeWorld run after-the-fact.

    Parameters
    ----------
    tree_locations : torch.Tensor | np.ndarray, shape (N, 2)
        XY coordinates of trees.
    tree_names : List[str], length N
        Names for each tree (used to pick an icon; falls back gracefully).
    tree_is_poisonous : List[bool], length N
        Poison flags for each tree (used for skull/outline).
    agent_positions : torch.Tensor | np.ndarray | List[Tuple[float, float]], shape (T, 2)
        Recorded agent XY positions over time.
    agent_healths : List[float] | np.ndarray | torch.Tensor, length T
        Health values aligned with agent_positions.
    max_health : float
        Maximum health (used to normalize color scale: black=healthy, red=dead).
    title : str
        Figure title.
    figsize : Tuple[int, int]
        Matplotlib figure size.
    icon_fontsize : int
        Font size for tree icons.
    padding_ratio : float
        Fractional padding around the plotted extents.
    save_path : str | None
        If provided, saves the figure to this path.
    show : bool
        If True, calls plt.show().

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    import math

    # --- Convert inputs to numpy ---
    def _to_np(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    tree_locs = _to_np(tree_locations).astype(float)
    agent_xy = _to_np(agent_positions).astype(float)
    health = _to_np(agent_healths).astype(float)

    assert tree_locs.ndim == 2 and tree_locs.shape[1] == 2, "tree_locations must be (N,2)"
    assert agent_xy.ndim == 2 and agent_xy.shape[1] == 2, "agent_positions must be (T,2)"
    assert len(health) == len(agent_xy), "agent_healths must align with agent_positions"
    assert len(tree_names) == len(tree_locs), "tree_names must align with tree_locations"
    assert len(tree_is_poisonous) == len(tree_locs), "tree_is_poisonous must align with tree_locations"
    assert max_health > 0, "max_health must be positive"

    # --- Bounds: "a little beyond the furthest trees" (and include trajectory extents) ---
    all_x = np.concatenate([tree_locs[:, 0], agent_xy[:, 0]])
    all_y = np.concatenate([tree_locs[:, 1], agent_xy[:, 1]])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))

    # Expand bounds by a small fraction
    x_span = x_max - x_min
    y_span = y_max - y_min
    pad_x = (x_span if x_span > 0 else 1.0) * padding_ratio
    pad_y = (y_span if y_span > 0 else 1.0) * padding_ratio
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    # --- Build a black→red colormap; 0 (healthy) → black, 1 (dead) → red ---
    # death_frac = 1 - (health / max_health)
    death_frac = 1.0 - np.clip(health / float(max_health), 0.0, 1.0)
    black_red = LinearSegmentedColormap.from_list("black_red", [(0, 0, 0), (1, 0, 0)])
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Create trajectory segments (T-1 segments between successive points)
    if len(agent_xy) >= 2:
        segs = np.stack([agent_xy[:-1], agent_xy[1:]], axis=1)  # (T-1, 2, 2)
        # Segment colors use the color of the *starting* point of each segment
        seg_colors = black_red(norm(death_frac[:-1]))
    else:
        segs = np.zeros((0, 2, 2))
        seg_colors = np.zeros((0, 4))

    # --- Tree "icons" by name (emoji where possible; fallbacks otherwise) ---
    # Edible emojis
    icon_map = {
        "apple": "🍎",
        "banana": "🍌",
        "cherry": "🍒",
        "cherries": "🍒",
        "orange": "🍊",
        "pear": "🍐",
        "peach": "🍑",
        "mango": "🥭",
        # Reasonable proxies
        "plum": "🍑",
        "nectarine": "🍑",
        "papaya": "🥭",
        "date": "🌴",          # no date-fruit emoji; palm as proxy
        "elderberry": "🫐",    # blueberries as proxy
        "fig": "🪴",           # potted plant as proxy
        # Poisonous defaults
        "nightshade": "☠️",
        "manchineel": "☠️",
        "strychnine fruit": "☠️",
        "desert rose": "🌹",
    }

    def pick_icon(name: str, poisonous: bool) -> str:
        key = name.strip().lower()
        if key in icon_map:
            return icon_map[key]
        # Fallbacks
        if poisonous:
            return "☠️"
        return "🌳"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)

    # Trees
    # Draw subtle base markers for hit area (helps visibility), color by poisonous
    edible_color = (0.1, 0.5, 0.1, 0.6)
    poison_color = (0.6, 0.0, 0.0, 0.6)
    for (x, y), name, is_poison in zip(tree_locs, tree_names, tree_is_poisonous):
        ax.scatter([x], [y],
                   s=120,
                   marker="o",
                   facecolors="none",
                   edgecolors=poison_color if is_poison else edible_color,
                   linewidths=2)
        # Emoji/text icon
        icon = pick_icon(name, is_poison)
        emoji_prop = get_emoji_fontprops()
        ax.text(x, y, icon, fontsize=icon_fontsize, ha="center", va="center",   
                fontproperties=emoji_prop)
        # Optional tiny label under the icon for clarity
        ax.text(x, y - 0.02 * max(x_span, y_span, 1.0), name,
                fontsize=9, ha="center", va="top", color="#444")

    # Agent trajectory as colored segments
    if len(segs) > 0:
        lc = LineCollection(segs, colors=seg_colors, linewidths=2.0)
        ax.add_collection(lc)

    # Start and end markers
    if len(agent_xy) > 0:
        ax.scatter([agent_xy[0, 0]], [agent_xy[0, 1]], s=40, marker="o", color="black", zorder=3, label="start")
        ax.scatter([agent_xy[-1, 0]], [agent_xy[-1, 1]], s=60, marker="X", color="red", zorder=3, label="end")

    # Axes styling
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linewidth=0.5, alpha=0.3)

    # Colorbar (health ↔ color reference)
    # Build a fake mappable that maps death_frac; show ticks in health units
    sm = plt.cm.ScalarMappable(cmap=black_red, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # Convert colorbar ticks (in death_frac units) back to health values for labels
    ticks = np.linspace(0, 1, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{(1-t)*max_health:.0f}" for t in ticks])
    cbar.set_label("Agent Health (black = healthy, red = dead)")

    ax.legend(loc="upper right", frameon=True)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig, ax
