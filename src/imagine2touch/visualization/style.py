"""The colours and type shared by the pictures of the data."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


class Style:
    """
    One colour scale per kind of quantity, so that a colour means the same in every picture:
    depth is one hue from dark (near) to light (far); the touch signal, which has a sign, runs from
    blue through grey at zero to red; a touch drawn on an object is one blue.
    """

    SURFACE, INK, INK_2, MUTED, HAIRLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    TOUCH = "#2a78d6"                                           # a touch drawn on an object
    SCAN = "#c3c2b7"            # the points of a scan
    MASK = "#104281"            # the pixels of a contact mask
    MISSING = "#e9e8e2"         # a pixel without a reading

    DEPTH = ListedColormap(["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7",
                            "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"], "depth")
    SIGNAL = LinearSegmentedColormap.from_list("signal", ["#184f95", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#a32a2a"])
    MASKS = ListedColormap([SURFACE, MASK], "mask")
    DEPTH = DEPTH.with_extremes(bad=MISSING, over=SURFACE)      # no reading; nothing within reach
    SIGNAL = SIGNAL.with_extremes(bad=SURFACE)                  # the corners of a tactile image

    RC = {"font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8, "axes.labelcolor": INK_2,
          "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "text.color": INK, "axes.edgecolor": "#c3c2b7",
          "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6, "figure.facecolor": SURFACE,
          "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE}

    @classmethod
    def apply(cls):
        plt.rcParams.update(cls.RC)

    @classmethod
    def frame(cls, ax, color=None, width=0.6):
        """An image panel: no ticks, a thin frame."""
        ax.set(xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set(visible=True, color=color or "#c3c2b7", lw=width)
