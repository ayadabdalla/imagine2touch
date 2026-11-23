"""Pictures of the shape classification dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imagine2touch.datasets import TactileImage
from .style import Style


class StampVisualizer:
    """
        StampVisualizer(dataset, photos="assets/stamps").overview("assets/stamps_overview.png")

    One column per stamp: its photograph (`photos` holds <stamp>.jpg) and, for a few of the positions
    it was pressed at, the mean touch signal there, as the cross of the five magnetometers (the field
    along the direction of the press).  A small drawing of the sensing surface shows each position.
    """

    PAD = 17.0      # side of the sensing surface [mm]

    def __init__(self, dataset, photos=None):
        Style.apply()
        self.dataset, self.photos = dataset, Path(photos) if photos else None
        self.along = TactileImage.along_touch(dataset.signal)
        for corner in TactileImage.CORNERS:
            self.along[(slice(None), *corner)] = np.nan
        self.limit = float(np.percentile(np.abs(dataset.signal[:, 2::3]), 98))

    def location(self, ax, position):
        """
        The sensing surface as a square, with the place of a press [mm] marked on it: the press grid's
        first axis to the right and its second upwards, which is the layout of TactileImage: the
        magnetometer nearest the stamp is the one that responds.
        """
        ax.add_patch(plt.Rectangle((0, 0), self.PAD, self.PAD, fill=False, ec=Style.INK_2, lw=0.8))
        ax.plot([self.PAD / 2] * 2, [0, self.PAD], color=Style.HAIRLINE, lw=0.6)
        ax.plot([0, self.PAD], [self.PAD / 2] * 2, color=Style.HAIRLINE, lw=0.6)
        ax.scatter(position[0], position[1], s=42, color=Style.INK, zorder=3)
        ax.set(xlim=(-1.5, self.PAD + 1.5), ylim=(-1.5, self.PAD + 1.5), aspect="equal")
        ax.axis("off")

    def overview(self, path=None, shown=((8.5, 8.5), (4.2, 8.5))):
        """`shown`: the press positions to draw [mm], of the eight in the dataset."""
        d = self.dataset
        position, positions = d.position_index, 1000 * np.unique(d.position.round(5), axis=0)
        chosen = [int(np.abs(positions - p).sum(1).argmin()) for p in shown]
        W, side, gap, left = 9.5, 1.05, 0.34, 2.25
        fig = plt.figure(figsize=(W, 1.25 + (side + 0.25) * (1 + len(chosen))))
        H = fig.get_figheight()
        fig.text(0.3 / W, 1 - 0.3 / H, "Five stamps pressed into the sensor", fontsize=12, weight="bold")
        fig.text(0.3 / W, 1 - 0.44 / H, f"{len(d):,} presses of stamps smaller than the sensor, {int((d.stamp == 0).sum())} of each, at {len(positions)} "
                 f"positions on it. Shown for {len(chosen)} positions:\nthe mean touch signal of every stamp, as the change of the "
                 "field along the press at the five magnetometers.", color=Style.INK_2, va="top", linespacing=1.3)
        top = 1.05
        for c, stamp in enumerate(d.STAMPS):
            x = (left + c * (side + gap)) / W
            fig.text(x, 1 - (top + 0.02) / H, stamp, fontsize=9, va="bottom")
            photo = None if self.photos is None else self.photos / f"{stamp}.jpg"
            if photo is not None and photo.exists():
                ax = fig.add_axes([x, 1 - (top + 0.08 + side) / H, side / W, side / H])
                ax.imshow(plt.imread(photo))
                ax.axis("off")
        for r, k in enumerate(chosen):
            bottom = 1 - (top + 0.08 + (side + 0.25) * (r + 1) + side) / H
            self.location(fig.add_axes([0.3 / W, bottom, side / W, side / H]), positions[k])
            if r == 0:
                fig.text(0.3 / W, bottom + (side + 0.08) / H, "centre of the stamp\non the sensor", fontsize=8.5,
                         color=Style.INK_2, va="bottom", linespacing=1.3)
            for c in range(len(d.STAMPS)):
                ax = fig.add_axes([(left + c * (side + gap)) / W, bottom, side / W, side / H])
                image = ax.imshow(self.along[(d.stamp == c) & (position == k)].mean(0), cmap=Style.SIGNAL,
                                  vmin=-self.limit, vmax=self.limit)
                Style.frame(ax)
        low = 1 - (top + 0.08 + (side + 0.25) * len(chosen) + side) / H
        bar = fig.colorbar(image, cax=fig.add_axes([(left + 5 * (side + gap) - gap + 0.12) / W, low, 0.07 / W,
                                                    ((side + 0.25) * (len(chosen) - 1) + side) / H]))
        bar.set_ticks([-self.limit, 0, self.limit], labels=[f"−{self.limit:.0f}", "0", f"+{self.limit:.0f}"])
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, labelsize=7.5)
        if path is None:
            return fig
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"saved {path}")
