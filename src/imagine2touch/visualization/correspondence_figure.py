"""
The paper's Fig. 2, drawn from the dataset as the paper's script drew it (reskin/tests/test_tsne.py of
the original repository, 27 March 2024).

The contact masks of all touches of the paper's dataset are laid out by t-SNE, and the points of
that layout are grouped into five clusters by k-means.  Example touches carry three thumbnails: the
mask, the tactile image and the colour image of the patch.  A red cross marks the mean mask of every
cluster in the layout, with that mean mask and the cluster's mean tactile image.

As in the script, every image is turned a quarter turn, and the tactile image is the grey value of
the change of the field at each magnetometer: the sizes of its three components weighted like the
red, green and blue of a colour picture.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from openTSNE import TSNE
from sklearn.cluster import KMeans

from imagine2touch.datasets import TactileImage

THUMBNAIL = 11          # side of an example's thumbnails [points]
MEAN = 17               # side of a cluster mean's images [points]


class CorrespondenceFigure:
    """
        CorrespondenceFigure(selection, settings, clusters).embed().draw("results/benchmark/fig_tsne")

    `selection` is a TouchSelection; `settings` holds `seed` and `spacing`.
    """

    WIDTH = 7.16            # a double column [in]
    OBJECTS = ("flower", "hexagon", "star", "parallelogram", "square", "cutter")

    def __init__(self, selection, settings, clusters):
        self.settings, self.clusters = settings, clusters
        dataset, rows = selection.dataset, selection.rows
        # the script laid out the masks after a turn by 90 degrees; they are drawn as the tactile image is laid out
        self.turned = np.rot90(dataset.mask[rows].astype(float), k=1, axes=(1, 2))
        self.masks = TactileImage.aligned(dataset.mask[rows].astype(float))
        self.rgb = TactileImage.aligned(dataset.rgb[rows], colour=True)
        self.tactile = self.visualisation(dataset.signal[rows])
        self.shape = np.asarray(selection.shape)
        # the order in which the script went through the touches: object by object, as its configuration listed them
        rank = [self.OBJECTS.index(name) if name in self.OBJECTS else len(self.OBJECTS) for name in self.shape]
        self.order = np.argsort(rank, kind="stable")

    @staticmethod
    def visualisation(signal):
        """
        (n, 3, 3) tactile images as the paper draws them (TactileImage.grey).  The paper's code took the
        size of every component when it subtracted the reading out of contact (`subtract_ambient` in
        its reskin_calibration/dataset.py).  A value of exactly zero is left empty.
        """
        image = TactileImage.grey(signal)
        return np.where(image == 0, np.nan, image)

    @staticmethod
    def cluster_mean(tactile):
        """The mean tactile image of a cluster; its corners stay empty."""
        mean = np.full(tactile.shape[1:], np.nan)
        cross = ~np.isnan(tactile).all(0)
        mean[cross] = np.nanmean(tactile[:, cross], axis=0)          # a touch without any signal has no image
        return mean

    def embed(self):
        flat = self.turned.reshape(len(self.turned), -1)
        seed = self.settings.seed
        embedding = TSNE(n_components=2, random_state=seed).fit(flat)
        self.points = np.asarray(embedding.transform(flat))         # the script lays the masks out a second time
        self.labels = KMeans(n_clusters=self.clusters, random_state=seed, n_init=10).fit_predict(self.points)
        means = np.array([flat[self.labels == k].mean(0) for k in range(self.clusters)])
        self.means = np.asarray(embedding.transform(means))
        return self

    def examples(self, width=0.0, mean_width=0.0, floor=-np.inf):
        """
        The touches that carry thumbnails, chosen as in the paper's script (`threshold_cluster_points`):
        going through the touches object by object, one is taken if it lies at least `spacing` away
        from every touch already taken and from every cluster mean, along both axes of the layout.
        Touches in the lowest twentieth of the layout, or below `floor`, are left out: their thumbnails
        would fall off the figure.

        The script's spacing was set by hand for its layout.  So that no two columns of thumbnails stand
        over each other in this one, the spacing across is at least `width`, the width of a column in
        units of the layout, and `mean_width` from a cluster mean, whose images are larger.

        Taken alone, that rule gives the free places to the first objects and can leave a cluster
        without an example.  So before it runs, every cluster in turn takes one touch of an object by
        the same rule, a different object each round, until every pair of cluster and object was offered;
        after the first round, every object that has no example yet takes one.
        """
        spacing = np.asarray(self.settings.spacing, float)
        near = np.maximum(spacing, [width, 0])
        near_mean = np.maximum(spacing, [(width + mean_width) / 2, 0])
        floor = max(floor, self.points[:, 1].min() + 0.05 * np.ptp(self.points[:, 1]))
        chosen = []

        def take(touches):
            """Take the first of `touches` that the rule allows."""
            for i in touches:
                point = self.points[i]
                if (point[1] >= floor and (np.abs(self.means - point) >= near_mean).all()
                        and (np.abs(self.points[chosen] - point) >= near).all()):
                    chosen.append(int(i))
                    return

        objects = [name for name in self.OBJECTS if (self.shape == name).any()]
        for turn in range(len(objects)):
            for k in range(self.clusters):
                name = objects[(k + turn) % len(objects)]
                take(self.order[(self.labels[self.order] == k) & (self.shape[self.order] == name)])
            if turn == 0:                           # every cluster has one: now every object that has none
                for name in objects:
                    if name not in self.shape[chosen]:
                        take(self.order[self.shape[self.order] == name])
        for i in self.order:
            take([i])
        return chosen

    @staticmethod
    def _stack(ax, xy, images, side, gap=1.5):
        """Draw images in a column below the point xy; each is (array, colour map)."""
        for rank, (image, cmap) in enumerate(images):
            pixels = image.shape[0]
            # as in the paper, every image is coloured over its own range, from its smallest to its largest value
            box = OffsetImage(image, zoom=side / pixels, cmap=cmap, interpolation="nearest")
            ax.add_artist(AnnotationBbox(box, xy, xybox=(0, -(rank + 0.5) * (side + gap + 2) - 8), xycoords="data",
                                         boxcoords="offset points", pad=0.25, bboxprops=dict(lw=0.7),
                                         arrowprops=dict(arrowstyle="->", lw=1.1, color="black") if rank == 0 else None))

    def draw(self, path):
        plasma = plt.get_cmap("plasma").copy()
        plasma.set_bad("white")
        enlarge = lambda cross: np.kron(cross, np.ones((16, 16)))
        fig, ax = plt.subplots(figsize=(self.WIDTH, 4.0))
        ax.scatter(*self.points.T, c=self.labels, cmap="tab10", s=9, lw=0)
        ax.margins(0.06, 0.14)
        ax.axis("off")
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        unit = (ax.transData.transform((1, 1)) - ax.transData.transform((0, 0))) * 72 / fig.dpi         # points per unit of the layout, across and up
        frame = 7                                                                                      # a column's frame and a gap [points]
        height = 3 * (THUMBNAIL + frame) + 10                                                          # a column of three thumbnails under its arrow
        floor = ax.get_ylim()[0] + height / unit[1]
        for i in self.examples((THUMBNAIL + frame) / unit[0], (MEAN + frame) / unit[0], floor):
            self._stack(ax, self.points[i], [(self.masks[i], plasma), (enlarge(self.tactile[i]), plasma), (self.rgb[i], None)],
                        THUMBNAIL)
        for k in range(self.clusters):
            inside = self.labels == k
            ax.scatter(*self.means[k], marker="x", color="red", s=130, lw=0.8, zorder=5)
            self._stack(ax, self.means[k], [(self.masks[inside].mean(0), plasma),
                                            (enlarge(self.cluster_mean(self.tactile[inside])), plasma)], MEAN)
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", dpi=300)
        plt.close(fig)
        print(f"saved {path}.{{pdf,png}}")
