"""Pictures of the visuo-tactile dataset."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pathlib import Path

from .style import Style
from .touch_view import TouchView

PANELS = ("where", "rgb", "camera", "scan", "mask", "field")
TITLES = {"where": "Where", "rgb": "Colour image", "camera": "Depth, wrist camera", "mask": "Contact mask, camera",
          "scan": "Depth, object scan", "field": "Touch signal"}
SIDE = 1.3              # side of an image panel [inches]


class DatasetVisualizer:
    """
        visualizer = DatasetVisualizer(dataset)
        visualizer.overview("assets/dataset_overview.png")      # the objects and where they were touched
        visualizer.animation("assets/dataset_touches.gif")      # a walk through touches of every object
        visualizer.touch(123)                                    # one touch, as a figure

    `sessions` are the sessions to show, a list of names or the ending of the names.  The 3D views
    and the depth image of the scan exist only for sessions whose scan can be placed (`placed`).
    Readings above `max_signal` are glitches of the sensor: they are counted and drawn on the object,
    but never chosen as an example and kept out of the colour scale.
    """

    def __init__(self, dataset, sessions, seed=0, max_signal=500, photos=None):
        Style.apply()
        self.photos = Path(photos) if photos else None          # a folder with a photograph <shape>.jpg of every object
        self.dataset, self.sessions = dataset, dataset.sessions(sessions)
        self.rows = np.flatnonzero(dataset.in_sessions(sessions))
        self.calm = np.abs(dataset.signal[self.rows]).max(1) <= max_signal
        self.placed = [k for k in self.sessions if dataset.has_scan(k)]
        self.rng = np.random.default_rng(seed)
        self.signal_limit = float(np.percentile(np.abs(dataset.signal[self.rows[self.calm]]), 98))

    # ------------------------------------------------------------------ choosing touches worth looking at
    def clear_touches(self, per_shape=1, sessions=None):
        """
        Touches that show the data well: not known to have missed, on the edge of an object (the mask
        is neither empty nor full), with a camera image almost without holes.  `per_shape` of every session.
        """
        d, chosen = self.dataset, []
        filled = d.mask[self.rows].reshape(len(self.rows), -1).mean(1)
        seen = (d.depth[self.rows] > 0).reshape(len(self.rows), -1).mean(1)
        good = self.calm & np.isin(d.outcome[self.rows], ("sensor", "unknown")) & (filled > 0.25) & (filled < 0.7) & (seen > 0.97)
        # where the scan is placed, also ask that the sensor stopped on it
        on_scan = np.array([not d.has_scan(int(d.object[i])) or d.distance_to_scan(i, int(d.object[i])) < 1.5 for i in self.rows])
        good &= on_scan
        for k in self.sessions if sessions is None else sessions:
            of_object = d.object[self.rows] == k
            candidates = self.rows[good & of_object]
            if len(candidates) == 0:                    # no touch of this object is that clear: take any
                candidates = self.rows[of_object]
            chosen += list(self.rng.choice(candidates, min(per_shape, len(candidates)), replace=False))
        return chosen

    def photo(self, ax, session):
        """A photograph of the object of a session, if there is one."""
        path = None if self.photos is None else self.photos / f"{self.dataset.shape_of_session(session)}.jpg"
        ax.set(xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_visible(False)
        if path is not None and path.exists():
            ax.imshow(plt.imread(path))
        else:
            ax.set_facecolor("#ececea")
            ax.text(0.5, 0.5, "no photograph", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=Style.INK_2)

    def strongest(self, touches):
        """Of some touches, the one with the largest touch signal."""
        return max(touches, key=lambda i: float(np.linalg.norm(self.dataset.signal[i])))

    # ------------------------------------------------------------------ the parts of a picture
    STICK = 5.0     # length of the line that shows where a touch came from [mm]
    ON_OBJECT = 2.0 # a touch is drawn if the scan has points within this distance of where the sensor stopped [mm],
    COVERED = 0.3   # at least this share of what a closed surface would have there

    def on_object(self, session, shown=None):
        """
        The touches of a session whose sensor stopped on the scan: the scan has points around the
        place, not only near it.  Returns them, or a random `shown` of them.  Some approaches stopped
        in free air or where the scan has a hole, and were recorded all the same.
        """
        d, scan = self.dataset, self.dataset.scan(session)
        rows = self.rows[d.object[self.rows] == session]
        closed = np.pi * self.ON_OBJECT ** 2 / scan.point_spacing ** 2        # points of a closed surface within ON_OBJECT
        rows = rows[[len(scan.points_near(d.pose[int(i)].t, self.ON_OBJECT)) >= self.COVERED * closed for i in rows]]
        if shown is not None and len(rows) > shown:
            rows = np.sort(np.random.default_rng(int(session)).choice(rows, shown, replace=False))
        return rows

    def object_panel(self, ax, session, highlight=None, shown=None):
        """
        The scan of one object in 3D.  Every touch is a dot where the sensor stopped and a short line
        back along the direction it came from.  With `highlight`
        (a TouchView), only that touch is drawn, with the outline of the sensing surface.
        """
        d = self.dataset
        scan = d.scan(session).points
        centre = np.median(scan, axis=0)
        sparse = d.scan(session).point_spacing > 0.6                  # some scans were thinned when they were recorded
        cloud = (scan if sparse else scan[::2]) - centre
        shade = 0.3 + 0.45 * (cloud[:, 2] - cloud[:, 2].min()) / max(np.ptp(cloud[:, 2]), 1e-9)      # higher is lighter
        ax.computed_zorder = False                              # draw in the order given: the touches over the cloud
        ax.scatter(*cloud.T, s=7 if sparse else 1.2, lw=0, c=shade, cmap="Greys_r", vmin=0, vmax=1, depthshade=False,
                   rasterized=True, zorder=1)
        rows = self.on_object(session, shown) if highlight is None else [highlight.index]
        at = np.array([d.pose[i].t for i in rows]) - centre
        back = at - self.STICK * np.array([d.pose[i].normal for i in rows])
        ax.add_collection3d(Line3DCollection(np.stack([at, back], 1), colors=Style.TOUCH,
                                             lw=1.8 if highlight is not None else 0.7,
                                             alpha=1.0 if highlight is not None else 0.75, zorder=3))
        ax.scatter(*at.T, s=16 if highlight is not None else 5, lw=0, color=Style.TOUCH, depthshade=False, zorder=4)
        if highlight is not None:
            corners = highlight.footprint - centre
            ax.plot(*corners.T, color=Style.TOUCH, lw=1.4, zorder=5)
        half = 0.52 * np.ptp(scan[:, :2], axis=0).max() + 2 * self.STICK
        if np.ptp(scan[:, :2], axis=0).max() > 100:              # a long object lies along the diagonal of the view: fill it
            half *= 0.52
        ax.set(xlim=(-half, half), ylim=(-half, half), zlim=(-half * 0.6, half * 0.7))
        ax.set_box_aspect((1, 1, 0.65), zoom=1.35)
        ax.view_init(elev=32, azim=-58)
        ax.set_axis_off()
        return half

    def touch_panels(self, axes, view):
        """Fill the panels of one touch.  Panels a session has no data for are left out of `axes` by the caller."""
        depth = dict(cmap=Style.DEPTH, vmin=0, vmax=view.reach, interpolation="nearest")
        if "where" in axes:
            self.object_panel(axes["where"], view.session, highlight=view)
        axes["rgb"].imshow(view.rgb)
        axes["camera"].imshow(np.clip(view.camera_gap, 0, None), **depth)
        if "scan" in axes:
            scan = np.where(view.scan_gap >= view.reach, 2 * view.reach, np.clip(view.scan_gap, 0, None))    # nothing in reach: drawn as far
            axes["scan"].imshow(scan, **depth)
        axes["mask"].imshow(view.mask, cmap=Style.MASKS, vmin=0, vmax=1, interpolation="nearest")
        axes["field"].imshow(view.field[2], cmap=Style.SIGNAL, vmin=-self.signal_limit, vmax=self.signal_limit)
        for name, ax in axes.items():
            if name != "where":
                Style.frame(ax)

    @staticmethod
    def _row(fig, top, side=SIDE, left=0.3, panels=PANELS):
        """Axes in a row whose top lies `top` inches below the figure's top: squares, the touch signal a little apart."""
        W, H = fig.get_size_inches()
        axes, x = {}, left
        for name in panels:
            small = name == "field"
            x += 0.28 if small else 0
            size = 0.8 * side if small else side
            box = [x / W, 1 - (top + (side + size) / 2) / H, size / W, size / H]
            axes[name] = fig.add_axes(box, projection="3d") if name == "where" else fig.add_axes(box)
            x += size + (0.07 if small else 0.12)
        return axes

    def _colour_bars(self, fig, axes):
        """A scale under the depth panels and one under the touch signal; returns the height they end at [figure fraction]."""
        H = fig.get_figheight()
        first, last = axes["camera"].get_position(), axes.get("scan", axes["camera"]).get_position()
        low = last.y0 - 0.2 / H
        bar = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, self.dataset.grid.reach), Style.DEPTH),
                           cax=fig.add_axes([first.x0, low, last.x1 - first.x0, 0.07 / H]), orientation="horizontal")
        bar.set_ticks([0, 3.5, 7], labels=["nearest", "3.5 mm", "7 mm"] if "scan" not in axes else ["touching", "3.5 mm in front", "7 mm"])
        first = last = axes["field"].get_position()
        bar_2 = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(-self.signal_limit, self.signal_limit), Style.SIGNAL),
                             cax=fig.add_axes([first.x0, low, last.x1 - first.x0, 0.07 / H]), orientation="horizontal")
        bar_2.set_ticks([-self.signal_limit, 0, self.signal_limit],
                        labels=[f"−{self.signal_limit:.0f}", "no change", f"+{self.signal_limit:.0f}"])
        for b in (bar, bar_2):
            b.outline.set_visible(False)
            b.ax.tick_params(length=2, labelsize=7.5)
        return low - 0.25 / H

    def _titles(self, axes):
        for name in PANELS:
            if TITLES[name]:
                axes[name].set_title(TITLES[name], fontsize=9, pad=4, loc="left")

    def _caption(self, view):
        text = f"{view.shape}, touch {view.index}"
        return text + (f"; the robot estimated {view.force:.1f} N" if np.isfinite(view.force) else "")

    # ------------------------------------------------------------------ the pictures
    def overview(self, path=None, examples=0, shown=60):
        """
        Where the touches are, and `examples` touches in detail, for the sessions whose scan can be
        placed under the sensor.
        """
        d, placed = self.dataset, self.placed
        if not placed:
            raise ValueError("none of these sessions has a scan that can be placed")
        rows = self.rows[np.isin(d.object[self.rows], placed)]
        main = self.strongest(self.clear_touches(3, placed))
        others = [i for i in self.clear_touches(1, placed) if d.object[i] != d.object[main]][:examples]
        more = 0.6 + (SIDE + 0.14) * len(others) if others else 0.0
        fig = plt.figure(figsize=(9.5, 2.6 + (1.05 if self.photos is not None else 0) + more))
        W, H = fig.get_size_inches()
        y, left = (lambda inches: 1 - inches / H), 0.3 / W           # a height measured from the top, in inches

        fig.text(left, y(0.3), "Imagine2touch dataset", fontsize=12, weight="bold")
        fig.text(left, y(0.44), f"{len(rows):,} touches of {len(placed)} objects; up to {shown} of each are drawn on the object's scan. A dot is where the "
                 "sensor stopped; the line points back the way it came.\nAbove each scan is a photograph of the object. The number of "
                 "touches is in brackets; the primitives are approx. 3 to 4 cm wide.",
                 color=Style.INK_2, va="top", linespacing=1.3)
        side = min(2.3, (W - 0.6 - 0.1 * (len(placed) - 1)) / len(placed))
        photo = 0.72 * side if self.photos is not None else 0.0
        for column, k in enumerate(placed):
            x = (0.3 + column * (side + 0.1)) / W
            n = int((d.object[self.rows] == k).sum())
            fig.text(x, y(1.0), f"{d.shape_of_session(k)} ({n})", fontsize=9)
            if photo:
                self.photo(fig.add_axes([x, y(1.1 + photo), photo / W, photo / H]), k)
            self.object_panel(fig.add_axes([x, y(1.0 + photo + side), side / W, side / H], projection="3d"), k, shown=shown)
        side = side + photo

        top = 1.45 + side
        if others:
            fig.text(left, y(top), "More touches", fontsize=12, weight="bold")
        for r, i in enumerate(others):
            row, view = self._row(fig, top + 0.2 + (SIDE + 0.14) * r), TouchView(d, i)
            self.touch_panels(row, view)
            box = row["where"].get_position()
            fig.text(box.x0, box.y1 - 0.02 / H, view.shape, fontsize=9, va="top")
        return self._save(fig, path)

    def sessions_overview(self, sessions, path=None, per_object=2, title="The sessions of the paper"):
        """
        Example touches of the sessions `sessions` (names), `per_object` of each, with what every session
        of the dataset has: the colour image, the camera's depth, the contact mask and the touch signal.
        """
        d, panels = self.dataset, ("rgb", "camera", "mask", "field")
        side, step = 0.95, 1.1
        fig = plt.figure(figsize=(10.4, 1.25 + step * len(sessions)))
        H = fig.get_figheight()
        fig.text(0.3 / 10.4, 1 - 0.3 / H, title, fontsize=12, weight="bold")
        total = int(d.in_sessions(sessions).sum())
        fig.text(0.3 / 10.4, 1 - 0.44 / H, f"{total:,} approaches of {len(sessions)} objects; {per_object} of each are shown, with the colour "
                 "image, the camera's depth\n(counted from its nearest pixel), the contact mask and the touch signal.",
                 color=Style.INK_2, va="top", linespacing=1.3)
        for r, name in enumerate(sessions):
            k = d.objects.index(name)
            rows = np.flatnonzero(d.object == k)
            filled = d.mask[rows].reshape(len(rows), -1).mean(1)
            seen = (d.depth[rows] > 0).reshape(len(rows), -1).mean(1)
            calm = np.abs(d.signal[rows]).max(1) < 10 * self.signal_limit            # not a glitch of the sensor
            good = rows[(filled > 0.25) & (filled < 0.7) & (seen > 0.97) & calm]
            chosen = self.rng.choice(good if len(good) >= per_object else rows, per_object, replace=False)
            top = 1.05 + step * r
            fig.text(0.3 / 10.4, 1 - (top + side / 2) / H, f"{d.shape_of_session(k)}\n{len(rows)}", fontsize=9, va="center",
                     linespacing=1.3)
            if self.photos is not None:
                self.photo(fig.add_axes([1.35 / 10.4, 1 - (top + side) / H, side / 10.4, side / H]), k)
            for column, i in enumerate(chosen):
                axes = self._row(fig, top, side, left=2.6 + column * 4.0, panels=panels)
                self.touch_panels(axes, TouchView(d, i))
                if r == 0:
                    for key, text in (("rgb", "Colour"), ("camera", "Depth"), ("mask", "Mask"), ("field", "Touch")):
                        axes[key].set_title(text, fontsize=8.5, pad=3, loc="left")
        return self._save(fig, path)

    def touch(self, i, path=None):
        """One touch as a figure of its own."""
        fig = plt.figure(figsize=(10.4, SIDE + 1.15))
        view = TouchView(self.dataset, i)
        axes = self._row(fig, 0.6)
        self.touch_panels(axes, view)
        self._titles(axes)
        self._colour_bars(fig, axes)
        fig.text(0.3 / 10.4, 1 - 0.22 / fig.get_figheight(), self._caption(view) + ".", color=Style.INK_2)
        return self._save(fig, path)

    def animation(self, path, per_shape=5, seconds=1.4, dpi=100):
        """A walk through `per_shape` touches of every session, as an animated GIF, with what every session has."""
        touches = self.clear_touches(per_shape)
        placed = all(self.dataset.has_scan(int(self.dataset.object[i])) for i in touches)
        panels = PANELS if placed else ("rgb", "camera", "mask", "field")
        width = 9.5 if placed else 6.6
        fig = plt.figure(figsize=(width, SIDE + 1.15), dpi=dpi)
        axes = self._row(fig, 0.6, panels=panels)
        caption = fig.text(0.3 / width, 1 - 0.22 / fig.get_figheight(), "", color=Style.INK_2)
        self._colour_bars(fig, axes)

        def frame(n):
            for ax in axes.values():
                ax.clear()
            view = TouchView(self.dataset, touches[n])
            self.touch_panels(axes, view)
            for key in panels:
                axes[key].set_title(TITLES[key], fontsize=9, pad=4, loc="left")
            caption.set_text(f"{view.shape}, touch {view.index}")

        FuncAnimation(fig, frame, frames=len(touches)).save(path, writer=PillowWriter(fps=1 / seconds))
        plt.close(fig)
        print(f"saved {path}")

    @staticmethod
    def _save(fig, path):
        if path is None:
            return fig
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"saved {path}")
