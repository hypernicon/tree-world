import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from matplotlib.patches import Circle
from matplotlib import transforms


class LiveVizMPL:
    def __init__(self, world, trail_len=2000, show_buttons=False, show_legend=False):
        self.show_legend = show_legend
        self.show_buttons = show_buttons

        plt.ion()
        self.world = world
        self.trail_len = trail_len

        self.fig, (self.ax_xy, self.ax_health) = plt.subplots(1, 2, figsize=(10, 5))
        try:
            self.fig.canvas.manager.set_window_title("TreeWorld Live")
        except Exception:
            pass

        # --- Buttons (optional) ---
        self.reset_btn = None
        if show_buttons:
            # x, y, w, h in figure fraction coords
            self.ax_reset = self.fig.add_axes([0.86, 0.02, 0.12, 0.06])
            self.reset_btn = Button(self.ax_reset, "Reset")
            self.reset_btn.on_clicked(self._on_reset_clicked)

        # Keyboard shortcut: press "r" to reset
        self._cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # --- Static trees layer ---
        self.s_poison = None
        self.s_edible = None
        self._init_trees()

        # --- Interpreted memory overlays (you may already have these) ---
        dark_gray = "#444444"
        (self.mem_edible,)  = self.ax_xy.plot([], [], linestyle="None", marker="o", ms=4,
                                            alpha=0.7, color=dark_gray, label="mem: edible")
        (self.mem_poison,)  = self.ax_xy.plot([], [], linestyle="None", marker="x", ms=5,
                                            alpha=0.7, color=dark_gray, label="mem: poison")
        (self.mem_neutral,) = self.ax_xy.plot([], [], linestyle="None", marker=".", ms=6,
                                            alpha=0.4, color=dark_gray, label="mem: neutral")

        # --- Beliefs ---
        (self.agent_belief_pt,)  = self.ax_xy.plot([], [], linestyle="None", marker="s", ms=9,
                                                mfc="none", mec="#116666", mew=1.2, alpha=0.95,
                                                label="agent belief")
        (self.target_belief_pt,) = self.ax_xy.plot([], [], linestyle="None", marker="o", ms=9,
                                                mfc="none", mec="#116666", mew=1.4, alpha=0.95,
                                                label="target belief")

        # Optional SD ring for the target belief
        self.target_belief_ring = Circle((0, 0), radius=1.0, fill=False, linewidth=1.2,
                                        alpha=0.9, edgecolor="#ffd166", linestyle="-")
        self.target_belief_ring.set_visible(False)
        self.ax_xy.add_patch(self.target_belief_ring)

        # How often to pull model.interpret() automatically (frames)
        self._interpret_every = 25

        # --- Agent + trail ---
        self.trail_x, self.trail_y = [], []
        # (self.agent_dot,) = self.ax_xy.plot([], [], marker="*", ms=10, linestyle="None", label="agent")
        (self.trail_line,) = self.ax_xy.plot([], [], linewidth=1.0, alpha=0.85, label="trail")

        # Agent triangle (base pointing +Y in local coords)
        span = self.world.config.tree_spacing * self.world.config.dim * 3
        self.agent_size = 0.03 * span  # tweak to taste

        # A small isosceles triangle centered at origin, pointing up (+Y)
        tri = np.array([
            [0.0,  1.0],     # tip
            [-0.7, -0.9],    # left base
            [0.7, -0.9],     # right base
        ]) * self.agent_size

        self._agent_tri_local = tri  # keep for rebuilds
        self.agent_patch = Polygon(tri, closed=True, facecolor="#ffd166", edgecolor="none", alpha=1.0)
        self.ax_xy.add_patch(self.agent_patch)

        # --- Agent sensory range circles ---
        outer_r = self.world.config.max_sense_distance
        inner_r = self.world.config.can_see_fruit_distance

        self.agent_outer_circle = Circle(
            (0, 0), radius=outer_r, fill=True, color="#aaaa00", alpha=0.15, linewidth=0
        )  # soft transparent yellow

        self.agent_inner_circle = Circle(
            (0, 0), radius=inner_r, fill=True, color="#aa9900", alpha=0.35, linewidth=0
        )  # darker yellow-orange

        self.ax_xy.add_patch(self.agent_outer_circle)
        self.ax_xy.add_patch(self.agent_inner_circle)

        # --- Health plot ---
        self.health_x, self.health_y = [], []
        (self.health_line,) = self.ax_health.plot([], [], linewidth=1.5)
        self.ax_health.set_ylim(0, self.world.config.max_health)
        self.ax_health.set_title("Health")
        self.ax_health.set_xlabel("Step")
        self.ax_health.set_ylabel("HP")

        # --- World axes / limits ---
        span = self.world.config.tree_spacing * self.world.config.dim * 3
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.set_xlim(-span, span)
        self.ax_xy.set_ylim(-span, span)
        self.ax_xy.set_title("TreeWorld")
        if self.show_legend:
            self.ax_xy.legend(loc="upper right")

        self.fig.tight_layout()
        self._needs_draw = True

    # ---------- public API ----------

    def update(self, step_idx: int):
        """Call every sim step."""
        p = self.world.agent.location
        px, py = (float(p[0]), float(p[1]))
        # self.agent_dot.set_data([px], [py])
        p = self.world.agent.location
        px, py = float(p[0]), float(p[1])

        # heading angle: atan2(y, x); our base triangle points +Y,
        # so rotate by (theta - pi/2) to align tip with heading.
        h = self.world.agent.heading
        theta = np.arctan2(float(h[1]), float(h[0]))
        rot = theta - np.pi/2

        # Build a fresh transform: rotate in data units, then translate to (px, py)
        t = transforms.Affine2D().rotate(rot).translate(px, py) + self.ax_xy.transData
        self.agent_patch.set_transform(t)

        px, py = float(p[0]), float(p[1])
        self.agent_patch.set_transform(t)

        # move sensory circles
        self.agent_outer_circle.center = (px, py)
        self.agent_inner_circle.center = (px, py)


        # Trail
        self.trail_x.append(px)
        self.trail_y.append(py)
        if len(self.trail_x) > self.trail_len:
            self.trail_x = self.trail_x[-self.trail_len:]
            self.trail_y = self.trail_y[-self.trail_len:]
        self.trail_line.set_data(self.trail_x, self.trail_y)

        # Health
        self.health_x.append(step_idx)
        self.health_y.append(float(self.world.agent.health))
        self.health_line.set_data(self.health_x, self.health_y)
        self.ax_health.set_xlim(max(0, step_idx - 1000), step_idx + 10)

        # Draw without blocking
        if self._needs_draw:
            self.fig.canvas.draw_idle()
        plt.pause(0.001)

        # --- pull interpretation from the model every N frames (cheap throttle) ---
       
        model = getattr(self.world.agent, "model", None)
        interp = getattr(model, "interpret", None)
        if callable(interp):
            out = interp(self.world.agent.location, full_interpret=((step_idx % self._interpret_every) == 0))
            if out is not None:
                # Expected: agent_loc, target_loc, memory_locs, memory_classes
                agent_loc, target_loc, mem_locs, mem_classes = out
                
                # Convert to numpy when present; all tensors are assumed batchless & projected into world coords already
                agent_xy  = agent_loc.detach().cpu().numpy() if agent_loc is not None else None
                target_xy = None
                if target_loc is not None:
                    # Handle (1,2) or (2,) shapes
                    t = target_loc.detach().cpu()
                    target_xy = t.squeeze(0).numpy() if t.ndim == 2 else t.numpy()

                if mem_locs is not None:

                    mem_xy = None
                    mem_cl = None
                    if mem_locs.numel() > 0:
                        # Handle (B,K,2) or (K,2)
                        t = mem_locs.detach().cpu()
                        if t.ndim == 3:
                            t = t[0]
                        mem_xy = t.numpy()
                        c = mem_classes.detach().cpu()
                        if c.ndim == 2:
                            c = c[0]
                        mem_cl = c.numpy()

                    # clear memory + belief layers
                    self.mem_edible.set_data([], [])
                    self.mem_poison.set_data([], [])
                    self.mem_neutral.set_data([], [])
                    
                    # Update layers
                    if mem_xy is not None and mem_cl is not None:
                        self.set_memory_interpretation(mem_xy, mem_cl)

                self.agent_belief_pt.set_data([], [])
                self.target_belief_pt.set_data([], [])
                self.target_belief_ring.set_visible(False)

                # If you can compute a target SD radius, put it here; otherwise None
                target_sd_radius = None
                self.set_agent_and_target_beliefs(agent_xy, target_xy, target_sd_radius)

    def set_memory_interpretation(self, mem_xy, drive_targets):
        # Coerce to numpy
        mem_xy = np.asarray(mem_xy)
        drive_targets = np.asarray(drive_targets)

        # Ensure mem_xy is 2D array with shape (N, 2)
        if mem_xy.ndim == 1:
            if mem_xy.size == 2:
                mem_xy = mem_xy.reshape(1, 2)
            else:
                raise ValueError(f"mem_xy has unexpected 1D size {mem_xy.size}, expected 2 or (N,2).")
        elif mem_xy.ndim == 2 and mem_xy.shape[1] != 2:
            raise ValueError(f"mem_xy shape {mem_xy.shape} expected (N, 2).")

        # Flatten drive_targets to shape (N,)
        drive_targets = drive_targets.reshape(-1)

        # If a single class was provided but multiple points exist, broadcast it
        if drive_targets.size == 1 and mem_xy.shape[0] > 1:
            drive_targets = np.full((mem_xy.shape[0],), drive_targets.item(), dtype=drive_targets.dtype)

        # Now dimensions must match
        if drive_targets.shape[0] != mem_xy.shape[0]:
            raise ValueError(
                f"drive_targets length ({drive_targets.shape[0]}) must match mem_xy N ({mem_xy.shape[0]})."
            )

        # Masks (adjust ids if your mapping differs)
        poison_mask  = (drive_targets == 0)
        edible_mask  = (drive_targets == 1)
        neutral_mask = (drive_targets == 2)

        def _set(line, mask):
            if np.any(mask):
                pts = mem_xy[mask]
                line.set_data(pts[:, 0], pts[:, 1])
            else:
                line.set_data([], [])

        _set(self.mem_poison,  poison_mask)
        _set(self.mem_edible,  edible_mask)
        # Don't show neutral memory points
        # _set(self.mem_neutral, neutral_mask)

        self.fig.canvas.draw_idle()

    def set_agent_and_target_beliefs(self, agent_xy=None, target_xy=None, target_sd_radius=None):
        """
        Update belief markers. agent_xy/target_xy: array-like (2,) in world coords.
        target_sd_radius: optional float for a ring radius (same units as world coords).
        """
        if agent_xy is None:
            self.agent_belief_pt.set_data([], [])
        else:
            ax, ay = float(agent_xy[0]), float(agent_xy[1])
            self.agent_belief_pt.set_data([ax], [ay])

        if target_xy is None:
            self.target_belief_pt.set_data([], [])
            self.target_belief_ring.set_visible(False)
        else:
            tx, ty = float(target_xy[0]), float(target_xy[1])
            self.target_belief_pt.set_data([tx], [ty])

            if target_sd_radius is not None and target_sd_radius > 0:
                self.target_belief_ring.center = (tx, ty)
                self.target_belief_ring.set_radius(float(target_sd_radius))
                self.target_belief_ring.set_visible(True)
            else:
                self.target_belief_ring.set_visible(False)

        self.fig.canvas.draw_idle()

    def reset(self, new_world=None, reinit_trees=True):
        """
        Wipe agent trail + health series, and (optionally) rebuild tree scatter
        from the current or provided world.
        """
        if new_world is not None:
            self.world = new_world

        # Trees
        if reinit_trees:
            self._init_trees()

        # Clear trail
        self.trail_x, self.trail_y = [], []
        self.trail_line.set_data([], [])

        # memory + belief layers
        self.mem_edible.set_data([], [])
        self.mem_poison.set_data([], [])
        self.mem_neutral.set_data([], [])
        self.agent_belief_pt.set_data([], [])
        self.target_belief_pt.set_data([], [])
        self.target_belief_ring.set_visible(False)

        # Recompute size if world span changed
        span = self.world.config.tree_spacing * self.world.config.dim * 3
        self.agent_size = 0.03 * span
        tri = np.array([[0.0, 1.0], [-0.7, -0.9], [0.7, -0.9]]) * self.agent_size
        self._agent_tri_local = tri
        self.agent_patch.set_xy(tri)   # local geometry; orientation applied in update()

        self.agent_outer_circle.set_radius(self.world.config.max_sense_distance)
        self.agent_inner_circle.set_radius(self.world.config.can_see_fruit_distance)

        # force a quick redraw
        self.fig.canvas.draw_idle()

        # Clear health graph
        self.health_x, self.health_y = [], []
        self.health_line.set_data([], [])
        self.ax_health.set_xlim(0, 10)  # will auto-extend as steps arrive

        # Re-center axes in case spacing/dim changed
        span = self.world.config.tree_spacing * self.world.config.dim * 3
        self.ax_xy.set_xlim(-span, span)
        self.ax_xy.set_ylim(-span, span)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


    def close(self):
        try:
            if self._cid_key is not None:
                self.fig.canvas.mpl_disconnect(self._cid_key)
            plt.ioff()
            plt.close(self.fig)
        except Exception:
            pass

    # ---------- internals ----------

    def _init_trees(self):
        """(Re)build the poisonous/edible scatter artists from self.world."""
        # Remove prior artists if present
        try:
            if self.s_poison is not None:
                self.s_poison.remove()
            if self.s_edible is not None:
                self.s_edible.remove()
        except Exception:
            pass

        locs = self.world.get_tree_locations().detach().cpu().numpy()
        poisonous = np.array([t.is_poisonous for t in self.world.trees], dtype=bool)

        # Poisonous = black ×
        self.s_poison = self.ax_xy.scatter(
            locs[poisonous, 0], locs[poisonous, 1],
            marker="x", s=30,
            c="black",        # or color="#000000"
            label="poison",
            alpha=0.9,
        )

        # Edible = red ○
        self.s_edible = self.ax_xy.scatter(
            locs[~poisonous, 0], locs[~poisonous, 1],
            marker="o", s=22,
            facecolors="none",   # hollow circles
            edgecolors="red",    # red outlines
            linewidths=1.2,
            label="edible",
            alpha=0.9,
        )


        # Keep legend fresh (in case the axes were cleared)
        if self.show_legend:
            handles, labels = self.ax_xy.get_legend_handles_labels()
            # Deduplicate labels
            uniq = {}
            for h, l in zip(handles, labels):
                uniq[l] = h
            self.ax_xy.legend(uniq.values(), uniq.keys(), loc="upper right")

    # ---------- callbacks ----------

    def _on_reset_clicked(self, _event):
        self.reset(reinit_trees=True)

    def _on_key(self, event):
        if event.key and event.key.lower() == "r":
            self.reset(reinit_trees=True)
