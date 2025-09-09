import sys
from matplotlib import patches
from matplotlib.backend_bases import MouseButton
from matplotlib.path import Path
from tsf.io.bsig import BsigReader
import bird_eye_view_config as config
from PyQt6.QtGui import QPixmap, QAction, QActionGroup, QPainter, QColor
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QDialog, QTreeWidget, \
    QTreeWidgetItem, QMenu, QSlider, QLabel, QLineEdit, QGridLayout, QComboBox, QFileDialog
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import pandas as pd
import numpy as np
import os
import re # for .[digit] path
# import time
# from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

class SignalNotFoundException(Exception):
    pass

if hasattr(config, "CustomBirdEyeSettings"):
    settings = config.CustomBirdEyeSettings()
else:
    settings = config.BaseConfig()

class BirdEveViewObject:
    """
    The BirdEveViewObject class handles visualization and processing of date from a BSIG file for objects and ego paths.

    Attributes:
        bsig_class: Instance of the signal class providing signal data.
        bsig: BSIG data extracted from the bsig_class.
        time_stamps: List of timestamps for the signals.
        current_time_stamp: Current timestamp being processed.
        figure: Matplotlib Figure object for plotting.
        canvas: Matplotlib canvas for rendering the figure.
        ax: Axes object for the main plot.
        object_to_index: Dictionary mapping artists (visual elements) to object indices.
        objects: List of visual elements (patches) currently drawn.
        dx_g, dy_g: Global movement data in x and y directions.
        psi: Series for yaw values.
        cp, sp: Cosine and sine components of yaw.
    """

    def __init__(self, bsig_file):
        """
        Initializes the BirdEveViewObject instance and computes derived quantities like global movement and yaw.

        Args:
            bsig_file: A signal file object containing the data to be visualized.
        """
        self.bsig_class = bsig_file
        self.bsig = self.bsig_class.get_bsig_object()
        self.time_stamps = self.bsig_class.get_timestamps()
        self.current_time_stamp = None
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('draw_event', self.update_grid)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(-30, 30)
        self.ax.set_ylim(-10, 100)
        self.ax.set_facecolor('black')
        self.press = None
        self.cur_xlim = self.ax.get_xlim()
        self.cur_ylim = self.ax.get_ylim()
        self.xpress = None
        self.ypress = None
        self.annotation = None  # Annotation for displaying ID
        self.object_to_index = {}
        self.objects = []  # Initialize the artists list
        self._velocity = pd.Series(self.bsig[settings.VELOCITY], index=self.time_stamps)
        self._acceleration = self.bsig[settings.ACCELERATION]
        self._yaw_rate = pd.Series(self.bsig[settings.YAW_RATE], index=self.time_stamps)
        self._slip_angle = self.bsig[settings.SLIP_ANGLE]

        ts = self._velocity.index.values
        dt = np.zeros(ts.shape)
        dt[0] = (ts[1] - ts[0]) * 1.0e-6
        dt[1:] = np.diff(ts) * 1.0e-6

        # compute the yaw
        psi = self._yaw_rate.values * dt
        cp = np.cos(psi)
        sp = np.sin(psi)

        # If slip angle is available consider it, otherwise assume no slip
        if self._slip_angle is not None:
            slip = self._slip_angle
            cs = np.cos(slip)
            ss = np.sin(slip)
        else:
            cs = 1
            ss = 0

        # compute the covered distance
        ds = self._acceleration / 2. * dt ** 2 + self._velocity * dt

        # split by their components
        dx = ds * np.cos(psi)
        dy = ds * np.sin(psi)

        # Compute global movement starting from (0,0)
        dx_g = 2.75 * (np.cos(psi) - 1) - dx * (cs * cp + ss * sp) + dy * (ss * cp - cs * sp)
        dy_g = -2.75 * np.sin(psi) + dx * (cs * sp + ss * cp) - dy * (cs * cp - ss * sp)

        self.dx_g = pd.Series(dx_g, index=ts)
        self.dy_g = pd.Series(dy_g, index=ts)
        self.psi = pd.Series(psi, index=ts)
        self.cp = pd.Series(cp, index=ts)
        self.sp = pd.Series(sp, index=ts)

    def create_path(self, current_timestamp, preview_duration=None, history_duration=None, y_shift=0.0):
        """
                Computes the ego path based on the current timestamp and optional preview/history durations.

                Args:
                    current_timestamp: The timestamp to center the computation on.
                    preview_duration: Duration for previewing the path forward (optional).
                    history_duration: Duration for history (optional).
                    y_shift: Vertical offset for the path visualization.

                Returns:
                    A DataFrame with x, y coordinates of the ego path.
        """
        # Calculate indices
        current_idx, history, preview = self.__calc_indices(current_timestamp, preview_duration, history_duration)

        # Ensure indices are within bounds for ts, psi_sum, cp_sum, and sp_sum
        start_idx = max(0, current_idx - history)
        end_idx = min(len(self._velocity), current_idx + preview)
        ts = self._velocity.index.values[start_idx:end_idx]

        # Calculate cumulative yaw (psi) and trigonometric functions
        psi_sum = np.cumsum(self.psi.iloc[start_idx:end_idx])

        # Safe centering of psi_sum by checking the length
        if len(psi_sum) > 0:
            center_value = psi_sum.iloc[history] if history < len(psi_sum) else psi_sum.iloc[-1]
            psi_sum -= center_value  # Center psi_sum
        else:
            psi_sum = np.zeros_like(ts)  # fallback in case psi_sum is empty

        cp_sum = np.ones(psi_sum.shape)
        sp_sum = np.zeros(psi_sum.shape)
        cp_sum[1:] = np.cos(psi_sum[:-1])
        sp_sum[1:] = np.sin(psi_sum[:-1])

        cycles = len(ts)
        cx_rel = np.zeros(cycles)
        cy_rel = np.zeros(cycles)

        # Past track loop, with bounds checking
        for c in range(1, min(history + 1, current_idx + 1)):
            if (current_idx - c + 1) < len(self.cp) and (history - c + 1) < len(cx_rel):
                cx_rel[history - c] = (self.cp.iloc[current_idx - c + 1] * cx_rel[history - c + 1] +
                                       self.sp.iloc[current_idx - c + 1] * cy_rel[history - c + 1] +
                                       self.dx_g.iloc[current_idx - c + 1])
                cy_rel[history - c] = (-self.sp.iloc[current_idx - c + 1] * cx_rel[history - c + 1] +
                                       self.cp.iloc[current_idx - c + 1] * cy_rel[history - c + 1] +
                                       self.dy_g.iloc[current_idx - c + 1])

        # Future track loop, with bounds checking
        for c in range(1, min(preview, len(self.cp) - current_idx)):
            if (current_idx + c - 1) < len(self.cp) and (history + c - 1) < len(cx_rel):
                cx_rel[history + c] = (self.cp.iloc[current_idx + c - 1] * cx_rel[history + c - 1] +
                                       self.sp.iloc[current_idx + c - 1] * cy_rel[history + c - 1] +
                                       self.dx_g.iloc[current_idx + c - 1])

                cy_rel[history + c] = (-self.sp.iloc[current_idx + c - 1] * cx_rel[history + c - 1] +
                                       self.cp.iloc[current_idx + c - 1] * cy_rel[history + c - 1] +
                                       self.dy_g.iloc[current_idx + c - 1])

        # Combine tracks to form global x, y positions
        k = np.ones(cycles)
        k[history:] = -1
        cx = k * (cp_sum * cx_rel) + sp_sum * (cy_rel + y_shift)
        cy = -(k * (sp_sum * cx_rel) - cp_sum * (cy_rel + y_shift))

        # Construct DataFrame
        df = pd.DataFrame(index=ts)
        df["x"] = pd.Series(cx, index=ts)
        df["y"] = pd.Series(cy, index=ts)

        return df

    def update_graph(self, index, object_view, show_path):
        """
        Update the graph to display objects and paths based on the specified index, object view, and path visibility.

        :param index: Index of the current timestamp to visualize.
        :param object_view: Type of object representation ("L-Shape" or "Classic").
        :param show_path: Boolean flag to indicate whether to show paths on the graph.
        """
        self.ax.clear()
        self.ax.set_xlim(self.cur_xlim)
        self.ax.set_ylim(self.cur_ylim)
        self.object_to_index.clear()  # Clear the previous mappings
        self.objects = []
        self.current_time_stamp = self.time_stamps[index]
        xlim_min, xlim_max = self.cur_xlim
        ylim_min, ylim_max = self.cur_ylim

        for object_index in range(80):
            dist_x = self.bsig[self.bsig_class.replace_signal(settings.REF_DIST_X, object_index)][index]
            dist_y = self.bsig[self.bsig_class.replace_signal(settings.REF_DIST_Y, object_index)][index]
            # Object dimensions
            vertices = []
            if xlim_min <= dist_y <= xlim_max and ylim_min <= dist_x <= ylim_max:
                if (object_view == "L-Shape"
                        and self.bsig_class.has_signal(self.bsig_class.replace_signal(settings.Left_SHAPE_POINT_X, str(object_index)))):
                    Left_SHAPE_POINT_X = self.bsig_class.replace_signal(settings.Left_SHAPE_POINT_X, str(object_index))
                    Left_SHAPE_POINT_Y = self.bsig_class.replace_signal(settings.Left_SHAPE_POINT_Y, str(object_index))
                    Middle_SHAPE_POINT_X = self.bsig_class.replace_signal(settings.Middle_SHAPE_POINT_X, str(object_index))
                    Middle_SHAPE_POINT_Y = self.bsig_class.replace_signal(settings.Middle_SHAPE_POINT_Y, str(object_index))
                    Right_SHAPE_POINT_X = self.bsig_class.replace_signal(settings.Right_SHAPE_POINT_X, str(object_index))
                    Right_SHAPE_POINT_Y = self.bsig_class.replace_signal(settings.Right_SHAPE_POINT_Y, str(object_index))

                    signal_list = [[Left_SHAPE_POINT_X, Left_SHAPE_POINT_Y],
                                   [Middle_SHAPE_POINT_X, Middle_SHAPE_POINT_Y],
                                   [Right_SHAPE_POINT_X, Right_SHAPE_POINT_Y]]

                    for p, q in enumerate(signal_list):
                        px = self.bsig[q[0]][index]
                        py = self.bsig[q[1]][index]
                        vertices.append((-(dist_y + py), dist_x + px))
                    vertices.append(vertices[1])

                elif (object_view == "Classic" and
                      self.bsig_class.has_signal(self.bsig_class.replace_signal(settings.WIDTH, str(object_index)))):
                    width = self.bsig[self.bsig_class.replace_signal(settings.WIDTH, object_index)][index]
                    length = self.bsig[self.bsig_class.replace_signal(settings.LENGTH, object_index)][index]
                    orientation = self.bsig[self.bsig_class.replace_signal(settings.ORIENTATION, object_index)][index]

                    p0 = np.array([[0, ], [width / 2.]])
                    p1 = np.array([[length], [width / 2.]])
                    p2 = np.array([[length], [-width / 2.]])
                    p3 = np.array([[0], [-width / 2.]])

                    rot = np.array(
                        [[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
                    p0t = np.dot(rot, p0)
                    p1t = np.dot(rot, p1)
                    p2t = np.dot(rot, p2)
                    p3t = np.dot(rot, p3)

                    # Apply the rotation matrix
                    vertices.append((-(dist_y + p0t[1, 0]), dist_x + p0t[0, 0]))
                    vertices.append((-(dist_y + p1t[1, 0]), dist_x + p1t[0, 0]))
                    vertices.append((-(dist_y + p2t[1, 0]), dist_x + p2t[0, 0]))
                    vertices.append((-(dist_y + p3t[1, 0]), dist_x + p3t[0, 0]))

                else:
                    print("No object dimension in bsig")
                    vertices.append((-(dist_y + 0.2), dist_x + 0))
                    vertices.append((-(dist_y + 0.2), dist_x + 0.4))
                    vertices.append((-(dist_y - 0.2), dist_x + 0.4))
                    vertices.append((-(dist_y - 0.2), dist_x + 0))

                vertices.append((None, None))

                if self.bsig_class.has_signal(self.bsig_class.replace_signal(settings.EM_DYN_PROP, object_index)):
                    dynamic_property = self.bsig[self.bsig_class.replace_signal(settings.EM_DYN_PROP, object_index)][index]
                elif self.bsig_class.has_signal(self.bsig_class.replace_signal(settings.ARS_DYN_PROP, object_index)):
                    dynamic_property = self.bsig[self.bsig_class.replace_signal(settings.ARS_DYN_PROP, object_index)][index]
                else:
                    dynamic_property = None

                relative_velocity = self.bsig[self.bsig_class.replace_signal(settings.V_REL_X, object_index)][index]

                facecolor = "#333333"
                if dynamic_property == 0 and relative_velocity <= 0:
                    edg_clr = "red"
                elif dynamic_property == 0 and relative_velocity > 0:
                    edg_clr = "green"
                elif dynamic_property == 1:
                    edg_clr = "yellow"
                elif dynamic_property == 2:
                    edg_clr = "blue"
                elif dynamic_property == 3:
                    edg_clr = "cyan"
                elif dynamic_property == 4:
                    edg_clr = "cyan"
                elif dynamic_property == 5:
                    edg_clr = "magenta"
                elif dynamic_property == 6:
                    edg_clr = "white"

                codes = [
                    Path.MOVETO,
                    Path.LINETO,
                    Path.LINETO,
                    Path.LINETO,
                    Path.CLOSEPOLY,
                ]
                path = Path(vertices, codes)
                patch = patches.PathPatch(path, facecolor=facecolor, edgecolor=edg_clr, alpha=0.8, lw=2.5, zorder=400,
                                          picker=True)
                self.ax.add_patch(patch)
                # Map this patch to loop_index for click handling
                self.object_to_index[patch] = object_index
                self.objects.append(patch)

        if show_path:
            self._make_path(self.time_stamps[index])
            self._make_vdy_curve(index)

        self.update_grid()
        self.canvas.draw()

    def update_grid(self, event=None):
        """
                Update the grid lines on the graph by removing old ones and adding new ones.

                :param event: Optional matplotlib event triggering the update.

        """
        # Remove only the grid lines, not all lines
        for line in self.ax.get_lines():
            if line.get_linestyle() in ['-', '--']:  # Assuming grid lines have solid or dashed styles
                line.remove()

        # Get the current major tick positions for both axes
        x_ticks = self.ax.get_xticks()
        y_ticks = self.ax.get_yticks()

        # Draw vertical grid lines at x-axis major tick positions
        for x in x_ticks:
            self.ax.axvline(x=x, color='white', linestyle='-', linewidth=0.5)

        # Draw horizontal grid lines at y-axis major tick positions
        for y in y_ticks:
            self.ax.axhline(y=y, color='white', linestyle='-', linewidth=0.5)

        # Optionally, you can draw minor grid lines (if desired) by accessing minor ticks
        x_minor_ticks = self.ax.get_xticks(minor=True)
        y_minor_ticks = self.ax.get_yticks(minor=True)

        # Draw minor vertical grid lines
        for x in x_minor_ticks:
            self.ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5)

        # Draw minor horizontal grid lines
        for y in y_minor_ticks:
            self.ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5)

    def _make_vdy_curve(self, current_timestamp):
        """
        Plot the VDY curve for the current timestamp.

        :param current_timestamp: Current timestamp for which the VDY curve is plotted.
        """
        try:
            curve = self.bsig[settings.VDY_CURVE_SIGNAL][current_timestamp]
            self.__plot_radius(curve, "#ffff20")
        except SignalNotFoundException:
            print("VDY curve signal not available")
            return

    def __plot_radius(self, curve, color):
        """
        Calculate and plot a radius segment based on the current curve value.

        :param curve: Curvature value, with sign indicating direction.
        :param color: Color to use for the plotted radius segment.
        """
        # Curve to radius
        if curve != 0:
            radius = 1 / abs(curve)
        else:
            radius = 1000

        if abs(radius) > 1000 or np.isnan(radius):
            radius = 1000

        if curve > 0:
            # left turn
            y = np.arange(radius, radius - 50, -radius / 1000.)
            x = np.sqrt(abs(radius ** 2 - y ** 2))
            radius = -radius
        else:
            # right turn
            y = np.arange(-radius, radius - 50, radius / 1000.)
            x = np.sqrt(abs(radius ** 2 - y ** 2))

        # Addition x limit, to save figure canvas / memory
        xv = np.logical_and(x > -20, x < 250)
        yv = np.logical_and(y > -50 - radius, y < 50 - radius)
        vv = np.logical_and(xv, yv)
        x_idxs_, = np.where(vv)

        xx = x[x_idxs_]
        yy = y[x_idxs_]

        self.ax.plot(yy + radius, xx, color=color, linewidth=1.2, )

    def _make_path(self, current_timestamp):
        """ Plots the ego path which is computed in ego VDY.
        The path will be displayed by two lines with y-offset of +/-0.9 meters roughly matching the ego width.

        :param current_timestamp: Timestamp for which the path is vizualized.
        """

        coordinates = self.create_path(current_timestamp, history_duration=2, preview_duration=10,
                                                      y_shift=0.9)
        right_path = coordinates[['y', 'x']].values

        coordinates = self.create_path(current_timestamp, history_duration=2, preview_duration=10,
                                                      y_shift=-0.9)
        left_path = coordinates[['y', 'x']].values

        p = Polygon(right_path, facecolor="none", edgecolor="green", alpha=1, animated=False, linewidth=1)
        p.set_closed(False)
        self.ax.add_patch(p)

        p = Polygon(left_path, facecolor="none", edgecolor="green", alpha=1, animated=False, linewidth=1)
        p.set_closed(False)
        self.ax.add_patch(p)

    def __calc_indices(self, current_timestamp, preview_duration=None, history_duration=None):
        """ Calculates the current index based on the given timestamp and the number of samples for the given durations
        for history and preview.


        :param current_timestamp:
        :param preview_duration:
        :param history_duration:
        :return:
        """
        # first compute current index
        _current_idx, = np.where(self._velocity.index.values <= current_timestamp)

        # Check if _current_idx is not empty
        if _current_idx.size > 0:
            current_idx = _current_idx[-1] + 1
        else:
            # Handle the case where there are no indices meeting the condition
            current_idx = 0

        if current_idx < 0:
            current_idx = 0

        # required preview samples
        if preview_duration:
            _preview, = np.where(self._velocity.index.values <= current_timestamp + preview_duration * 1e6)
            if len(_preview) > 0:
                preview_samples = _preview[-1] - current_idx
            else:
                preview_samples = 0
        else:
            # entire recording
            preview_samples = len(self._velocity) - current_idx - 1

        # required history samples
        if history_duration:
            _history, = np.where(self._velocity.index.values <= current_timestamp - history_duration * 1e6)
            if len(_history) > 0:
                history_samples = current_idx - _history[-1]
            else:
                history_samples = current_idx  # history is prio rec start. Draw from first sample.
        else:
            history_samples = 0

        return current_idx, history_samples, preview_samples

    def on_pick(self, event):
        """
            Handles mouse click events on graphical objects in the plot.
            Displays a dialog box with information about the clicked object.

            :param event: The matplotlib event triggered by a mouse click.
            """
        # Check if the event was triggered by a mouse click
        if event.mouseevent.button != MouseButton.LEFT:
            return
        # Get the object that was picked
        bird_eye_object = event.artist
        # Retrieve the index from the dictionary
        index = self.object_to_index.get(bird_eye_object, None)
        if index is not None:
            prefix = f"SIM VFB ALL.FPS.FPS_GenObjectListMeas.aObject[{index}].Kinematic."
            self.info_dialog = QDialog()
            self.info_dialog.setWindowTitle(f"Object_{index}")

            layout = QVBoxLayout()
            label = QLabel(f"Clicked on object with index: {index}\n"
                           f"At TimeStamp: {self.time_stamps[np.argmax(self.time_stamps == self.current_time_stamp)]}\n\n"
                           f"fDistX = {self.bsig[f'{prefix}fDistX'][np.argmax(self.time_stamps == self.current_time_stamp)]}\n"
                           f"fDistY = {self.bsig[f'{prefix}fDistY'][np.argmax(self.time_stamps == self.current_time_stamp)]}\n"
                           f"fVrelX= {self.bsig[f'{prefix}fVrelX'][np.argmax(self.time_stamps == self.current_time_stamp)]}\n"
                           f"fVrelY= {self.bsig[f'{prefix}fVrelY'][np.argmax(self.time_stamps == self.current_time_stamp)]}\n"
                           f"fArelX= {self.bsig[f'{prefix}fArelX'][np.argmax(self.time_stamps == self.current_time_stamp)]}\n")
            layout.addWidget(label)

            self.info_dialog.setLayout(layout)
            self.info_dialog.setModal(False)  # Make the dialog non-blocking
            self.info_dialog.show()

    def on_hover(self, event):
        """
        Handles mouse hover events over the plot.
        Highlights objects under the cursor and displays an annotation with the object ID.

        :param event: The matplotlib event triggered by mouse movement.
        """
        if event.inaxes == self.ax:
            for artist in self.objects:
                artist.set_alpha(0.8)

            for artist in self.objects:
                contains, _ = artist.contains(event, radius=5)
                if contains:
                    artist.set_alpha(1.0)
                    index = self.object_to_index.get(artist, None)
                    self.show_annotation(event.xdata, event.ydata, index)
                    break
            else:
                if self.annotation:
                    self.annotation.remove()
                    self.annotation = None

            self.canvas.draw_idle()

    def show_annotation(self, x, y, object_id):
        """
        Displays an annotation for the object currently under the cursor.

        :param x: X-coordinate of the cursor.
        :param y: Y-coordinate of the cursor.
        :param object_id: The ID of the artist being annotated.
        """
        if self.annotation:
            self.annotation.remove()
        self.annotation = self.ax.annotate(
            f"ID: {object_id}", xy=(x, y), xytext=(5, 5),
            textcoords='offset points', ha='center', color='red',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='yellow', alpha=0.5), zorder=10000
        )
        self.canvas.draw_idle()

    def on_scroll(self, event):
        """
        Handles zooming in and out on the plot using the scroll wheel.

        :param event: The matplotlib event triggered by scrolling.
        """
        scale_factor = 1.1
        if event.button == 'up':
            scale_factor = 1 / scale_factor
        elif event.button == 'down':
            scale_factor = scale_factor

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.cur_xlim = self.ax.get_xlim()
        self.cur_ylim = self.ax.get_ylim()

        self.ax.figure.canvas.draw()

    def on_press(self, event=None):
        """
        Handles mouse press events for initiating panning.

        :param event: The matplotlib event triggered by a mouse button press.
        """
        if event.button == MouseButton.LEFT:
            self.press = True
            self.cur_xlim = self.ax.get_xlim()
            self.cur_ylim = self.ax.get_ylim()
            self.xpress = event.xdata
            self.ypress = event.ydata

    def on_motion(self, event=None):
        """
        Handles mouse drag events for panning the plot.

        :param event: The matplotlib event triggered by mouse movement while dragging.
        """
        if self.press is None:
            return
        if event.inaxes != self.ax:
            return
        dx = event.xdata - self.xpress
        dy = event.ydata - self.ypress
        self.ax.set_xlim(self.cur_xlim[0] - dx, self.cur_xlim[1] - dx)
        self.ax.set_ylim(self.cur_ylim[0] - dy, self.cur_ylim[1] - dy)
        self.cur_xlim = self.ax.get_xlim()
        self.cur_ylim = self.ax.get_ylim()
        self.ax.figure.canvas.draw()

    def on_release(self, event=None):
        """
        Handles mouse release events to stop panning.

        :param event: The matplotlib event triggered by releasing a mouse button.
        """
        self.press = None
        self.ax.figure.canvas.draw()

class BsigObject(object):
    """
    Represents an object for storing and interacting with a Bsig file.

    This class provides methods to read signals, filter them based on a config-file
    and access signal data and timestamps.

    Attributes:
                name (str): Name of the Bsig file (path).
                available_signals (list): List of all signal paths available in the Bsig file.
                bsig_object (dict): Filtered signals stored in the Bsig file.
                signal_keys (list): Keys of the filtered signals.
                time_stamps (np.ndarray): Array of timestamps extracted from the signals.
    """

    def __init__(self, path):
        """
            Initializes a BsigObject instance by loading signals from the specified Bsig file path.

            Args:
                path (str): Path to the Bsig file.

        """
        self.name = path
        unique_signals = config.get_unique_signals().values()

        with BsigReader(path) as bsig_obj:
            self.available_signals = list(bsig_obj.signals)  # all Paths

            if unique_signals:  # if config file is there to filter
                expanded_signals_list = []

                for path in unique_signals:
                    if '[{}]' in path:
                        expanded_signals = []  # Temporary list for this path's expansions

                        for i in range(100):
                            indexed_path = path.replace('{}', str(i))  # Generate the indexed path
                            if indexed_path in self.available_signals:
                                expanded_signals.append(indexed_path)

                        expanded_signals_list.extend(expanded_signals)
                    else:
                        expanded_signals_list.append(path)

                filtered_signals = [
                    sig for sig in self.available_signals if sig in expanded_signals_list
                ]
            else:
                filtered_signals = self.available_signals

            self.bsig_object = bsig_obj._signals(filtered_signals)
            self.signal_keys = list(self.bsig_object.keys())
            self.time_stamps = np.array(self.bsig_object['MTS.Package.TimeStamp'])

    def get_timestamps(self):
        """
        Retrieves the timestamps from the Bsig file.

        Returns:
            np.ndarray: Array of timestamps.
        """
        return self.time_stamps

    def get_bsig_object(self):
        """
        Retrieves the filtered signals stored in the Bsig object.

        Returns:
            dict: Dictionary of filtered signals.
        """
        return self.bsig_object

    def replace_signal(self, signal, object_index):
        """
        Replaces a ({}- )placeholder in the signal path with the specified index.

        Args:
            signal (str): Signal path with a placeholder '{}'.
            object_index (int): Index to replace the placeholder.

        Returns:
            str: Signal path with the placeholder replaced by the index.
        """
        if self.has_signal(signal, object_index):
            signal = signal.replace('{}', str(object_index))
        return signal

    def has_signal(self, path, ts_index=None):
        """
        Checks if a signal exists in the Bsig object.

        Args:
            path (str): Signal path to check.
            ts_index (int, optional): Index to replace the placeholder '{}' in the signal path, if applicable.

        Returns:
            bool: True if the signal exists, False otherwise.
        """
        try:
            if ts_index is not None and '[{}]' in path:
                path = path.replace('{}', str(ts_index))
            if len(self.bsig_object[path]) > 0:
                return True
        except KeyError:
            return False

class BsigTreeObject(object):
    """
    A class that manages a QTreeWidget representing a tree of BSIG files and their associated signals.
    """

    def __init__(self):
        """
        Initializes the BsigTreeObject by creating a QTreeWidget, setting its properties, and
        creating a shared root node for the tree.
        """
        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.setHeaderLabels(["BSIG Files"])
        self.root_item = QTreeWidgetItem(self.tree)  # Shared root node
        self.root_item.setText(0, "Bsigs")  # Name the root node
        self.tree.addTopLevelItem(self.root_item)
        self.bsig_class_list = []
        self.current_bsig_file = None

    def set_current_bsig(self, bsig_name):
        for bsig in self.bsig_class_list:
            if bsig.name == bsig_name:
                self.current_bsig_file = bsig.get_bsig_object()

    def add_bsig_to_tree(self, signal_list, bsig_class):
        """
        Adds a new BSIG node and its associated signals to the tree.

        Args:
            signal_list (list): A list of signal names, where each name is a dot-separated string
                                 representing the hierarchy of the signal.
            bsig_class (BsigObject): The bsig_class object to be assigned to the new BSIG node in the tree.
        """
        # Add a new top-level node under the shared root for each BSIG path
        bsig_root = QTreeWidgetItem(self.root_item)  # Add the new BSIG as a child of the shared root

        bsig_root.setText(0, bsig_class.name)

        # Create a nested dictionary for the signals
        self.bsig_class_list.append(bsig_class)
        root = {}
        for signal in signal_list:
            parts = signal.split('.')
            current_level = root

            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

        # Populate the tree under the new BSIG node
        self.add_items(bsig_root, root, "")

    def add_items(self, parent_item, elements, full_name):
        """
        Recursively adds items to the QTreeWidget under the given parent item.

        Args:
            parent_item (QTreeWidgetItem): The parent item under which the new items will be added.
            elements (dict): A dictionary where keys are item names and values are either sub-dictionaries
                             or empty if the item is a leaf.
            full_name (str): The full name (hierarchical path) to associate with each item.
        """
        for key, value in elements.items():
            item = QTreeWidgetItem([key])
            parent_item.addChild(item)
            new_full_name = full_name + "." + key if full_name else key
            item.setData(0, 1, new_full_name)  # Store the full name in the item using Qt.ItemDataRole.UserRole

            if isinstance(value, dict):
                self.add_items(item, value, new_full_name)

    def update_tree_with_values(self, tree_to_update, timestamp):
        self.set_current_bsig(tree_to_update)
        root = self.tree.invisibleRootItem()  # Get the invisible root item
        bsigs_root = root.child(0)  # Get the "Bsigs" root item

        if bsigs_root is not None:
            for i in range(bsigs_root.childCount()):
                child = bsigs_root.child(i)
                child_name = child.text(0)  # Get the name of the child item
                if child_name == tree_to_update:
                    self.update_item(child, timestamp)

    def update_item(self, item, timestamp):
        if item.childCount() == 0 and item.parent() and item.parent().isExpanded():
            full_name = item.data(0, 1)
            last_part = full_name.split('.')[-1]  # last part of the full name
            if not re.match(r'^\[\d+\]$', last_part):  # currently a bug in name format with [n] at the end
                value = np.array(self.current_bsig_file[full_name])[timestamp]
                item.setText(0, f"{last_part} : {value}")
            # else: # funktioniert nicht
            #     original_format = self.revert_string_format(full_name)
            #     y = np.array(self.bsig_object[original_format])
            #     value = y[index]
            #     item.setText(0, f"{last_part} : {value}")
        else:
            for i in range(item.childCount()):
                self.update_item(item.child(i),timestamp)

    def filter_tree(self, text):
        """
        Filters the tree by searching for items whose names contain the given text.

        Args:
            text (str): The search text to filter items by.
        """
        self.tree.setUpdatesEnabled(False)  # Disable updates for performance
        root = self.tree.invisibleRootItem()
        any_visible = self.filter_items(root, text)
        self.tree.setUpdatesEnabled(True)  # Re-enable updates

        if not any_visible:
            # Optionally, show a message or handle the case where no items are found
            print("No matches found")

    def filter_items(self, parent, text):
        """
        Recursively filters the items in the tree and returns whether any item matches the text.

        Args:
            parent (QTreeWidgetItem): The parent item whose children will be checked.
            text (str): The search text to match against the items' labels.

        Returns:
            bool: True if any item matches the search text, False otherwise.
        """
        any_visible = False
        for i in range(parent.childCount()):
            item = parent.child(i)
            if text.lower() in item.text(0).lower():
                self.expand_parents(item)
                any_visible = True
            if self.filter_items(item, text):
                any_visible = True
        return any_visible


    @staticmethod
    def get_item_path(item):
        path = []
        while item is not None:
            full_name = item.data(0, 1)
            if full_name:
                path.insert(0, full_name.split('.')[-1])  # Use only the last part of the full name
            item = item.parent()
        return '.'.join(path)

    @staticmethod
    def expand_parents(item):
        """
        Expands all the parent nodes of the given item to ensure the item is visible in the tree.

        Args:
            item (QTreeWidgetItem): The item whose parent nodes will be expanded.
        """
        parent = item.parent()
        while parent:
            if not parent.isExpanded():
                parent.setExpanded(True)
            parent = parent.parent()

    def remove_bsig_from_tree(self, bsig_name):
        """
        Removes a BSIG node and its associated signals from the tree.

        Args:
            bsig_name (str): The name of the BSIG node to be removed.
        """
        root = self.tree.invisibleRootItem()  # Get the invisible root item
        bsigs_root = root.child(0)  # Get the "BSIGS" root item

        if bsigs_root is not None:
            for i in range(bsigs_root.childCount()):
                child = bsigs_root.child(i)
                if child.text(0) == bsig_name:
                    bsigs_root.removeChild(child)
                    self.bsig_class_list = [bsig for bsig in self.bsig_class_list if bsig.name != bsig_name]
                    print(f"Removed BSIG node: {bsig_name}")
                    return

        print(f"BSIG node with name '{bsig_name}' not found.")

class PlotWindow(QDialog):
    """
    A dialog window that displays a plot with interactive capabilities, such as adding and dragging a vertical line.

    Attributes:
        position_changed (pyqtSignal): Signal emitted when the position of the vertical line changes.
        x (array-like): X-axis data for the initial plot.
        y (array-like): Y-axis data for the initial plot.
        vertical_line (matplotlib.lines.Line2D): Reference to the vertical line in the plot.
        dragging (bool): Indicates whether the vertical line is being dragged.
        canvas (FigureCanvas): Matplotlib canvas for rendering the plot.
        ax (matplotlib.axes.Axes): Axes object for the plot.
    """
    position_changed: pyqtSignal = pyqtSignal(float)

    def __init__(self, x=None, y=None, title="Plot Window", *args, **kwargs):
        """
        Initialize the PlotWindow.

        Args:
            x (array-like, optional): X-axis data for the initial plot.
            y (array-like, optional): Y-axis data for the initial plot.
            title (str, optional): Title of the plot window. Defaults to "Plot Window".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super(PlotWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle(title)
        self.x = x
        self.y = y
        self.vertical_line = None
        self.dragging = False  # Track whether the line is being dragged
        self.canvas = None
        self.ax = None
        self.init_ui()

    def init_ui(self):
        """
        Initialize the user interface by setting up the layout and adding the plot.
        """
        layout = QVBoxLayout()
        self.setLayout(layout)

        if self.x is not None and self.y is not None:
            fig = Figure()
            self.canvas = FigureCanvas(fig)
            layout.addWidget(self.canvas)
            self.ax = fig.add_subplot(111)
            self.ax.plot(self.x, self.y, linewidth=2.0)
            self.canvas.draw()

            self.canvas.mpl_connect('button_press_event', self.on_click)
            self.canvas.mpl_connect('motion_notify_event', self.on_motion)
            self.canvas.mpl_connect('button_release_event', self.on_release)

    def add_plot(self, x, y):
        """
        Add a plot to the existing axes.

        Args:
            x (array-like): X-axis data for the new plot.
            y (array-like): Y-axis data for the new plot.
        """
        self.ax.plot(x, y, linewidth=2.0)
        self.canvas.draw()

    def add_vertical_line(self, x_position):
        """
        Add or update the vertical line at a specified x-axis position.

        Args:
            x_position (float): X-axis position for the vertical line.
        """
        if self.vertical_line is None:
            self.vertical_line = self.ax.axvline(x=x_position, color='r', linestyle='--')
        else:
            self.vertical_line.set_xdata([x_position])  # Set xdata as a list
        self.canvas.draw()

    def update_vertical_line(self, x_position):
        """
        Update the position of the existing vertical line.

        Args:
            x_position (float): New X-axis position for the vertical line.
        """
        if self.vertical_line is not None:
            self.vertical_line.set_xdata([x_position])
            self.canvas.draw()

    def on_click(self, event):
        """
        Handle mouse click events to add or start dragging the vertical line.

        Args:
            event (MouseEvent): Matplotlib mouse event.
        """
        if event.inaxes != self.ax:
            return
        if self.vertical_line is None:
            self.add_vertical_line(event.xdata)
        else:
            contains, _ = self.vertical_line.contains(event)
            if contains:
                self.dragging = True  # Start dragging

    def on_motion(self, event):
        """
        Handle mouse motion events to drag the vertical line.

        Args:
            event (MouseEvent): Matplotlib mouse event.
        """
        if event.inaxes != self.ax or not self.dragging:
            return
        self.add_vertical_line(event.xdata)
        self.position_changed.emit(event.xdata)

    def on_release(self, event):
        """
        Handle mouse release events to stop dragging the vertical line.

        Args:
            event (MouseEvent): Matplotlib mouse event.
        """
        if event.inaxes != self.ax:
            return
        if self.dragging:
            self.dragging = False  # Stop dragging
            self.position_changed.emit(event.xdata)

class CustomSlider(QSlider):
    """
        A custom slider widget with the ability to mark start and end points
        using timestamps and display them as vertical lines on the slider.
    """
    def __init__(self, orientation, parent=None):
        """
            Initializes the CustomSlider.

            Args:
                orientation (Qt.Orientation): The orientation of the slider (horizontal or vertical).
                parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(orientation, parent)
        self.timestamps = []
        self.start = None
        self.end = None
        self.start_index = None
        self.end_index = None

    def set_timestamps(self, time):
        """
            Sets the timestamps for the slider.

            Args:
                time (nd): A numpy array of timestamps.
        """
        self.timestamps = time

    def contextMenuEvent(self, event):
        """
        Handles the context menu event triggered by right-clicking on the slider.
        Adds options to label start, label end, or reset the slider values.

        Args:
            event (QContextMenuEvent): The context menu event.
        """
        # Create the context menu
        menu = QMenu(self)

        # Create actions for the context menu
        reset_action = QAction("Reset", self)
        menu.addAction(reset_action)

        label_start_action = QAction("Label as Start", self)
        menu.addAction(label_start_action)

        label_end_action = QAction("Label as End", self)
        menu.addAction(label_end_action)

        save_label_action = QAction("Save Labels", self)
        menu.addAction(save_label_action)

        # Connect actions to their corresponding functions
        reset_action.triggered.connect(self.reset_slider)
        label_start_action.triggered.connect(self.label_start)
        label_end_action.triggered.connect(self.label_end)
        save_label_action.triggered.connect(self.save_label)

        # Show the menu at the global mouse position
        menu.exec(self.mapToGlobal(event.pos()))

    def reset_slider(self):
        """
            Function to reset the slider value to the minimum and clears start and end markers.
        """
        self.setValue(self.minimum())
        self.start = None
        self.end = None
        self.start_index = None
        self.end_index = None
        self.update()


    def label_start(self):
        """
            Labels the current slider position as the start point and redraws the slider.
        """
        current_value = self.value()  # Get the current value of the slider
        self.start = self.timestamps[current_value]
        self.start_index = current_value
        print(f"Current start: {self.start}")
        self.update()  # Redraw the slider to show the start line

    def label_end(self):
        """
            Labels the current slider position as the start point and redraws the slider.
        """
        current_value = self.value()  # Get the current value of the slider
        self.end = self.timestamps[current_value]
        self.end_index = current_value
        print(f"Current end: {self.end}")
        self.update()  # Redraw the slider to show the end line

    def save_label(self):
        pass
        # save in txt for example

    def paintEvent(self, event):
        """
            Overrides the paint event to draw custom vertical lines representing
            the start and end positions on the slider.

            Args:
                event (QPaintEvent): The paint event.
        """
        super().paintEvent(event)

        # Get the range of the slider
        min_value = self.minimum()
        max_value = self.maximum()

        painter = QPainter(self)

        # Draw start line if start exists
        if self.start is not None and self.start_index is not None:
            start_position = (self.start_index - min_value) / (max_value - min_value) * self.width()
            start_position = int(start_position)  # Cast to int
            painter.setPen(QColor(255, 0, 0))  # Red color
            painter.drawLine(start_position, 0, start_position, self.height())  # Draw vertical line

        # Draw end line if end exists
        if self.end is not None and self.end_index is not None:
            end_position = (self.end_index - min_value) / (max_value - min_value) * self.width()
            end_position = int(end_position)  # Cast to int
            painter.setPen(QColor(0, 0, 255))  # Blue color for the end line
            painter.drawLine(end_position, 0, end_position, self.height())  # Draw vertical line

def process_bsig_path(bsig_path):
    """Standalone function to process a single bsig_path."""
    tmp = BsigObject(bsig_path)
    time_stamps = tmp.get_timestamps()
    return tmp, time_stamps

class MainWindow(QMainWindow):
    """
        Represents the main application window for the evaluation tool.
        Manages the layout, widgets, user interactions, and core functionality
        of the application.
    """
    def __init__(self, *args, **kwargs):
        """
            Initializes the main window and its components, including menu actions,
            layout configurations, and signal/slot connections.
        """
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("evaltool")
        self.file_tag = '_at_'
        self.bsig_path_list = []

        # Horizontal slider for the TimeStamps
        # self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider = CustomSlider(Qt.Orientation.Horizontal)
        self.time_stamps = None
        self.available_signals = []
        self.bsig_list = []
        self.bsig_class_obj = None
        self.current_bsig_file = None

        self.bsig_selector = QComboBox()
        self.bsig_selector.currentIndexChanged.connect(self.on_bsig_selected)
        self.menu_bar = self.menuBar()
        self.time_stamp_label = QLineEdit(self)
        self.time_stamp_label.setPlaceholderText(f"{self.time_stamps}")
        self.time_stamp_label.returnPressed.connect(self.on_timestamp_entered)

        self.image_frame = QLabel() # for video frames (jpeg)

        self.filter_text = QLineEdit()
        self.filter_text.setPlaceholderText("Filter signals...")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.on_search_button_clicked)

        self.canvas = None
        self.bird_eye = None
        self.bird_eye_list = []

        self.tree = BsigTreeObject()
        self.tree.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.tree.customContextMenuRequested.connect(self.open_menu)

        # Set up the grid layout
        self.grid_layout = QGridLayout()
        self.config_widgets()

        container = QWidget()
        container.setLayout(self.grid_layout)
        self.setCentralWidget(container)
        self.plot_windows = {}
        self.window_counter = 0
        self.new_windows = []  # List to keep references to new windows
        self.show_path = True
        self.show_bird_eye_view = True
        self.object_view = "Classic"

    def config_widgets(self):
        """
            Configures the layout and widgets of the main window, including the menu bar,
            sliders, buttons, and other interactive elements.
        """
        self.image_frame.setFixedSize(300, 300)

        file_menu = self.menu_bar.addMenu("File")
        edit_menu = self.menu_bar.addMenu("Edit")
        view_menu = self.menu_bar.addMenu("View")
        about_menu = self.menu_bar.addMenu("About")

        # Add actions to the menu
        open_action = QAction("Open File", self)
        open_action.triggered.connect(self.open_file_explorer)
        file_menu.addAction(open_action)

        new_window_action = QAction("New Window", self)
        new_window_action.triggered.connect(self.open_new_window)
        file_menu.addAction(new_window_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.quit)
        file_menu.addAction(exit_action)

        # Add a checkable action to the View menu
        toggle_action = QAction("Show Path", self, checkable=True)
        toggle_action.setChecked(True)  # Set the initial state to checked
        toggle_action.triggered.connect(self.show_path_option)
        view_menu.addAction(toggle_action)

        # Add a checkable action to the View menu
        toggle_action = QAction("Show Bird Eye View", self, checkable=True)
        toggle_action.setChecked(True)  # Set the initial state to checked
        toggle_action.triggered.connect(self.show_bird_eye_view_option)
        view_menu.addAction(toggle_action)

        # Add a group of exclusive checkable actions to the View menu
        action_group = QActionGroup(self)
        option1 = QAction("Classic", self, checkable=True)
        option2 = QAction("L-Shape", self, checkable=True)

        option1.setChecked(True)

        action_group.addAction(option1)
        action_group.addAction(option2)

        view_menu.addAction(option1)
        view_menu.addAction(option2)

        option1.triggered.connect(lambda: self.select_option("Classic"))
        option2.triggered.connect(lambda: self.select_option("L-Shape"))

        self.grid_layout.addWidget(self.bsig_selector, 0, 0, 1, 5)  # Add ComboBox to the layout
        self.grid_layout.addWidget(self.slider, 1, 0, 1, 5)
        self.grid_layout.addWidget(self.filter_text, 2, 0, 1, 4)  # Filter text spans 4 columns
        self.grid_layout.addWidget(self.search_button, 2, 4)  # Search button next to filter text
        self.grid_layout.addWidget(self.image_frame, 3, 0)
        # self.grid_layout.addWidget(self.canvas, 3, 1, 8, 1) # todo change into bird eye object
        # self.grid_layout.addWidget(self.tree, 3, 2, 6, 3) # todo change into tree object
        self.grid_layout.addWidget(self.time_stamp_label, 9, 2, 1, 3)

    def on_timestamp_entered(self):
        """
            Handles the event when the user enters a timestamp manually in the input field.
            Finds the closest timestamp in the data and updates the slider and other UI components accordingly.
        """
        entered_text = self.time_stamp_label.text().strip()
        try:
            entered_timestamp = int(entered_text)
        except ValueError:
            print("Invalid input: Please enter a numeric timestamp.")
            return

        if entered_timestamp <= self.time_stamps[0]:
            closest_index = 0
        elif entered_timestamp >= self.time_stamps[-1]:
            closest_index = len(self.time_stamps) - 1
        else:
            closest_index = min(range(len(self.time_stamps)),key=lambda i: abs(self.time_stamps[i] - entered_timestamp))

        self.slider.setValue(closest_index)
        self.update_slider_value()

    def update_position(self, position):
        """
            Updates the slider and associated components to reflect a new position
            based on a given timestamp or position.

            :param position: The timestamp to align the slider and components with.
        """
        # Find the closest index to the position
        closest_index = min(range(len(self.time_stamps)), key=lambda i: abs(self.time_stamps[i] - position))
        self.slider.blockSignals(True)  # Block signals to prevent recursive updates
        self.slider.setValue(closest_index)
        self.update_slider_value()
        for plot_window in self.plot_windows.values():
            plot_window.update_vertical_line(position)
        self.slider.blockSignals(False)  # Re-enable signals

    def update_image(self, index):
        """
            Updates the displayed image based on the given index.
            The closest image is selected by name to correspond with the current timestamp.

            :param index: The index of the timestamp to determine the image to display.
        """

        # ToDo: use folder path of corresponding bsig, each folder is named after the bsig (unique)
        folder_path = "./2021.05.03_at_10.46.53_radar-mi_5304"
        image_paths = self.get_image_paths(folder_path)

        if 0 <= index < len(self.time_stamps):
            current_timestamp = self.time_stamps[index]
            closest_image = self.find_closest_image(current_timestamp, image_paths)

            if closest_image:
                pixmap = QPixmap(closest_image)
                scaled_pixmap = pixmap.scaled(self.image_frame.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
                self.image_frame.setPixmap(scaled_pixmap)
            else:
                self.image_frame.clear()
        else:
            self.image_frame.clear()

    def open_file_explorer(self):
        """
            Opens a file dialog to allow the user to select BSIG files for analysis.
            Newly selected files are added to the list of files to be processed.
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select BSIG Files",
            "",
            "BSIG Files (*.bsig);;All Files (*)"
        )
        if file_paths:
            bsigs_to_open = []
            for file_path in file_paths:
                if file_path not in self.bsig_path_list:
                    self.bsig_path_list.append(file_path)
                    bsigs_to_open.append(file_path)
            if bsigs_to_open:
                self.open_bsigs(bsigs_to_open)

    def open_bsigs(self, paths):
        """
        Opens the specified BSIG files, processes them, and updates the application with
        the new data. Also sets up the bird's-eye view and slider with the time stamps
        from the first BSIG file.

        Args:
            paths (list of str): List of file paths to the BSIG files to open.
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_bsig_path, bsig_path) for bsig_path in paths]
            results = [future.result() for future in futures]

        # Append the new results to self.bsig_list
        new_bsig_objects = [result[0] for result in results]
        self.bsig_list.extend(new_bsig_objects)  # Append the new BSIG objects to the list

        self.bsig_class_obj = self.bsig_list[0]
        self.current_bsig_file = self.bsig_class_obj.get_bsig_object()
        self.grid_layout.addWidget(self.tree.tree, 3, 2, 6, 3)
        for temp_obj in new_bsig_objects:
            self.tree.add_bsig_to_tree(self.convert_string_format(temp_obj.signal_keys), temp_obj)
            self.bsig_selector.addItem(temp_obj.name)  # Add each BSIG's name to the ComboBox

        self.bird_eye = BirdEveViewObject(self.bsig_class_obj)
        self.bird_eye_list.append(self.bird_eye)
        self.canvas = self.bird_eye.canvas
        self.grid_layout.addWidget(self.canvas, 3, 1, 8, 1)
        self.time_stamps = np.array(self.current_bsig_file['MTS.Package.TimeStamp'])
        self.slider.setEnabled(True)
        self.slider.setRange(0, len(self.time_stamps) - 1)
        self.slider.set_timestamps(self.time_stamps)
        self.slider.valueChanged.connect(self.update_slider_value)
        self.update_slider_value()

    def on_bsig_selected(self, index):
        """
        Updates the current BSIG file and BirdEyeViewObject when a new BSIG is selected
        from the ComboBox. Reinitializes the timestamps and updates the canvas for the
        birds eye view.

        Args:
            index (int): Index of the selected BSIG file in the ComboBox.
        """
        if 0 <= index < len(self.bsig_list):
            self.bsig_class_obj = self.bsig_list[index]
            self.current_bsig_file = self.bsig_class_obj.get_bsig_object()
            # Update BirdEveViewObject
            #for bird_eye in self.bird_eye_list:
            #    if bird_eye.bsig_class.name == self.bsig_class_obj.name:
            #        self.bird_eye = bird_eye
            self.bird_eye = BirdEveViewObject(self.bsig_class_obj)
            new_canvas = self.bird_eye.canvas
            # Remove the old canvas from the layout
            if self.canvas:
                self.grid_layout.removeWidget(self.canvas)
                self.canvas.deleteLater()  # Remove the old canvas widget to free memory

            # Add the new canvas
            self.canvas = new_canvas
            self.grid_layout.addWidget(self.canvas, 3, 1, 8, 1)
            self.time_stamps = self.bsig_class_obj.get_timestamps()
            self.slider.setRange(0, len(self.time_stamps) - 1)
            self.slider.set_timestamps(self.time_stamps)
            self.update_slider_value()

    def show_path_option(self, checked):
        """
                Toggles the display of paths in the bird eye view based on the checked state
                of the "Show Path" menu option.

                Args:
                    checked (bool): True if the "Show Path" option is enabled, False otherwise.
        """
        if checked:
            self.show_path = True
            self.bird_eye.update_graph(self.slider.value(), self.object_view, self.show_path)
            #self.update_graph(self.slider.value())
        else:
            self.show_path = False
            self.bird_eye.update_graph(self.slider.value(), self.object_view, self.show_path)
            #self.update_graph(self.slider.value())

        # Keep the View menu open
        self.menuBar().setActiveAction(self.menuBar().actions()[2])

    def show_bird_eye_view_option(self, checked):
        """
            Toggles the visibility of the bird's-eye view based on the checked state
            of the "Show Bird Eye View" menu option.

            Args:
                checked (bool): True if the "Show Bird Eye View" option is enabled,
                                False otherwise.
        """
        if checked:
            self.show_bird_eye_view = True
            self.canvas.show()
            self.bird_eye.update_graph(self.slider.value(), self.object_view, self.show_path)
            # self.update_graph(self.slider.value())
        else:
            self.show_bird_eye_view = False
            self.canvas.hide()
            # self.update_graph(self.slider.value())

        # Keep the View menu open
        self.menuBar().setActiveAction(self.menuBar().actions()[2])

    def select_option(self, param):
        """
            Updates the object view mode (e.g., "Classic" or "L-Shape") in the bird's-eye view
            and refreshes the graph accordingly.

            Args:
                param (str): The selected object view mode.
        """
        self.object_view = param
        self.bird_eye.update_graph(self.slider.value(), self.object_view, self.show_path)
        # self.update_graph(self.slider.value())

    def plot_item(self, item):
        """
            Creates a new plot window for the selected item in the tree, plotting its data against the time stamps.
            Adds a vertical line to the plot at the current slider position.

            Args:
                item (QTreeWidgetItem): The tree item representing the signal to be plotted.
        """
        full_name = self.get_item_path(item)  # Retrieve the full path of the item
        # print("Clicked on:", full_name)
        y = np.array(self.current_bsig_file[full_name])
        x = np.array(self.current_bsig_file['MTS.Package.TimeStamp'])
        self.window_counter += 1
        title = f"Plot Window {self.window_counter}"
        plot_window = PlotWindow(x, y, title=title)
        index = self.slider.value()
        value = self.time_stamps[index]
        plot_window.add_vertical_line(value)
        plot_window.show()  # Use show() instead of exec() to allow multiple windows
        self.plot_windows[title] = plot_window
        plot_window.position_changed.connect(self.update_position)
        plot_window.finished.connect(lambda: self.remove_plot_window(title))

    def remove_plot_window(self, title):
        """
            Removes a plot window from the internal list of active plot windows by its title.

            Args:
                title (str): The title of the plot window to be removed.
        """
        if title in self.plot_windows:
            del self.plot_windows[title]

    def add_plot_to_existing(self, item, title=""):
        """
            Adds a new signal plot to an existing plot window by title.

            Args:
                item (QTreeWidgetItem): The tree item representing the signal to be plotted.
                title (str): The title of the existing plot window to add the plot to.
        """
        full_name = self.get_item_path(item)  # Retrieve the full path of the item
        y = np.array(self.current_bsig_file[full_name])
        x = np.array(self.current_bsig_file['MTS.Package.TimeStamp'])

        # Assuming you want to add the plot to the first existing plot window
        plot_window = self.plot_windows[title]
        plot_window.add_plot(x, y)  # You need to implement add_plot method in PlotWindow class

    def on_item_clicked(self, item):
        """
            Handles item clicks in the signal tree. If the clicked item is a leaf node
            (i.e., no children), it plots the item's signal data.

            Args:
                item (QTreeWidgetItem): The tree item that was clicked.
        """
        if item.childCount() == 0 and item.parent() and item.parent() != self.tree.root_item:
            self.plot_item(item)
            # self.adjustSize()

    def open_menu(self, position):
        """
            Displays a context menu at the specified position with options to plot signals,
            copy paths or timestamps, remove BSIGs, and collapse all tree items. Provides
            additional functionality based on the currently selected item in the tree.

            Args:
                position (QPoint): The position where the context menu is invoked.
        """
        menu = QMenu()
        item = self.tree.tree.itemAt(position)

        new_plot_action = menu.addAction("Show in new osci view")
        new_plot_action.setDisabled(not (item and item.childCount() == 0))

        add_plot_to_existing_menu = menu.addMenu("Add to osci view..")
        add_plot_to_existing_menu.setDisabled(len(self.plot_windows) == 0)

        copy_path_action = menu.addAction("Copy URL to clipboard")
        copy_current_timestamp_action = menu.addAction("Copy MTS TimeStamp to clipboard")
        remove_bsig_action = menu.addAction("Remove BSIG")

        # Disable the remove action if the item is None or its parent is not the root item
        if item is None or item.parent() != self.tree.root_item:
            remove_bsig_action.setDisabled(True)

        collapse_all_action = menu.addAction("Collapse All")

        for title in self.plot_windows:
            action = add_plot_to_existing_menu.addAction(title)
            action.triggered.connect(lambda checked, t=title: self.add_plot_to_existing(item, t))

        action = menu.exec(self.tree.tree.viewport().mapToGlobal(position))

        if action == new_plot_action:
            if item and item.childCount() == 0:  # Check if the item is a leaf node (end of tree)
                self.plot_item(item)

        elif action == add_plot_to_existing_menu:
            if item and item.childCount() == 0:  # Check if the item is a leaf node (end of tree)
                self.add_plot_to_existing(item)

        elif action == copy_path_action:
            if item:
                clipboard = QApplication.clipboard()
                clipboard.setText(item.data(0, 1))  # Copy the full path to the clipboard

        elif action == copy_current_timestamp_action:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.time_stamp_label.text())

        elif action == collapse_all_action:
            self.tree.tree.collapseAll()

        elif action == remove_bsig_action:
            if item and item.parent() == self.tree.root_item:
                bsig_name = item.text(0)
                self.tree.remove_bsig_from_tree(bsig_name)
                # Remove from combobox bsig_selector
                index = self.bsig_selector.findText(bsig_name)
                if index != -1:
                    self.bsig_selector.removeItem(index)

                # Remove from bsig_list
                self.bsig_list = [bsig for bsig in self.bsig_list if bsig.name != bsig_name]
                self.bsig_path_list = [path for path in self.bsig_path_list if path != bsig_name]

                # Handle the case where the last BSIG is removed
                if not self.bsig_list:
                    # Clear canvas content and disable slider
                    self.canvas.figure.clear()
                    self.canvas.draw()
                    self.slider.setEnabled(False)
                else:
                    # If current BSIG was closed and bsig_list is not empty, get the first element from the list
                    self.on_bsig_selected(0)

    def update_slider_value(self):
        """
            Updates the label, tree, image, and bird's-eye view based on the slider's current
            value. The value corresponds to a specific timestamp, which is reflected in all
            relevant views.
        """
        index = self.slider.value()
        value = self.time_stamps[index]
        self.time_stamp_label.setText(f"{value}")
        self.tree.update_tree_with_values(self.bsig_class_obj.name, self.slider.value())
        self.update_image(int(index))
        if self.show_bird_eye_view:
            self.bird_eye.update_graph(self.slider.value(), self.object_view, self.show_path)

    def on_search_button_clicked(self):
        """
            Filters the tree view based on the text entered in the filter input field.
            Matches are displayed, and non-matching items are hidden.
        """
        text = self.filter_text.text()
        self.tree.filter_tree(text)

    def open_new_window(self):
        """
            Creates and opens a new instance of the MainWindow class. The new window is
            tracked in the application to manage multiple active instances.
        """
        new_window = MainWindow()
        new_window.show()
        self.new_windows.append(new_window)

    @staticmethod
    def get_image_paths(folder_path):
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    @staticmethod
    def find_closest_image(timestamp, image_paths):
        closest_image = None
        min_diff = float('inf')

        for image_path in image_paths:
            # Extract the number from the image filename
            image_name = os.path.basename(image_path)
            image_number = int(image_name.split('.')[0])
            # ToDo: Make sure image_name is a number

            # Calculate the difference between the timestamp and the image number
            diff = np.abs(np.int64(timestamp) - np.int64(image_number))

            if diff < min_diff:
                min_diff = diff
                closest_image = image_path

        return closest_image

    @staticmethod
    def convert_string_format(input_list):
        output_list = []
        for item in input_list:
            if isinstance(item, tuple):
                base, index = item
                formatted_string = f"{base}.[{index}]"
                output_list.append(formatted_string)
            else:
                output_list.append(item)
        return output_list

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())