from ipywidgets import (
    IntSlider,
    Output,
    interactive_output,
    fixed,
    Checkbox,
    VBox,
    HBox,
    Play,
    jslink,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import h5py
import pandas as pd


from plotly.offline import init_notebook_mode
import plotly.graph_objects as go
from scipy.stats import norm
import seaborn as sns

import sys

# Check if running in Google Colab
if "google.colab" not in sys.modules:
    from IPython.display import display


def tokamak_plasma_cross_section(R0=0.48, a=0.3, kappa=1.5, delta=0.3, num_points=200):
    """
    Generates a symmetric tokamak core plasma cross-section with elongation (κ) and triangularity (δ).

    Parameters:
        R0 (float): Major radius (center of the plasma).
        a (float): Minor radius (plasma width).
        kappa (float): Elongation factor (vertical stretching).
        delta (float): Triangularity factor (shaping).
        num_points (int): Number of points to generate for smoothness.

    Returns:
        x, y (numpy arrays): Coordinates of the tokamak plasma boundary.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Parametric equations for the plasma boundary
    x = R0 + a * np.cos(theta) - delta * a * np.sin(theta) ** 2  # Add triangularity
    y = kappa * a * np.sin(theta)  # Apply elongation

    return x, y


class LOSPlotter:
    st_40_geometry = np.array(
        [[0.15, 0.95, 0.95, 0.15, 0.15], [0.7, 0.7, -0.7, -0.7, 0.7]]  # R  # z
    )

    def __init__(
        self,
        sensor_df,
        qoi_df=None,
        geometry_df=None,
        plasma_nc_filename="Imaging_states_handsOn.nc",
    ):
        self.sensor_df = sensor_df
        if qoi_df is None:
            qoi_df = pd.read_csv("qoi.csv")
        self.qoi_df = qoi_df
        if geometry_df is None:
            geometry_df = pd.read_csv("Imaging_geometry_handsOn.csv")
        self.geometry_df = geometry_df
        self.plasma_states, self.plasma_coords = self.interpret_state_nc(
            plasma_nc_filename
        )
        # self.z_eff_ave = self.qoi_df["Z_eff_ave"]
        # self.z_eff_max = self.qoi_df["Z_eff_max"]

        self.values = sensor_df.values
        self.n_rows = len(self.values)
        self.norm = plt.Normalize(self.values.min(), self.values.max())
        self.cmap = plt.cm.plasma

        self.origin_R = self.geometry_df["R start (m)"]
        self.origin_z = self.geometry_df["Z start"]
        self.end_R = self.geometry_df["R end (m)"]
        self.end_z = self.geometry_df["Z end"]

        self.measurement_range = (
            self.sensor_df.min().min(),
            self.sensor_df.max().max(),
        )
        self.plasma_state_range = (
            np.nanmin(self.plasma_states),
            np.nanmax(self.plasma_states),
        )

        self.output = Output()

    def __call__(self):
        self.display_interactive_plot()

    def plot_los(
        self, selected_lines=None, run=0, print_str=None, interacted=False, **kwargs
    ):
        if selected_lines is None:
            selected_lines = [k for k, v in kwargs.items() if v]

        if interacted:
            self.chosen_design = selected_lines

        selected_inds = [self.sensor_df.columns.get_loc(col) for col in selected_lines]

        num_lines = len(selected_inds)
        self.colours = self.cmap(self.norm(self.values[run]))

        with self.output:
            self.output.clear_output(wait=True)

            # Create figure with GridSpec to control layout
            fig, axs = plt.subplots(
                1,
                3,
                figsize=(10, 6),
                gridspec_kw={"width_ratios": [0.05, 1, 0.05]},
                dpi=100,
            )

            # Extract the axes
            cax_left, ax, cax_right = axs

            # Add secondary y-axis on the right side
            ax_right = ax.twinx()

            for ind in selected_inds:
                ax.plot(
                    [self.origin_R[ind], self.end_R[ind]],
                    [self.origin_z[ind], self.end_z[ind]],
                    color=self.colours[ind],
                    lw=2,
                )

            # Determine indices for labeling (first, middle, last)
            label_indices = [0, num_lines // 2, num_lines - 1][:num_lines]
            for i, j in enumerate([selected_inds[i] for i in label_indices]):
                sf = 0.05 + (i / len(label_indices)) * 0.1
                ax.text(
                    self.end_R[j] + sf * (self.end_R[j] - self.origin_R[j]),
                    self.end_z[j] + sf * (self.end_z[j] - self.origin_R[j]),
                    self.sensor_df.columns[j],
                    fontsize=8,
                    ha="right",
                    va="center",
                )

            # Add Z eff
            ax.text(
                -0.6,
                0.6,
                "$\\langle Z_{\\mathrm{eff.}} \\rangle$ = "
                + f"{(np.mean(self.qoi_df.iloc[run].values)):.1f}",
                fontsize=12,
                ha="center",
                va="center",
            )
            ax.text(
                -0.6,
                -0.6,
                "$Z_{\\mathrm{eff., max}}$ = "
                + f"{(np.max(self.qoi_df.iloc[run].values)):.1f}",
                fontsize=12,
                ha="center",
                va="center",
            )

            # Draw bounding box
            ax.plot(*self.st_40_geometry, "k-", lw=2, label="ST40 Boundary")
            ax.plot(-self.st_40_geometry[0], self.st_40_geometry[1], "k-", lw=2)

            # Draw tokamak cross section (assuming function is defined elsewhere)
            ax.plot(
                *tokamak_plasma_cross_section(),
                "k--",
                lw=1,
                label="Plasma Cross Section",
            )
            ax.plot(
                -tokamak_plasma_cross_section()[0],
                tokamak_plasma_cross_section()[1],
                "k--",
                lw=1,
            )

            # Add the plasma states (flipping R values)
            im = ax.pcolormesh(
                -self.plasma_coords[0],
                self.plasma_coords[1],
                self.plasma_states[run],  # .reshape(self.plasma_shape),
                cmap="inferno",
                shading="nearest",
                vmin=self.plasma_state_range[0],
                vmax=self.plasma_state_range[1],
            )

            # Add diagnostic colorbar
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax_right, orientation="vertical")
            cbar.set_label("Sensor (XRCS) Measurement [Wm$^{-2}$]")
            cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(5, 5))

            # Add plasma state colorbar
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
            sm.set_array([])
            cbar = plt.colorbar(im, cax=cax_left, orientation="vertical")
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.set_label_position("left")
            cbar.set_label("Plasma State [Wm$^{-3}$]")
            cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(5, 5))

            ax.set_xlim(-1.0, 1)
            # ax.set_ylim(-0.8, 1.0)

            ax_right.set_ylabel("Height [m]")
            ax_right.set_ylim(
                ax.get_ylim()
            )  # Ensure the right axis matches the main plot's y-range
            ax_right.yaxis.set_label_position("right")
            ax_right.yaxis.set_ticks_position("right")

            ax.set_aspect("equal")
            ax.set_xlabel("Major Radius [m]")
            ax.set_ylabel("Height [m]")
            ax.legend(ncol=2, loc="upper center")
            plt.subplots_adjust(wspace=1)
            ax.set_aspect(1.0)

            if print_str is not None:
                print(print_str)

            plt.show()
            plt.close()

    def display_plot(self, selected_lines=None, print_str=None):
        if selected_lines is None:
            selected_lines = list(self.sensor_df.columns)
        slider = IntSlider(
            min=0, max=len(self.sensor_df) - 1, step=1, value=0, continuous_update=True
        )
        play = Play(
            interval=50,
            value=0,
            min=0,
            max=self.n_rows - 1,
            step=1,
            description="Press play",
            disabled=False,
            repeat=True,
        )
        # Link the play button to the slider
        jslink((play, "value"), (slider, "value"))
        controls = HBox([play, slider])

        # Make the interactive plot link and return
        out = interactive_output(
            self.plot_los,
            {
                "run": slider,
                "selected_lines": fixed(selected_lines),
                "print_str": fixed(print_str),
                "interacted": fixed(False),
            },
        )

        # Display the combined controls and the output
        display(VBox([controls, out]))

        return self.output

    def display_interactive_plot(self, lines_of_sight=None):
        column_names = list(self.sensor_df.columns)
        if lines_of_sight is None:
            # Default to first 3
            lines_of_sight = [0, 1, 2]
        if isinstance(lines_of_sight[0], int):
            # It is an index list rather than specific reference to column_names
            lines_of_sight = [column_names[i] for i in lines_of_sight]
        checkboxes = {
            option: Checkbox(
                description=option,
                value=option in lines_of_sight,
                # layout=widgets.Layout(width=width)  # Fixed width per checkbox
            )
            for option in column_names
        }
        row_size = 5
        rows = [
            HBox(list(checkboxes.values())[i : i + row_size], width="80px")
            for i in range(0, len(column_names), row_size)
        ]

        checkbox_container = VBox(rows)

        slider = IntSlider(
            min=0, max=len(self.sensor_df) - 1, step=1, value=0, continuous_update=True
        )
        play = Play(
            interval=50,
            value=0,
            min=0,
            max=self.n_rows - 1,
            step=1,
            description="Press play",
            disabled=False,
            repeat=True,
        )
        # Link the play button to the slider
        jslink((play, "value"), (slider, "value"))
        controls = HBox([play, slider])

        out = interactive_output(
            self.plot_los,
            {
                "run": slider,
                "print_str": fixed(None),
                "interacted": fixed(True),
                **checkboxes,
            },
        )

        # Display the combined controls and the output
        display(VBox([controls, checkbox_container, out]))
        return self.output

    def interpret_state_nc(self, state_nc_fn, z_lim=(-0.7, 0.7)):
        with h5py.File(state_nc_fn, "r") as nc_file:
            state_coords = np.array((nc_file["Redge"][:], nc_file["Zedge"][:]))
            state_coords = state_coords[:, 1:]

            states = nc_file["data"][:]
            valid_z = (state_coords[1] >= z_lim[0]) & (state_coords[1] <= z_lim[1])
            state_coords = state_coords[0], state_coords[1][valid_z]
            states = states[:, valid_z, :]
        return states, state_coords


class InteractiveHistogram:
    colors = [
        "#16D5C2",  # Keppel
        "#16425B",  # Indigo
        "#EBF38B",  # Key Lime
        "#0000000",  # Black
    ]

    def __init__(
        self, df, title="EIG Distribution of all Design Sets", n_bins=20, bar_color=None
    ):
        self.df = self.interpret_df(df)
        self.title = title
        self.n_bins = n_bins
        if bar_color is None:
            bar_color = self.colors[1]
        self.bar_color = bar_color
        self.fig = None
        self.plot()

    def interpret_df(self, df):

        # Sample DataFrame with continuous values
        return pd.DataFrame(
            {"Sets": df.keys(), "value": [df[k]["mean_score"] for k in df.keys()]}
        )

    def process_data(self):
        # Define bins
        bin_size = (self.df["value"].max() - self.df["value"].min()) / self.n_bins
        bin_edges = np.arange(
            self.df["value"].min(), self.df["value"].max() + bin_size, bin_size
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Assign values to bins
        self.df["EIG"] = pd.cut(self.df["value"], bins=bin_edges, right=False)

        # Group by bins and concatenate names
        df_grouped = (
            self.df.groupby("EIG")
            .agg({"value": "count", "Sets": lambda x: ", ".join(x)})
            .reset_index()
        )

        df_grouped["EIG"] = df_grouped["EIG"].astype(
            str
        )  # Convert bins to string for display
        df_grouped["Frequency"] = df_grouped["value"]

        # Create hover text
        hover_texts = [
            f"EIG range: [{round(bin_edges[i], 2)}, {round(bin_edges[i+1], 2)})<br>Sets: {row_name}"
            for i, row_name in enumerate(df_grouped["Sets"].values)
        ]

        return (
            bin_centers,
            df_grouped["Frequency"].values,
            hover_texts,
            np.diff(bin_edges),
        )

    def plot(self):
        bin_centers, bin_counts, hover_texts, bin_widths = self.process_data()

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=bin_centers,  # Continuous x-axis
                y=bin_counts,
                hoverinfo="text",
                hovertext=hover_texts,
                width=bin_widths,  # Bin widths
                marker=dict(color=self.bar_color, line=dict(width=1, color="black")),
            )
        )

        # Update layout
        fig.update_layout(
            title=self.title,
            xaxis_title="Expected Information Gain (higher is better)",
            yaxis_title="Frequency",
            xaxis=dict(
                tickmode="linear",
                dtick=0.1,  # Adjust this value to set the desired tick frequency
            ),
            bargap=0,  # No gaps between bins
        )

        self.fig = fig
        fig.show()
