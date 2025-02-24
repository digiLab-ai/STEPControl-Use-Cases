from ipywidgets import IntSlider, Output, interact, SelectMultiple, fixed
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import h5py
import pandas as pd


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
        geometry_df=None,
        plasma_nc_filename="Imaging_states_handsOn.nc",
    ):
        self.sensor_df = sensor_df
        if geometry_df is None:
            geometry_df = pd.read_csv("Imaging_geometry_handsOn.csv")
        self.geometry_df = geometry_df
        self.plasma_states, self.plasma_coords, self.z_eff = self.interpret_state_nc(
            plasma_nc_filename
        )

        self.values = self.sensor_df.values
        self.norm = plt.Normalize(self.values.min(), self.values.max())
        self.cmap = plt.cm.viridis

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

    def plot_los(self, run, selected_lines=None, print_str=None):
        selected_inds = [self.sensor_df.columns.get_loc(col) for col in selected_lines]
        num_lines = len(selected_inds)
        self.colours = self.cmap(self.norm(self.values[run]))

        with self.output:
            self.output.clear_output(wait=True)

            # Create figure with GridSpec to control layout
            fig, axs = plt.subplots(
                1, 3, figsize=(10, 6), gridspec_kw={"width_ratios": [0.05, 1, 0.05]}
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
            label_indices = [0, num_lines // 2, num_lines - 1]
            for i in [selected_inds[i] for i in label_indices]:
                ax.text(
                    self.end_R[i] + 0.05 * (self.end_R[i] - self.origin_R[i]),
                    self.end_z[i] + 0.05 * (self.end_z[i] - self.origin_R[i]),
                    self.sensor_df.columns[i],
                    fontsize=8,
                    ha="right",
                    va="center",
                )

            # Add Z eff
            ax.text(
                -0.6,
                0.6,
                "$\\langle Z_{\\mathrm{eff.}} \\rangle$ = " + f"{self.z_eff[run]:.1f}",
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
            cbar.set_label("Sensor (XRCS) Measurement")
            cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(5, 5))

            # Add plasma state colorbar
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
            sm.set_array([])
            cbar = plt.colorbar(im, cax=cax_left, orientation="vertical")
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.set_label_position("left")
            cbar.set_label("Plasma State")
            cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(5, 5))

            ax.set_xlim(-1.0, 1)
            ax.set_ylim(-0.8, 1.0)

            ax_right.set_ylabel("Height (m)")
            ax_right.set_ylim(
                ax.get_ylim()
            )  # Ensure the right axis matches the main plot's y-range
            ax_right.yaxis.set_label_position("right")
            ax_right.yaxis.set_ticks_position("right")

            ax.set_aspect("equal")
            ax.set_xlabel("Major Radius (m)")
            ax.set_ylabel("Height (m)")
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
        interact(
            self.plot_los,
            run=slider,
            selected_lines=fixed(selected_lines),
            print_str=fixed(print_str),
        )
        display(self.output)

    def display_interactive_plot(self):
        column_names = list(self.sensor_df.columns)
        multi_select = SelectMultiple(
            options=column_names,
            description="Click while holding shift to select 3 lines of sight:",
            rows=len(column_names) // 5,
            value=column_names[:3],
        )
        slider = IntSlider(
            min=0, max=len(self.sensor_df) - 1, step=1, value=0, continuous_update=True
        )
        interact(
            self.plot_los,
            run=slider,
            selected_lines=multi_select,
            print_str=fixed(None),
        )
        display(self.output)

    def interpret_state_nc(self, state_nc_fn, z_lim=(-0.7, 0.7)):
        with h5py.File(state_nc_fn, "r") as nc_file:
            state_coords = np.array((nc_file["Redge"][:], nc_file["Zedge"][:]))
            state_coords = state_coords[:, 1:]

            states = nc_file["data"][:]
            valid_z = (state_coords[1] >= z_lim[0]) & (state_coords[1] <= z_lim[1])
            state_coords = state_coords[0], state_coords[1][valid_z]
            states = states[:, valid_z, :]
            z_eff = nc_file["Zeff"][:]
            rho = nc_file["Rho"][:]
            ave_z_eff = self.calc_expected_z_eff(rho, z_eff)
        return states, state_coords, ave_z_eff

    def calc_expected_z_eff(self, rho, z_eff):
        # return np.trapezoid(z_eff, rho, axis=1) / np.trapezoid(np.ones_like(rho), rho)
        return np.trapz(z_eff, rho, axis=1) / np.trapz(np.ones_like(rho), rho)
