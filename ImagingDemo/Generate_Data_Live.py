import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray
from indica.models.plasma import Plasma
from indica.defaults.load_defaults import load_default_objects
from indica.operators import tomo_1D
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_2d_map
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_parameter
from indica.operators.tomo_asymmetry import InvertPoloidalAsymmetry
from indica.utilities import set_axis_sci
from indica.models.plasma import PlasmaProfiler
from indica.profilers.profiler_gauss import initialise_gauss_profilers
from typing import Callable

from indica.defaults.load_defaults import load_default_objects
from indica.models import BolometerCamera, SXRcamera, BremsstrahlungDiode
import pickle
from indica.defaults.load_defaults import get_filename_default_objects
from indica.converters import LineOfSightTransform
from typing import Tuple
import csv
from shapely import box, LineString, normalize, Polygon, intersection
import math
import random
import imageio
import os
from netCDF4 import Dataset
from ImagingDependencies import (
    example_plasma,
    return_transform,
    return_readings,
    find_intersection,
)


Tnum = 1001  # arabian nights
Nchannels = 25  # number of potential lines of sight
los_starts = [(0.65, 0.55)]  # potential imaging system loc
thetas = [0.7]  # spread of the channels
diagnosticTypes = [BolometerCamera]  # type of diagnostics to choose from


# define the plasma to be used
impurity = "ar"
equilibrium = load_default_objects("st40", "equilibrium")
plasma = example_plasma(
    machine="st40",
    pulse=None,
    tstart=0.06999999999,
    tend=0.07,
    dt=(0.07 - 0.06999999999) / Tnum,
    main_ion="h",
    impurities=(impurity,),
    load_from_pkl=False,
)
plasma.set_equilibrium(equilibrium)
rho_2d = plasma.equilibrium.rho.interp(t=plasma.t)
el_density_2d = plasma.electron_density.interp(rho_poloidal=rho_2d)
ion_density_2d = plasma.ion_density.interp(rho_poloidal=rho_2d)
imp_density_2d = ion_density_2d.sel(element=impurity).drop_vars("element")
lz_tot_2d = (
    plasma.lz_tot[impurity]
    .sum("ion_charge")
    .interp(rho_poloidal=imp_density_2d.rho_poloidal)
)
radiation = el_density_2d * imp_density_2d * lz_tot_2d

# define output data to be saved for every plasma state
dataout = []
headers = []
frames = []
temp_dir = "gif_images//"
if not os.path.isdir(temp_dir):
    os.mkdir(temp_dir)
Rrads = 0
Zrads = 0
Rads = []
Zeffs = []


# iterate through plasma states, saving and plotting relevant information
for i in range(0, len(plasma.t)):
    Zeffs.append(np.array(plasma.zeff.sel(t=plasma.t[i]).sum("element")))
    # print(Zeff)
    # print(np.sum(np.array(plasma.electron_density.sel(t=plasma.t[i]))))
    ion_density_1d = np.sum(np.array(plasma.ion_density.sel(t=plasma.t[i])), axis=0)
    fig, axs = plt.subplots(1, 1)
    radplot = radiation.sel(t=plasma.t[i]).plot(cmap="plasma")
    mesh = radplot.get_coordinates()
    X = mesh[:, :, 0]
    Y = mesh[:, :, 1]
    Rrads = X
    Zrads = Y
    Z = radplot.get_array()
    Rads.append(Z)
    axs.set_xlim([0.15, 0.95])
    axs.set_ylim([-0.7, 0.7])
    axs.set_aspect("equal")
    frame_path = "gif_images//" + str(i) + ".png"
    # save an image every 30th state
    if i % 50 == 0:
        plt.tight_layout()
        plt.savefig(frame_path, bbox_inches="tight")
        frames.append(imageio.v3.imread(frame_path))
    plt.close()

output_gif = "animated_plot.gif"
imageio.mimsave(
    "Images//" + output_gif, frames, duration=0.2
)  # Adjust duration as needed


# Cleanup: remove temporary images
for file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, file))
os.rmdir(temp_dir)


fancolors = ["#B63B34", "#333333", "#DA3F40", "#1D1D1D", "#F09236", "#000000"]
fan_num = 0
colornum = 0
xlinestarts = []
xlineends = []
ylinestarts = []
ylineends = []
for los_start in los_starts:
    for theta in thetas:
        # create the and plot geometry of the camera sensors
        transform = return_transform(
            nchannels=Nchannels, los_start=los_start, theta=theta
        )
        transform.plot()
        plt.show()
        fig, axs = plt.subplots(1, 1)
        for k in range(len(transform.origin_x)):
            color = "grey"
            opacity = 0.4
            if k == 5:
                color = "#479DDD"
                opacity = 1
            xline = [
                transform.origin_x[k],
                transform.origin_x[k] + 10 * transform.direction_x[k],
            ]
            yline = [
                transform.origin_z[k],
                transform.origin_z[k] + 10 * transform.direction_z[k],
            ]
            Xrect = [0.15, 0.95, 0.95, 0.15, 0.15]
            Yrect = [-0.7, -0.7, 0.7, 0.7, -0.7]
            # determine the end locations of the lines of sight
            xline[1], yline[1] = find_intersection(xline, yline, Xrect, Yrect)[1]

            plt.plot(
                xline,
                yline,
                color="#7DB928",
                linewidth=1,
                alpha=1,
            )
            xlinestarts.append(xline[0])
            xlineends.append(xline[1])
            ylinestarts.append(yline[0])
            ylineends.append(yline[1])
        for diagnosticType in diagnosticTypes:
            colornum += 1
            # plt.plot([originx,directionx],[originz,directionz])
            data_config = return_readings(
                plot=0,
                plasma=plasma,
                transform=transform,
                diagnosticType=diagnosticType,
            )[-1]["brightness"].squeeze()
            for k in range(len(data_config[0])):
                dataout.append(np.array(data_config[:, k]))
                name = "LOS_" + str(fan_num) + "_" + str(k)
                headers.append(name)
            fan_num += 1
            axs.set_xlim([0.15, 0.95])
            axs.set_ylim([-0.7, 0.7])
            axs.set_aspect("equal")
            plt.savefig("Images//" + str(fan_num) + ".png", transparent=True)
            plt.show()
outFolder = "Data//Output//"

# Export plasma state information into a netCDF
filename = outFolder + "LOS_state_handsOn.nc"
dataset = Dataset(filename, "w", format="NETCDF4")
dataset.createDimension("R", len(Rrads[0]) - 1)
dataset.createDimension("Z", len(Zrads[:, 0]) - 1)
dataset.createDimension("state", len(Rads))
dataset.createDimension("Redgedim", len(Rrads[0]))
dataset.createDimension("Zedgedim", len(Rrads[0]))
dataset.createDimension("Rhodim", len(plasma.rho))
x_var = dataset.createVariable("Redge", "f4", ("Redgedim",))
y_var = dataset.createVariable("Zedge", "f4", ("Zedgedim",))
Rho_1d = dataset.createVariable("Rho", "f4", ("Rhodim",))
ZeffExport = dataset.createVariable(
    "Zeff",
    "f4",
    (
        "state",
        "Rhodim",
    ),
)

# Assign coordinate values
x_var[:] = Rrads[0]
y_var[:] = Zrads[:, 0]
Rho_1d[:] = np.array(plasma.rho)
ZeffExport[:] = np.array(Zeffs)

# Create a variable to store the array
var = dataset.createVariable(
    "data", "f4", ("state", "R", "Z")
)  # "f4" means 32-bit float

# Store the array in the variable
var[:, :, :] = Rads

# Add metadata (optional)
dataset.description = "2D array exported to NetCDF"
var.units = "Wm^-3"
# Close the dataset
print(dataset)
dataset.close()
print(f"Saved to {filename}")


# export sensor geometries into a CSV
dataout = np.transpose(dataout)
data_geometry = np.transpose(
    np.array([headers, xlinestarts, xlineends, ylinestarts, ylineends])
)
filename = outFolder + "LOS_geometry_handsOn.csv"
headers_geometry = ["LOS", "R start (m)", "R end (m)", "Z start", "Z end"]
# Writing to CSV
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(headers_geometry)

    # Write the data rows
    writer.writerows(data_geometry)

print(f"Data successfully exported to {filename}")


# File name to save the CSV
filename = outFolder + "LOS_demo.csv"

# Writing to CSV
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(headers)

    # Write the data rows
    writer.writerows(dataout)

print(f"Data successfully exported to {filename}")
