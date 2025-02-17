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


def find_intersection(xline, yline, Xrect, Yrect):
    """
    Finds the intersection points between a line and a rectangle.

    Parameters:
    - xline, yline: Coordinates of  line
    - Xrect: List of X coordinates of the rectangle's vertices
    - Yrect: List of Y coordinates of the rectangle's vertices

    Returns:
    - List of intersection points (tuples)
    """

    # Define the line segment
    line = LineString([(xline[0], yline[0]), (xline[1], yline[1])])

    # Define the rectangle as a polygon
    rect = Polygon(zip(Xrect, Yrect))

    # Find intersection
    intersection = line.intersection(rect)

    # Extract intersection points
    if intersection.is_empty:
        return []  # No intersection
    elif intersection.geom_type == "Point":
        return [(intersection.x, intersection.y)]  # Single intersection
    elif intersection.geom_type == "MultiPoint":
        return [(pt.x, pt.y) for pt in intersection.geoms]  # Multiple intersections
    elif intersection.geom_type == "LineString":
        return list(intersection.coords)  # If the line overlaps an edge
    else:
        return []


def example_plasma(
    machine: str = "st40",
    pulse: int = None,
    tstart=0.02,
    tend=0.1,
    dt=0.01,
    main_ion="h",
    impurities: Tuple[str, ...] = ("c", "ar", "he"),
    load_from_pkl: bool = False,
    **kwargs,
):

    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=main_ion,
        impurities=impurities,
        **kwargs,
    )
    plasma.build_atomic_data()
    profile_names = [
        "electron_density",
        "ion_temperature",
        "toroidal_rotation",
        "electron_temperature",
    ]

    for impurity in impurities:
        profile_names.append("impurity_density:" + impurity)
    profilers = initialise_gauss_profilers(
        plasma.rho,
        profile_names=profile_names,
    )
    plasma_profiler = PlasmaProfiler(
        plasma=plasma,
        profilers=profilers,
    )

    # Assign profiles to time-points
    nt = len(plasma.t)
    ne_peaking = np.linspace(1, 1, nt)
    te_peaking = np.linspace(1, 1, nt)
    _y0 = plasma_profiler.profilers["toroidal_rotation"].y0
    vrot0 = np.linspace(
        _y0 * 1.1,
        _y0 * 1.1,
        nt,
    )
    vrot_peaking = np.linspace(1, 1, nt)

    _y0 = plasma_profiler.profilers["ion_temperature"].y0
    ti0 = np.linspace(_y0 * 1.1, _y0 * 1.1, nt)
    print(plasma_profiler.profilers)
    _y0 = plasma_profiler.profilers[f"impurity_density:{impurities[0]}"].y0
    nimp_y0 = _y0 * 2 * np.array(random.sample(list(np.linspace(1, 2, nt * 10)), nt))
    nimp_peaking = np.array(random.sample(list(np.linspace(1, 2, nt * 10)), nt))
    nimp_wcenter = np.array(random.sample(list(np.linspace(0.4, 0.1, nt * 10)), nt))
    for i, t in enumerate(plasma.t):
        parameters = {
            "electron_temperature.peaking": te_peaking[i],
            "ion_temperature.peaking": te_peaking[i],
            "ion_temperature.y0": ti0[i],
            "toroidal_rotation.peaking": vrot_peaking[i],
            "toroidal_rotation.y0": vrot0[i],
            "electron_density.peaking": ne_peaking[i],
            "impurity_density:ar.peaking": nimp_peaking[i],
            "impurity_density:ar.y0": nimp_y0[i],
            "impurity_density:ar.wcenter": nimp_wcenter[i],
        }

        plasma_profiler(parameters=parameters, t=t)

    return plasma


def return_transform(los_start=(0.3, 0.6), nchannels=10, theta=0.1):
    los_end = np.full((nchannels, 3), 0.0)

    # keptsensors = [11, 12, 13]
    plasma_center = (0.5, 0.0)
    central_vector = np.array(plasma_center) - np.array(los_start)
    theta_seg = theta / (nchannels)
    for i in range(nchannels):
        theta_num = i - nchannels / 2 + 0.5
        channel_vectorx = central_vector[0] * math.cos(
            theta_seg * theta_num
        ) - central_vector[1] * math.sin(theta_seg * theta_num)
        channel_vectory = central_vector[0] * math.sin(
            theta_seg * theta_num
        ) + central_vector[1] * math.cos(theta_seg * theta_num)
        channelvector = [channel_vectorx, channel_vectory]
        central_end = channelvector + np.array(los_start)
        los_end[i, 0] = central_end[0]
        los_end[i, 1] = 0.0
        los_end[i, 2] = central_end[1]
    los_start = np.array([[los_start[0], 0, los_start[1]]] * los_end.shape[0])
    origin = los_start
    direction = los_end - los_start
    keptsensors = range(nchannels)
    transform = LineOfSightTransform(
        origin[0:nchannels, 0][keptsensors],
        origin[0:nchannels, 1][keptsensors],
        origin[0:nchannels, 2][keptsensors],
        direction[0:nchannels, 0][keptsensors],
        direction[0:nchannels, 1][keptsensors],
        direction[0:nchannels, 2][keptsensors],
        name="xrcs",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        passes=1,
    )
    return transform


def return_readings(
    plot=False,
    plasma=1,
    instrument="blom_xy1",
    transform=1,
    diagnosticType=BolometerCamera,
):
    equilibrium = load_default_objects("st40", "equilibrium")
    model = diagnosticType
    transform.set_equilibrium(equilibrium)
    model = model(instrument)
    model.set_transform(transform)
    model.set_plasma(plasma)

    bckc = model(sum_beamlets=False)

    if plot and hasattr(model, "plot"):
        plt.ioff()
        model.plot()
        plt.show()
    return plasma, model, bckc
