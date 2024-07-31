#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: David Leutwyler, Marina Dütsch, Mathias Hauser
# Date: March 2015

import numpy as np
import sys


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# -----------------------------------------------------------------------------


def readsim(filename, varnames):
    """ Load data from a ``.npz`` file.

    Parameters
    ----------
    filename : str
        Path to the ``.npz`` file.
    varnames : sequence[str]
        The variables to retrieve.

    Returns
    -------
    list[np.ndarray] :
        List of variables retrieved from ``filename``.
    """
    if isinstance(varnames, str):
        varnames = [varnames]

    # variables to load
    variables = (
        "u00",
        "th00",
        "thl",
        "dx",
        "nz",
        "nx",
        "time",
        "topomx",
        "topowd",
        "height",
        "x",
    )

    # Display accumulated precipitation as well
    if "specific_rain_water_content" in varnames:
        varnames.append("accumulated_precipitation")

    # allows to access var.xp
    var = dotdict()
    # with automatically closes the file
    with np.load(filename) as data:
        for variable in variables:
            var[variable] = data[variable]

        for varname in varnames:
            try:
                var[varname] = data[varname]
            except:
                sys.exit("Variable not in NetCDF File or wrong timestep passed")

    if "specific_rain_water_content" in varnames:
        varnames.remove("accumulated_precipitation")

    # convert
    var.dx = var.dx / 1000.0
    var.topomx = var.topomx / 1000.0
    var.topowd = var.topowd / 1000.0
    var.zp = var.height / 1000.0

    var.xp = np.zeros(shape=var.zp.shape[-2:])
    var.xp[:, :] = var.x[np.newaxis, :]  # Add an Axis

    # Create Topography
    var.topo = np.zeros(var.nx)
    x = np.arange(var.nx, dtype="float32")
    x0 = (var.nx - 1) / 2.0 + 1
    x = (x + 1 - x0) * var.dx
    toponf = var.topomx * np.exp(-(x / float(var.topowd)) ** 2)
    var.topo[1:-1] = toponf[1:-1] + 0.25 * (
        toponf[0:-2] - 2.0 * toponf[1:-1] + toponf[2:]
    )

    # calculate theta levels
    var.dth = var.thl / var.nz
    theta1d = np.arange(var.nz) * var.dth + var.th00 + var.dth / 2.0
    var.theta = np.zeros(shape=var.zp.shape[-2:])
    var.theta[:, :] = theta1d[:, np.newaxis]

    return var
