# -*- coding: utf-8 -*-
"""


***************************************************************
* 2-dimensional isentropic model                              *
* Christoph Schaer, Spring 2000                               *
* Several extensions, Juerg Schmidli, 2005                    *
* Converted to matlab, David Masson, 2009                     *
* Subfunction structure, bugfixes, and Kessler scheme added   *
* Wolfgang Langhans, 2009/2010                                *
* 2 moment scheme, bug fixes, vectorizations, addition of     *
* improvements_n by Mathias Hauser, Deniz Ural,                 *
* Maintenance Lukas Papritz, 2012 / 2013                      *
* Maintenance David Leutwyler 2014                            *
* Port of dry model to python 2.7, Martina Beer, 2014         *
* Finish python v1.0, David Leutwyler, Marina Dütsch 2015     *
* Maintenance Roman Brogli 2017                               *
* Ported to Python3, maintenance, Christian Zeman 2018/2019   *
***************************************************************

TODO:
    - Move definitions in own fuction -> remove cirecular dependency
    - Then re-write output using import to get rid of massive if/else trees

 -----------------------------------------------------------
 -------------------- MAIN PROGRAM: SOLVER -----------------
 -----------------------------------------------------------

"""

import numpy as np  # Scientific computing with Python
from time import time as tm  # Benchmarking tools
import sys

# import model functions
from nmwc_model.makesetup import maketopo, makeprofile
from nmwc_model.boundary import periodic, relax
from nmwc_model.prognostics import (
    prog_isendens,
    prog_velocity,
    prog_moisture,
    prog_numdens,
)
from nmwc_model.diagnostics import diag_montgomery, diag_pressure, diag_height
from nmwc_model.diffusion import horizontal_diffusion
from nmwc_model.output import makeoutput, write_output
from nmwc_model.microphysics import kessler, seifert

# import global namelist variables
from nmwc_model.namelist import (
    imoist as imoist_n,
    imicrophys as imicrophys_n,
    irelax as irelax_n,
    idthdt as idthdt_n,
    idbg as idbg_n,
    iprtcfl as iprtcfl_n,
    nts as nts_n,
    dt as dt_n,
    iiniout as iiniout_n,
    nout as nout_n,
    iout as iout_n,
    dx as dx_n,
    nx as nx_n,
    nb as nb_n,
    nz as nz_n,
    nz1 as nz1_n,
    nab as nab_n,
    diff as diff_n,
    diffabs as diffabs_n,
    topotim as topotim_n,
    itime as itime_n,
)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank_p = comm.Get_rank()
rank_size = comm.Get_size()

assert nx_n % rank_size == 0, "Number of grid points_n must be compatible with rank size"


def exchange_borders(data, tag: int):
    """Exchange broders with next rank for 2-dimensional data set `data`. (scattered along first axis)"""
    left_rank = (rank_p - 1) % rank_size
    right_rank = (rank_p + 1) % rank_size

    send_to_right = data[-2*nb_n:-nb_n]
    send_to_left = data[nb_n:2*nb_n]

    new_left_border = np.empty(nb_n)
    new_right_border = np.empty(nb_n)

    comm.Sendrecv(sendbuf=send_to_left, dest=left_rank, sendt_nag=rank_p * 10_000 + 100 * left_rank +
                  tag, recvbuf=new_right_border, source=left_rank, recvtag=left_rank * 10_000 + 100 * rank_p + tag)

    comm.Sendrecv(sendbuf=send_to_right, dest=right_rank, sendt_nag=rank_p * 10_000 + 100 * right_rank +
                  tag, recvbuf=new_left_border, source=left_rank, recvtag=right_rank * 10_000 + 100 * rank_p + tag)


def initialize_gathered_variables(nout: int):
    # region Define zero-filled gathered variables

    # topography
    topo_g = np.zeros((nx_n + 2 * nb_n, 1))

    # height in z-coordinates
    zhtold_g = np.zeros((nx_n + 2 * nb_n, nz1_n))
    zhtnow_g = np.zeros_like(zhtold_g)
    Z_g = np.zeros((nout, nz1_n, nx_n))  # auxilary field for output

    # horizontal velocity
    uold_g = np.zeros((nx_n + 1 + 2 * nb_n, nz_n))
    unow_g = np.zeros_like(uold_g)
    unew_g = np.zeros_like(uold_g)
    U_g = np.zeros((nout, nz_n, nx_n))  # auxilary field for output

    # isentropic density
    sold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    snow_g = np.zeros_like(sold_g)
    snew_g = np.zeros_like(sold_g)
    S_g = np.zeros((nout, nz_n, nx_n))  # auxilary field for output

    # Montgomery potential
    mtg_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    mtgnew_g = np.zeros_like(mtg_g)

    # Exner function
    exn_g = np.zeros((nx_n + 2 * nb_n, nz1_n))

    # pressure
    prs_g = np.zeros((nx_n + 2 * nb_n, nz1_n))

    # output time vector
    T_g = np.arange(1, nout + 1)

    # precipitation
    prec_g = np.zeros(nx_n + 2 * nb_n)
    PREC_g = np.zeros((nout, nx_n))  # auxiliary field for output

    # accumulated precipitation
    tot_prec_g = np.zeros(nx_n + 2 * nb_n)
    TOT_PREC_g = np.zeros((nout, nx_n))  # auxiliary field for output

    # specific humidity
    qvold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    qvnow_g = np.zeros_like(qvold_g)
    qvnew_g = np.zeros_like(qvold_g)
    QV_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # specific cloud water content
    qcold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    qcnow_g = np.zeros_like(qcold_g)
    qcnew_g = np.zeros_like(qcold_g)
    QC_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # specific rain water content
    qrold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    qrnow_g = np.zeros_like(qrold_g)
    qrnew_g = np.zeros_like(qrold_g)
    QR_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # cloud droplet number density
    ncold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    ncnow_g = np.zeros_like(ncold_g)
    ncnew_g = np.zeros_like(ncold_g)
    NC_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # rain-droplet number density
    nrold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    nrnow_g = np.zeros_like(nrold_g)
    nrnew_g = np.zeros_like(nrold_g)
    NR_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # latent heating
    dthetadt_g = np.zeros((nx_n + 2 * nb_n, nz1_n))
    DTHETADT_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # Define fields at lateral boundaries
    # 1 denotes the left boundary
    # 2 denotes the right boundary
    # ----------------------------------------------------------------------------
    # topography
    tbnd1_g = 0.0
    tbnd2_g = 0.0

    # isentropic density
    sbnd1_g = np.zeros(nz_n)
    sbnd2_g = np.zeros(nz_n)

    # horizontal velocity
    ubnd1_g = np.zeros(nz_n)
    ubnd2_g = np.zeros(nz_n)

    # specific humidity
    qvbnd1_g = np.zeros(nz_n)
    qvbnd2_g = np.zeros(nz_n)

    # specific cloud water content
    qcbnd1_g = np.zeros(nz_n)
    qcbnd2_g = np.zeros(nz_n)

    # specific rain water content
    qrbnd1_g = np.zeros(nz_n)
    qrbnd2_g = np.zeros(nz_n)

    # latent heating
    dthetadtbnd1 = np.zeros(nz1_n)
    dthetadtbnd2 = np.zeros(nz1_n)

    # cloud droplet number density
    ncbnd1_g = np.zeros(nz_n)
    ncbnd2_g = np.zeros(nz_n)

    # rain droplet number density
    nrbnd1_g = np.zeros(nz_n)
    nrbnd2_g = np.zeros(nz_n)

    # variables later set by `makeprofile`
    th0_g = np.empty((nz_n + 1))
    exn0_g = np.empty((nz_n + 1))
    prs0_g = np.empty((nz_n + 1))
    z0_g = np.empty((nz_n + 1))
    mtg0_g = np.empty((nz_n))
    s0_g = np.empty((nz_n + 1))
    qv0_g = np.empty((nz_n))
    qc0_g = np.empty((nz_n))
    qr0_g = np.empty((nz_n))
    tau_g = np.empty((nz_n))

    # endregion

    # Set initial conditions
    # -----------------------------------------------------------------------------

    # region Run `makeprofile`
    if idbg_n == 1:
        print("Setting initial conditions ...\n")

    if imoist_n == 0:
        # Dry atmosphere
        th0_g, exn0_g, prs0_g, z0_g, mtg0_g, s0_g, u0_g, sold_g, snow_g, uold_g, unow_g, mtg_g, mtgnew_g = makeprofile(
            sold_g, uold_g, mtg_g, mtgnew_g
        )
    elif imicrophys_n == 0 or imicrophys_n == 1:
        # moist atmosphere with kessler scheme
        th0_g, exn0_g, prs0_g, z0_g, mtg0_g, s0_g, u0_g, sold_g, snow_g, uold_g, unow_g, mtg_g, mtgnew_g, qv0_g, qc0_g, qr0_g, qvold_g, qvnow_g, qcold_g, qcnow_g, qrold_g, qrnow_g = makeprofile(
            sold_g,
            uold_g,
            qvold=qvold_g,
            qvnow=qvnow_g,
            qcold=qcold_g,
            qcnow=qcnow_g,
            qrold=qrold_g,
            qrnow=qrnow_g,
        )
    elif imicrophys_n == 2:
        # moist atmosphere with 2-moment scheme
        th0_g, exn0_g, prs0_g, z0_g, mtg0_g, s0_g, u0_g, sold_g, snow_g, uold_g, unow_g, mtg_g, mtgnew_g, qv0_g, qc0_g, qr0_g, qvold_g, qvnow_g, qcold_g, qcnow_g, qrold_g, qrnow_g, ncold_g, ncnow_g, nrold_g, nrnow_g = makeprofile(
            sold_g,
            uold_g,
            qvold=qvold_g,
            qvnow=qvnow_g,
            qcold=qcold_g,
            qcnow=qcnow_g,
            qrold=qrold_g,
            qrnow=qrnow_g,
            ncold=ncold_g,
            ncnow=ncnow_g,
            nrold=nrold_g,
            nrnow=nrnow_g,
        )

    # endregion

    # region Save boundary values for the lateral boundary relaxation
    if irelax_n == 1:
        if idbg_n == 1:
            print("Saving initial lateral boundary values ...\n")

        sbnd1_g[:] = snow_g[0, :]
        sbnd2_g[:] = snow_g[-1, :]

        ubnd1_g[:] = unow_g[0, :]
        ubnd2_g[:] = unow_g[-1, :]

        if imoist_n == 1:
            qvbnd1_g[:] = qvnow_g[0, :]
            qvbnd2_g[:] = qvnow_g[-1, :]

            qcbnd1_g[:] = qcnow_g[0, :]
            qcbnd2_g[:] = qcnow_g[-1, :]

        if imicrophys_n != 0:
            qrbnd1_g[:] = qrnow_g[0, :]
            qrbnd2_g[:] = qrnow_g[-1, :]

        # 2-moment microphysics scheme
        if imicrophys_n == 2:
            ncbnd1_g[:] = ncnow_g[0, :]
            ncbnd2_g[:] = ncnow_g[-1, :]

            nrbnd1_g[:] = nrnow_g[0, :]
            nrbnd2_g[:] = nrnow_g[-1, :]

        if idthdt_n == 1:
            dthetadtbnd1[:] = dthetadt_g[0, :]
            dthetadtbnd2[:] = dthetadt_g[-1, :]

    # endregion

    # region Make topography
    # ----------------
    topo_g = maketopo(topo_g, nx_n + 2 * nb_n)

    # switch between boundary relaxation / periodic boundary conditions
    # ------------------------------------------------------------------
    if irelax_n == 1:  # boundary relaxation
        if idbg_n == 1:
            print("Relax topography ...\n")

        # save lateral boundary values of topography
        tbnd1_g = topo_g[0]
        tbnd2_g = topo_g[-1]

        # relax topography
        topo_g = relax(topo_g, nx_n, nb_n, tbnd1_g, tbnd2_g)
    else:
        if idbg_n == 1:
            print("Periodic topography ...\n")

        # make topography periodic
        topo_g = periodic(topo_g, nx_n, nb_n)

    # endregion

    # region Heigh-dependent settings

    # calculate geometric height (staggered)
    zhtnow_g = diag_height(
        prs0_g[np.newaxis, :], exn0_g[np.newaxis,
                                      :], zhtnow_g, th0_g, topo_g, 0.0
    )

    # Height-dependent diffusion coefficient
    # --------------------------------------
    tau_g = diff_n * np.ones(nz_n)

    # *** Exercise 3.1 height-dependent diffusion coefficient ***
    tau_g = diff_n + (diffabs_n - diff_n) * \
        np.sin(np.pi / 2 * (np.arange(nz_n) - nz_n + nab_n - 1) / nab_n)**2
    tau_g[0:nz_n-nab_n] = diff_n
    # *** Exercise 3.1 height-dependent diffusion coefficient ***

    # endregion

    # region Output initial fields
    its_out_g = -1  # output index
    if iiniout_n == 1 and imoist_n == 0:
        its_out_g, Z_g, U_g, S_g, T_g = makeoutput(
            unow_g, snow_g, zhtnow_g, its_out_g, 0, Z_g, U_g, S_g, T_g)
    elif iiniout_n == 1 and imoist_n == 1:
        if imicrophys_n == 0 or imicrophys_n == 1:
            if idthdt_n == 0:
                its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g = makeoutput(
                    unow_g,
                    snow_g,
                    zhtnow_g,
                    its_out_g,
                    0,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    qvnow=qvnow_g,
                    qcnow=qcnow_g,
                    qrnow=qrnow_g,
                    tot_prec=tot_prec_g,
                    prec=prec_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    TOT_PREC=TOT_PREC_g,
                    PREC=PREC_g,
                )
            elif idthdt_n == 1:
                its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, DTHETADT_g = makeoutput(
                    unow_g,
                    snow_g,
                    zhtnow_g,
                    its_out_g,
                    0,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    qvnow=qvnow_g,
                    qcnow=qcnow_g,
                    qrnow=qrnow_g,
                    tot_prec=tot_prec_g,
                    prec=prec_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    TOT_PREC=TOT_PREC_g,
                    PREC=PREC_g,
                    dt_nhetadt_n=dthetadt_g,
                    dt_nHETAdt_n=DTHETADT_g,
                )
        elif imicrophys_n == 2:
            if idthdt_n == 0:
                its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, NC_g, NR_g = makeoutput(
                    unow_g,
                    snow_g,
                    zhtnow_g,
                    its_out_g,
                    0,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    qvnow=qvnow_g,
                    qcnow=qcnow_g,
                    qrnow=qrnow_g,
                    tot_prec=tot_prec_g,
                    prec=prec_g,
                    nrnow=nrnow_g,
                    ncnow=ncnow_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    TOT_PREC=TOT_PREC_g,
                    PREC=PREC_g,
                    NC=NC_g,
                    NR=NR_g,
                )
            elif idthdt_n == 1:
                its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, NC_g, NR_g, DTHETADT_g = makeoutput(
                    unow_g,
                    snow_g,
                    zhtnow_g,
                    its_out_g,
                    0,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    qvnow=qvnow_g,
                    qcnow=qcnow_g,
                    qrnow=qrnow_g,
                    tot_prec=tot_prec_g,
                    prec=prec_g,
                    nrnow=nrnow_g,
                    ncnow=ncnow_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    TOT_PREC=TOT_PREC_g,
                    PREC=PREC_g,
                    NC=NC_g,
                    NR=NR_g,
                    dt_nhetadt_n=dthetadt_g,
                    dt_nHETAdt_n=DTHETADT_g,
                )

    # endregion

    # region Return relevant variables
    return (
        sold_g, snow_g, snew_g, S_g,
        uold_g, unow_g, unew_g, U_g,
        qvold_g, qvnow_g, qvnew_g, QV_g,
        qcold_g, qcnow_g, qcnew_g, QC_g,
        qrold_g, qrnow_g, qrnew_g, QR_g,
        ncold_g, ncnow_g, ncnew_g, NC_g,
        nrold_g, nrnow_g, nrnew_g, NR_g,
        dthetadt_g, DTHETADT_g,
        mtg_g, tau_g, prs0_g, prs_g, T_g,
        prec_g, PREC_g, tot_prec_g, TOT_PREC_g,
        topo_g, zhtold_g, zhtnow_g, Z_g, th0_g, dthetadtbnd1, dthetadtbnd2
    )
    # endregion


def main():
    # Print the full precision
    # DL: REMOVE FOR STUDENT VERSION
    np.set_printoptions(threshold=sys.maxsize)

    # Define physical fields
    # -------------------------
    nx_p = nx_n // rank_size

    # region Allocate process-specific variables
    sold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    snow_p = np.empty_like(sold_p)
    snew_p = np.empty_like(sold_p)
    uold_p = np.empty((nx_p + 1 + 2 * nb_n, nz_n))
    unow_p = np.empty_like(uold_p)
    qvold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    qvnow_p = np.empty_like(qvold_p)
    qvnew_p = np.empty_like(qvold_p)
    qcold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    qcnow_p = np.empty_like(qcold_p)
    qcnew_p = np.empty_like(qcold_p)
    qrold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    qrnow_p = np.empty_like(qrold_p)
    qrnew_p = np.empty_like(qrold_p)
    ncold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    ncnow_p = np.empty_like(ncold_p)
    ncnew_p = np.empty_like(ncold_p)
    nrold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    nrnow_p = np.empty_like(nrold_p)
    nrnew_p = np.empty_like(nrold_p)
    mtg_p = np.empty((nx_p + 2 * nb_n, nz_n))
    tau_p = np.empty(nz_n)
    prs0_p = np.zeros(nz_n + 1)
    prs_p = np.empty((nx_p + 2 * nb_n, nz_n))
    topo_p = np.empty((nx_p + 2 * nb_n, 1))
    zhtold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    zhtnow_p = np.empty((nx_p + 2 * nb_n, nz_n))
    th0_p = np.empty((nz_n + 1))
    # endregion

    # increase number of output steps by 1 for initial profile
    nout = nout_n
    if iiniout_n == 1:
        nout += 1

    if rank_p == 0:
        (
            sold_g, snow_g, snew_g, S_g,
            uold_g, unow_g, unew_g, U_g,
            qvold_g, qvnow_g, qvnew_g, QV_g,
            qcold_g, qcnow_g, qcnew_g, QC_g,
            qrold_g, qrnow_g, qrnew_g, QR_g,
            ncold_g, ncnow_g, ncnew_g, NC_g,
            nrold_g, nrnow_g, nrnew_g, NR_g,
            dthetadt_g, DTHETADT_g,
            mtg_g, tau_g, prs0_g, prs_g, T_g,
            prec_g, PREC_g, tot_prec_g, TOT_PREC_g,
            topo_g, zhtold_g, zhtnow_g, Z_g, th0_g, dthetadtbnd1, dthetadtbnd2
        ) = initialize_gathered_variables(nout)

        # region Distribute slices to processes
        for i in range(1, rank_size):
            start_index = i * nx_p
            end_index = (i + 1) * nx_p + 2 * nb_n
            rank_slice = slice(start_index, end_index)
            rank_slice_staggered = slice(start_index, end_index + 1)

            comm.Send(sold_g[rank_slice, :], dest=i, tag=i * 1000 + 0)
            comm.Send(snow_g[rank_slice, :], dest=i, tag=i * 1000 + 1)
            comm.Send(snew_g[rank_slice, :], dest=i, tag=i * 1000 + 2)

            comm.Send(uold_g[rank_slice_staggered, :],
                      dest=i, tag=i * 1000 + 3)
            comm.Send(unow_g[rank_slice_staggered, :],
                      dest=i, tag=i * 1000 + 4)

            comm.Send(qvold_g[rank_slice, :], dest=i, tag=i * 1000 + 5)
            comm.Send(qvnow_g[rank_slice, :], dest=i, tag=i * 1000 + 6)
            comm.Send(qvnew_g[rank_slice, :], dest=i, tag=i * 1000 + 7)

            comm.Send(qcold_g[rank_slice, :], dest=i, tag=i * 1000 + 8)
            comm.Send(qcnow_g[rank_slice, :], dest=i, tag=i * 1000 + 9)
            comm.Send(qcnew_g[rank_slice, :], dest=i, tag=i * 1000 + 10)

            comm.Send(qrold_g[rank_slice, :], dest=i, tag=i * 1000 + 11)
            comm.Send(qrnow_g[rank_slice, :], dest=i, tag=i * 1000 + 12)
            comm.Send(qrnew_g[rank_slice, :], dest=i, tag=i * 1000 + 13)

            if imoist_n == 1 and imicrophys_n == 2:
                comm.Send(ncold_g[rank_slice, :], dest=i, tag=i * 1000 + 14)
                comm.Send(ncnow_g[rank_slice, :], dest=i, tag=i * 1000 + 15)
                comm.Send(ncnew_g[rank_slice, :], dest=i, tag=i * 1000 + 16)

                comm.Send(nrold_g[rank_slice, :], dest=i, tag=i * 1000 + 17)
                comm.Send(nrnow_g[rank_slice, :], dest=i, tag=i * 1000 + 18)
                comm.Send(nrnew_g[rank_slice, :], dest=i, tag=i * 1000 + 19)

            comm.Send(mtg_g[rank_slice, :], dest=i, tag=i * 1000 + 20)
            comm.Send(tau_g, dest=i, tag=i * 1000 + 21)
            comm.Send(prs0_g, dest=i, tag=i * 1000 + 22)
            comm.Send(prs_g[rank_slice, :], dest=i, tag=i * 1000 + 23)
            comm.Send(topo_g[rank_slice, :], dest=i, tag=i * 1000 + 24)
            comm.Send(zhtold_g[rank_slice, :], dest=i, tag=i * 1000 + 25)
            comm.Send(zhtnow_g[rank_slice, :], dest=i, tag=i * 1000 + 26)
            comm.Send(th0_g, dest=i, tag=i * 1000 + 27)
        # endregion

        # region Set process variables for process with rank 0
        start_index = 0
        end_index = nx_p + 2 * nb_n
        rank_slice = slice(start_index, end_index)
        rank_slice_staggered = slice(start_index, end_index + 1)

        sold_p = sold_g[rank_slice, :]
        snow_p = snow_g[rank_slice, :]
        snew_p = snew_g[rank_slice, :]

        uold_p = uold_g[rank_slice_staggered, :]
        unow_p = unow_g[rank_slice_staggered, :]

        qvold_p = qvold_g[rank_slice, :]
        qvnow_p = qvnow_g[rank_slice, :]
        qvnew_p = qvnew_g[rank_slice, :]

        qcold_p = qcold_g[rank_slice, :]
        qcnow_p = qcnow_g[rank_slice, :]
        qcnew_p = qcnew_g[rank_slice, :]

        qrold_p = qrold_g[rank_slice, :]
        qrnow_p = qrnow_g[rank_slice, :]
        qrnew_p = qrnew_g[rank_slice, :]

        if imoist_n == 1 and imicrophys_n == 2:
            ncold_p = ncold_g[rank_slice, :]
            ncnow_p = ncnow_g[rank_slice, :]
            ncnew_p = ncnew_g[rank_slice, :]

            nrold_p = nrold_g[rank_slice, :]
            nrnow_p = nrnow_g[rank_slice, :]
            nrnew_p = nrnew_g[rank_slice, :]

        mtg_p = mtg_g[rank_slice, :]
        tau_p = tau_g
        prs0_p = prs0_g
        prs_p = prs_g[rank_slice, :]
        topo_p = topo_g[rank_slice, :]
        zhtold_p = zhtold_g[rank_slice, :]
        zhtnow_p = zhtnow_g[rank_slice, :]
        th0_p = th0_g
        # endregion
    else:
        # region Receive process-specific variable values
        comm.Recv(sold_p, source=0, tag=rank_p * 1000 + 0)
        comm.Recv(snow_p, source=0, tag=rank_p * 1000 + 1)
        comm.Recv(snew_p, source=0, tag=rank_p * 1000 + 2)

        comm.Recv(uold_p, source=0, tag=rank_p * 1000 + 3)
        comm.Recv(unow_p, source=0, tag=rank_p * 1000 + 4)

        comm.Recv(qvold_p, source=0, tag=rank_p * 1000 + 5)
        comm.Recv(qvnow_p, source=0, tag=rank_p * 1000 + 6)
        comm.Recv(qvnew_p, source=0, tag=rank_p * 1000 + 7)

        comm.Recv(qcold_p, source=0, tag=rank_p * 1000 + 8)
        comm.Recv(qcnow_p, source=0, tag=rank_p * 1000 + 9)
        comm.Recv(qcnew_p, source=0, tag=rank_p * 1000 + 10)

        comm.Recv(qrold_p, source=0, tag=rank_p * 1000 + 11)
        comm.Recv(qrnow_p, source=0, tag=rank_p * 1000 + 12)
        comm.Recv(qrnew_p, source=0, tag=rank_p * 1000 + 13)

        if imoist_n == 1 and imicrophys_n == 2:
            comm.Recv(ncold_p, source=0, tag=rank_p * 1000 + 14)
            comm.Recv(ncnow_p, source=0, tag=rank_p * 1000 + 15)
            comm.Recv(ncnew_p, source=0, tag=rank_p * 1000 + 16)

            comm.Recv(nrold_p, source=0, tag=rank_p * 1000 + 17)
            comm.Recv(nrnow_p, source=0, tag=rank_p * 1000 + 18)
            comm.Recv(nrnew_p, source=0, tag=rank_p * 1000 + 19)

        comm.Recv(mtg_p, source=0, tag=rank_p * 1000 + 20)
        comm.Recv(tau_p, source=0, tag=rank_p * 1000 + 21)
        comm.Recv(prs0_p, source=0, tag=rank_p * 1000 + 22)
        comm.Recv(prs_p, source=0, tag=rank_p * 1000 + 23)
        comm.Recv(topo_p, source=0, tag=rank_p * 1000 + 24)

        comm.Recv(zhtold_p, source=0, tag=i * 1000 + 25)
        comm.Recv(zhtnow_p, source=0, tag=i * 1000 + 26)
        comm.Recv(th0_p, source=0, tag=i * 1000 + 27)
        # endregion

    # ########## TIME LOOP #######################################################
    # ----------------------------------------------------------------------------
    # Loop over all time steps
    # ----------------------------------------------------------------------------

    # region Time loop
    if idbg_n == 1 and rank_p == 0:
        print("Starting time loop ...\n")

    t0_p = tm()
    for its_p in range(1, int(nts_n + 1)):
        # calculate time
        time_p = its_p * dt_n

        if itime_n == 1:
            if idbg_n == 1 or idbg_n == 0:
                print("========================================================\n")
                print("Working on timestep %g; time = %g s; process = %g\n" %
                      (its_p, time_p, rank_p))
                print("========================================================\n")

        # initially increase height of topography only slowly
        topofact_p: float = min(1.0, float(time_p) / topotim_n)

        # Special treatment of first time step
        # -------------------------------------------------------------------------
        if its_p == 1:
            dtdx_p: float = dt_n / dx_n / 2.0
            dthetadt_p = None
            if imoist_n == 1 and idthdt_n == 1:
                # No latent heating for first time-step
                dthetadt_p = np.zeros((nx_p + 2 * nb_n, nz1_n))
            if idbg_n == 1:
                print("Using Euler forward step for 1. step ...\n")
        else:
            dtdx_p: float = dt_n / dx_n

        # *** Exercise 2.1 isentropic mass density ***
        # *** time step for isentropic mass density ***
        snew_p = prog_isendens(sold_p, snow_p, unow_p,
                               dtdx_p, dthetadt=dthetadt_p, nx=nx_p)
        #
        # *** Exercise 2.1 isentropic mass density ***

        # *** Exercise 4.1 / 5.1 moisture ***
        # *** time step for moisture scalars ***
        if imoist_n == 1:
            if idbg_n == 1:
                print("Add function call to prog_moisture")
            qvnew_p, qcnew_p, qrnew_p = prog_moisture(
                unow_p, qvold_p, qcold_p, qrold_p, qvnow_p, qcnow_p, qrnow_p, dtdx_p, dt_nhetadt_n=dthetadt_n, nx_n=nx_p)

            if imicrophys_n == 2:
                ncnew_p, nrnew_p = prog_numdens(
                    unow_p, ncold_p, nrold_p, ncnow_p, nrnow_p, dtdx_p, dt_nhetadt_n=dthetadt_n, nx_n=nx_p)

        #
        # *** Exercise 4.1 / 5.1 moisture scalars ***

        # *** Exercise 2.1 velocity ***
        # *** time step for momentum ***
        #

        # *** edit here ***
        unew_p = prog_velocity(uold_p, unow_p, mtg_p,
                               dtdx_p, dt_nhetadt_n=dthetadt_n, nx_n=nx_p)
        #
        # *** Exercise 2.1 velocity ***

        # exchange boundaries if periodic
        # -------------------------------------------------------------------------
        if irelax_n == 0:
            if rank_p == 0:
                snew_p[0:nb_n, :] = comm.Sendrecv(sendbuf=snew_p[nb_n:2*nb_n, :], dest=rank_size - 1,
                                                  sendt_nag=11, recvbuf=None, source=rank_size - 1, recvtag=22)
                unew_p[0:nb_n, :] = comm.Sendrecv(sendbuf=unew_p[nb_n:2*nb_n, :], dest=rank_size - 1,
                                                  sendt_nag=111, recvbuf=None, source=rank_size - 1, recvtag=222)
            elif rank_p == rank_size - 1:
                snew_p[-nb_n:, :] = comm.Sendrecv(sendbuf=snew_p[-2*nb_n:-nb_n],
                                                  dest=0, sendt_nag=22, recvbuf=None, source=0, recvtag=11)
                unew_p[-nb_n:, :] = comm.Sendrecv(sendbuf=unew_p[-2*nb_n:-nb_n],
                                                  dest=0, sendt_nag=222, recvbuf=None, source=0, recvtag=111)

            if imoist_n == 1:
                pass
                # qvnew_p = periodic(qvnew_p, nx_n, nb_n) # TODO
                # qcnew_p = periodic(qcnew, nx_n, nb_n) # TODO
                # qrnew_p = periodic(qrnew, nx_n, nb_n) # TODO

            # 2-moment scheme
            if imoist_n == 1 and imicrophys_n == 2:
                pass
                # ncnew = periodic(ncnew, nx_n, nb_n) # TODO
                # nrnew = periodic(nrnew, nx_n, nb_n) # TODO

        # relaxation of prognostic fields
        # -------------------------------------------------------------------------
        if irelax_n == 1:
            if idbg_n == 1:
                print("Relaxing prognostic fields ...\n")
            # snew = relax(snew, nx_n, nb_n, sbnd1, sbnd2) # TODO
            # unew = relax(unew, nx_n + 1, nb_n, ubnd1, ubnd2) # TODO
            if imoist_n == 1:
                pass
                # qvnew = relax(qvnew, nx_n, nb_n, qvbnd1, qvbnd2) # TODO
                # qcnew = relax(qcnew, nx_n, nb_n, qcbnd1, qcbnd2) # TODO
                # qrnew = relax(qrnew, nx_n, nb_n, qrbnd1, qrbnd2) # TODO

            # 2-moment scheme
            if imoist_n == 1 and imicrophys_n == 2:
                pass
                # ncnew = relax(ncnew, nx_n, nb_n, ncbnd1, ncbnd2) # TODO
                # nrnew = relax(nrnew, nx_n, nb_n, nrbnd1, nrbnd2) # TODO

        # Diffusion and gravity wave absorber
        # ------------------------------------

        if imoist_n == 0:
            [unew_p, snew_p] = horizontal_diffusion(
                tau_p, unew_p, snew_p, nx_n=nx_p)
        else:
            if imicrophys_n == 2:
                [unew_p, snew_p, qvnew_p, qcnew_p, qrnew_p, ncnew_p, nrnew_p] = horizontal_diffusion(
                    tau_p,
                    unew_p,
                    snew_p,
                    qvnew=qvnew_p,
                    qcnew=qcnew_p,
                    qrnew=qrnew_p,
                    ncnew=ncnew_p,
                    nrnew=nrnew_p,
                )
            else:
                [unew_p, snew_p, qvnew_p, qcnew_p, qrnew_p] = horizontal_diffusion(
                    tau_p, unew_p, snew_p, qvnew=qvnew_p, qcnew=qcnew_p, qrnew=qrnew_p, nx_n=nx_p
                )

        # *** Exercise 2.2 Diagnostic computation of pressure ***
        # *** Diagnostic computation of pressure ***
        #

        # *** edit here ***
        prs_p = diag_pressure(prs0_p, prs_p, snew_p)
        #
        # *** Exercise 2.2 Diagnostic computation of pressure ***

        # *** Exercise 2.2 Diagnostic computation of Montgomery ***
        # *** Calculate Exner function and Montgomery potential ***
        #

        # *** edit here ***
        exn_p, mtg_p = diag_montgomery(prs_p, mtg_p, th0_p, topo_p, topofact_p)
        #
        # *** Exercise 2.2 Diagnostic computation of Montgomery ***

        # Calculation of geometric height (staggered)
        # needed for output and microphysics schemes
        # ---------------------------------
        zhtold_p[...] = zhtnow_p[...]
        zhtnow_p = diag_height(prs_p, exn_p, zhtnow_p,
                               th0_p, topo_p, topofact_p)

        if imoist_n == 1:
            # *** Exercise 4.1 Moisture ***
            # *** Clipping of negative values ***
            # *** edit here ***
            #

            if idbg_n == 1:
                print("Implement moisture clipping")
            qvnew_p[qvnew_p < 0] = 0
            qcnew_p[qcnew_p < 0] = 0
            qrnew_p[qrnew_p < 0] = 0

            if imicrophys_n == 2:
                ncnew_p[ncnew_p < 0] = 0
                nrnew_p[nrnew_p < 0] = 0

            #
            # *** Exercise 4.1 Moisture ***

        if imoist_n == 1 and imicrophys_n == 1:
            # *** Exercise 4.2 Kessler ***
            # *** Kessler scheme ***
            # *** edit here ***
            #

            if idbg_n == 1:
                print("Add function call to Kessler microphysics")
            [lheat, qvnew_p, qcnew_p, qrnew_p, prec_p, prec_tot] = kessler(
                snew_p, qvnew_p, qcnew_p, qrnew_p, prs_p, exn_p, zhtnow_p, th0_p, prec, tot_prec)

            #
            # *** Exercise 4.2 Kessler ***
        elif imoist_n == 1 and imicrophys_n == 2:
            # *** Exercise 5.1 Two Moment Scheme ***
            # *** Two Moment Scheme ***
            # *** edit here ***
            #

            if idbg_n == 1:
                print("Add function call to two moment microphysics")
            [lheat, qvnew_p, qcnew_p, qrnew_p, tot_prec, prec, ncnew_p, nrnew_p] = seifert(
                unew_p, th0_p, prs_p, snew_p, qvnew_p, qcnew_p, qrnew_p, exn_p, zhtold_p, zhtnow_p, tot_prec, prec, ncnew_p, nrnew_p, dthetadt_n)
            #
            # *** Exercise 5.1 Two Moment Scheme ***

        if imoist_n == 1 and imicrophys_n > 0:
            if idthdt_n == 1:
                # Stagger lheat to model levels and compute tendency
                k = np.arange(1, nz_n)
                if imicrophys_n == 1:
                    dthetadt_n[:, k] = topofact_p * 0.5 * \
                        (lheat[:, k - 1] + lheat[:, k]) / dt_n
                else:
                    dthetadt_n[:, k] = topofact_p * 0.5 * \
                        (lheat[:, k - 1] + lheat[:, k]) / (2.0 * dt_n)

                # force dt_nhetadt_n to zeros at the bottom and at the top
                dthetadt_n[:, 0] = 0.0
                dthetadt_n[:, -1] = 0.0

                # periodic lateral boundary conditions
                # ----------------------------
                if irelax_n == 0:
                    dthetadt_n = periodic(dthetadt_n, nx_n, nb_n)
                else:
                    # Relax latent heat fields
                    # ----------------------------
                    dthetadt_n = relax(dthetadt_n, nx_n, nb_n,
                                       dthetadtbnd1, dthetadtbnd2)
            else:
                dthetadt_n = np.zeros((nx_n + 2 * nb_n, nz1_n))

        if idbg_n == 1:
            print("Preparing next time step ...\n")

        # region Exchange isentropic mass density and velocity
        if imicrophys_n == 2:
            ncold_p = ncnow_p
            ncnow_p = ncnew_p

            nrold_p = nrnow_p
            nrnow_p = nrnew_p

        sold_p = snow_p
        snow_p = snew_p

        uold_p = unow_p
        unow_p = unew_p

        if imoist_n == 1:
            qvold_p = qvnow_p
            qvnow_p = qvnew_p

            qcold_p = qcnow_p
            qcnow_p = qcnew_p

            qrold_p = qrnow_p
            qrnow_p = qrnew_p
            if idbg_n == 1:
                print("exchange moisture variables")

            if imicrophys_n == 2:
                if idbg_n == 1:
                    print("exchange number densitiy variables")
        # endregion

        # region Check maximum cfl criterion
        # ---------------------------------
        if iprtcfl_n == 1:
            u_max = np.amax(np.abs(unow_p))
            cfl_max = u_max * dtdx_p
            print("============================================================\n")
            print("CFL MAX: %g U MAX: %g m/s \n" % (cfl_max, u_max))
            if cfl_max > 1:
                print("!!! WARNING: CFL larger than 1 !!!\n")
            elif np.isnan(cfl_max):
                print("!!! MODEL ABORT: NaN values !!!\n")
            print("============================================================\n")
        # endregion

        # region Output every 'iout_n'-th time step
        # ---------------------------------
        if np.mod(its_p, iout_n) == 0:
            # TODO: gather unow_g, snow_g, zhtnow_g
            if imoist_n == 0:
                if rank_p == 0:
                    its_out, Z_g, U_g, S_g, T_g = makeoutput(
                        unow_g, snow_g, zhtnow_g, its_out, its_p, Z_g, U_g, S_g, T_g
                    )
            elif imoist_n == 1:
                if imicrophys_n == 0 or imicrophys_n == 1:
                    if idthdt_n == 0:
                        # TODO: gather qvnow_g, qcnow_g, qrnow_g, tot_prec_g, prec_g
                        if rank_p == 0:
                            its_out, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g = makeoutput(
                                unow_g,
                                snow_g,
                                zhtnow_g,
                                its_out,
                                its_p,
                                Z_g,
                                U_g,
                                S_g,
                                T_g,
                                qvnow=qvnow_g,
                                qcnow=qcnow_g,
                                qrnow=qrnow_g,
                                tot_prec=tot_prec_g,
                                prec=prec_g,
                                QV=QV_g,
                                QC=QC_g,
                                QR=QR_g,
                                TOT_PREC=TOT_PREC_g,
                                PREC=PREC_g,
                            )
                    elif idthdt_n == 1:
                        # TODO: gather qvnow_g, qcnow_g, qrnow_g, tot_prec_g, prec_g, dthetadt_g
                        if rank_p == 0:
                            its_out, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, DTHETADT_g = makeoutput(
                                unow_g,
                                snow_g,
                                zhtnow_g,
                                its_out,
                                its_p,
                                Z_g,
                                U_g,
                                S_g,
                                T_g,
                                qvnow=qvnow_g,
                                qcnow=qcnow_g,
                                qrnow=qrnow_g,
                                tot_prec=tot_prec_g,
                                PREC=PREC_g,
                                prec=prec_g,
                                QV=QV_g,
                                QC=QC_g,
                                QR=QR_g,
                                TOT_PREC=TOT_PREC_g,
                                dthetadt=dthetadt_g,
                                DTHETADT=DTHETADT_g,
                            )
                if imicrophys_n == 2:
                    if idthdt_n == 0:
                        # TODO: gather qvnow_g, qcnow_g, qrnow_g, tot_prec_g, prec_g, nrnow_g, ncnow_g
                        if rank_p == 0:
                            its_out, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, NR_g, NC_g = makeoutput(
                                unow_g,
                                snow_g,
                                zhtnow_g,
                                its_out,
                                its_p,
                                Z_g,
                                U_g,
                                S_g,
                                T_g,
                                qvnow=qvnow_g,
                                qcnow=qcnow_g,
                                qrnow=qrnow_g,
                                tot_prec=tot_prec_g,
                                prec=prec_g,
                                nrnow=nrnow_g,
                                ncnow=ncnow_g,
                                QV=QV_g,
                                QC=QC_g,
                                QR=QR_g,
                                TOT_PREC=TOT_PREC_g,
                                PREC=PREC_g,
                                NR=NR_g,
                                NC=NC_g,
                            )
                    if idthdt_n == 1:
                        # TODO: gather qvnow_g, qcnow_g, qrnow_g, tot_prec_g, prec_g, nrnow_g, ncnow_g
                        if rank_p == 0:
                            its_out, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, NR_g, NC_g, DTHETADT_g = makeoutput(
                                unow_g,
                                snow_g,
                                zhtnow_g,
                                its_out,
                                its_p,
                                Z_g,
                                U_g,
                                S_g,
                                T_g,
                                qvnow=qvnow_g,
                                qcnow=qcnow_g,
                                qrnow=qrnow_g,
                                tot_prec=tot_prec_g,
                                prec=prec_g,
                                nrnow=nrnow_g,
                                ncnow=ncnow_g,
                                QV=QV_g,
                                QC=QC_g,
                                QR=QR_g,
                                TOT_PREC=TOT_PREC_g,
                                PREC=PREC_g,
                                NR=NR_g,
                                NC=NC_g,
                                dthetadt=dthetadt_g,
                                DTHETADT=DTHETADT_g,
                            )
        # endregion

        if idbg_n == 1:
            print("\n\n")

        # Exchange borderpoints_n
        exchange_borders(sold_p, 0)
        exchange_borders(snow_p, 1)
        exchange_borders(snew_p, 2)

        exchange_borders(uold_p, tag=3)
        exchange_borders(unow_p, tag=4)

        exchange_borders(qvold_p, tag=5)
        exchange_borders(qvnow_p, tag=6)
        exchange_borders(qvnew_p, tag=7)

        exchange_borders(qcold_p, tag=8)
        exchange_borders(qcnow_p, tag=9)
        exchange_borders(qcnew_p, tag=10)

        exchange_borders(qrold_p, tag=11)
        exchange_borders(qrnow_p, tag=12)
        exchange_borders(qrnew_p, tag=13)

        if imoist_n == 1 and imicrophys_n == 2:
            exchange_borders(ncold_p, tag=14)
            exchange_borders(ncnow_p, tag=15)
            exchange_borders(ncnew_p, tag=16)

            exchange_borders(nrold_p, tag=17)
            exchange_borders(nrnow_p, tag=18)
            exchange_borders(nrnew_p, tag=19)

        exchange_borders(mtg_p, tag=20)
        exchange_borders(prs_p, tag=23)

    # -----------------------------------------------------------------------------
    # ########## END OF TIME LOOP ################################################
    if idbg_n > 0:
        print("\nEnd of time loop ...\n")

    tt = tm()
    print("Elapsed computation time without writing: %g s\n" % (tt - t0_p))

    # endregion

    # region Write output
    # ---------------------------------
    print("Start wrtiting output.\n")
    if imoist_n == 0:
        write_output(nout, Z, U, S, T)
    elif imicrophys_n == 0 or imicrophys_n == 1:
        if idthdt_n == 1:
            write_output(
                nout,
                Z,
                U,
                S,
                T,
                QV=QV,
                QC=QC,
                QR=QR,
                PREC=PREC,
                TOT_PREC=TOT_PREC,
                dt_nHETAdt_n=DTHETADT_n,
            )
        else:
            write_output(
                nout, Z, U, S, T, QV=QV, QC=QC, QR=QR, PREC=PREC, TOT_PREC=TOT_PREC
            )
    elif imicrophys_n == 2:
        if idthdt_n == 1:
            write_output(
                nout,
                Z,
                U,
                S,
                T,
                QV=QV,
                QC=QC,
                QR=QR,
                PREC=PREC,
                TOT_PREC=TOT_PREC,
                NR=NR,
                NC=NC,
                dt_nHETAdt_n=DTHETADT_n,
            )
        else:
            write_output(
                nout,
                Z,
                U,
                S,
                T,
                QV=QV,
                QC=QC,
                QR=QR,
                PREC=PREC,
                TOT_PREC=TOT_PREC,
                NR=NR,
                NC=NC,
            )
    # endregion
    t1 = tm()

    if itime_n == 1:
        print("Total elapsed computation time: %g s\n" % (t1 - t0_p))


if __name__ == '__main__':
    main()

# END OF SOLVER.PY
