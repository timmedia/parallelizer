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
* improvements by Mathias Hauser, Deniz Ural,                 *
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
    imoist,
    imicrophys,
    irelax,
    idthdt,
    idbg,
    iprtcfl,
    nts,
    dt,
    iiniout,
    nout as _nout,
    iout,
    dx,
    nx,
    nb,
    nz,
    nz1,
    nab,
    rdcp,
    g,
    diff,
    diffabs,
    topotim,
    itime,
)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank_size = comm.Get_size()

assert nx % rank_size == 0, "Number of grid points must be compatible with rank size"

nx_p = nx // rank_size


def exchange_borders(data, tag: int):
    """Exchange broders with next rank for 2-dimensional data set `data`. (scattered along first axis)"""
    left_rank = (rank - 1) % rank_size
    right_rank = (rank + 1) % rank_size

    send_to_right = data[-2*nb:-nb]
    send_to_left = data[nb:2*nb]

    new_left_border = np.empty(nb)
    new_right_border = np.empty(nb)

    comm.Sendrecv(sendbuf=send_to_left, dest=left_rank, sendtag=rank * 10_000 + 100 * left_rank +
                  tag, recvbuf=new_right_border, source=left_rank, recvtag=left_rank * 10_000 + 100 * rank + tag)

    comm.Sendrecv(sendbuf=send_to_right, dest=right_rank, sendtag=rank * 10_000 + 100 * right_rank +
                  tag, recvbuf=new_left_border, source=left_rank, recvtag=right_rank * 10_000 + 100 * rank + tag)


def main():
    # Print the full precision
    # DL: REMOVE FOR STUDENT VERSION
    np.set_printoptions(threshold=sys.maxsize)

    nout = _nout

    # increase number of output steps by 1 for initial profile
    if iiniout == 1:
        nout += 1

    # Define physical fields
    # -------------------------
    if rank == 0:
        # topography
        topo = np.zeros((nx + 2 * nb, 1))

        # height in z-coordinates
        zhtold = np.zeros((nx + 2 * nb, nz1))
        zhtnow = np.zeros_like(zhtold)
        Z = np.zeros((nout, nz1, nx))  # auxilary field for output

        # horizontal velocity
        uold = np.zeros((nx + 1 + 2 * nb, nz))
        unow = np.zeros_like(uold)
        unew = np.zeros_like(uold)
        U = np.zeros((nout, nz, nx))  # auxilary field for output

        # isentropic density
        sold = np.zeros((nx + 2 * nb, nz))
        snow = np.zeros_like(sold)
        snew = np.zeros_like(sold)
        S = np.zeros((nout, nz, nx))  # auxilary field for output

        # Montgomery potential
        mtg = np.zeros((nx + 2 * nb, nz))
        mtgnew = np.zeros_like(mtg)

        # Exner function
        exn = np.zeros((nx + 2 * nb, nz1))

        # pressure
        prs = np.zeros((nx + 2 * nb, nz1))

        # output time vector
        T = np.arange(1, nout + 1)

        if imoist == 1:
            # precipitation
            prec = np.zeros(nx + 2 * nb)
            PREC = np.zeros((nout, nx))  # auxiliary field for output

            # accumulated precipitation
            tot_prec = np.zeros(nx + 2 * nb)
            TOT_PREC = np.zeros((nout, nx))  # auxiliary field for output

            # specific humidity
            qvold = np.zeros((nx + 2 * nb, nz))
            qvnow = np.zeros_like(qvold)
            qvnew = np.zeros_like(qvold)
            QV = np.zeros((nout, nz, nx))  # auxiliary field for output

            # specific cloud water content
            qcold = np.zeros((nx + 2 * nb, nz))
            qcnow = np.zeros_like(qcold)
            qcnew = np.zeros_like(qcold)
            QC = np.zeros((nout, nz, nx))  # auxiliary field for output

            # specific rain water content
            qrold = np.zeros((nx + 2 * nb, nz))
            qrnow = np.zeros_like(qrold)
            qrnew = np.zeros_like(qrold)
            QR = np.zeros((nout, nz, nx))  # auxiliary field for output

            if imicrophys == 2:
                # cloud droplet number density
                ncold = np.zeros((nx + 2 * nb, nz))
                ncnow = np.zeros_like(ncold)
                ncnew = np.zeros_like(ncold)
                NC = np.zeros((nout, nz, nx))  # auxiliary field for output

                # rain-droplet number density
                nrold = np.zeros((nx + 2 * nb, nz))
                nrnow = np.zeros_like(nrold)
                nrnew = np.zeros_like(nrold)
                NR = np.zeros((nout, nz, nx))  # auxiliary field for output

            if idthdt == 1:
                # latent heating
                dthetadt = np.zeros((nx + 2 * nb, nz1))
                # auxiliary field for output
                DTHETADT = np.zeros((nout, nz, nx))

        # Define fields at lateral boundaries
        # 1 denotes the left boundary
        # 2 denotes the right boundary
        # ----------------------------------------------------------------------------
        # topography
        tbnd1 = 0.0
        tbnd2 = 0.0

        # isentropic density
        sbnd1 = np.zeros(nz)
        sbnd2 = np.zeros(nz)

        # horizontal velocity
        ubnd1 = np.zeros(nz)
        ubnd2 = np.zeros(nz)

        if imoist == 1:
            # specific humidity
            qvbnd1 = np.zeros(nz)
            qvbnd2 = np.zeros(nz)

            # specific cloud water content
            qcbnd1 = np.zeros(nz)
            qcbnd2 = np.zeros(nz)

            # specific rain water content
            qrbnd1 = np.zeros(nz)
            qrbnd2 = np.zeros(nz)

        if idthdt == 1:
            # latent heating
            dthetadtbnd1 = np.zeros(nz1)
            dthetadtbnd2 = np.zeros(nz1)

        if imicrophys == 2:
            # cloud droplet number density
            ncbnd1 = np.zeros(nz)
            ncbnd2 = np.zeros(nz)

            # rain droplet number density
            nrbnd1 = np.zeros(nz)
            nrbnd2 = np.zeros(nz)

        # Set initial conditions
        # -----------------------------------------------------------------------------
        if idbg == 1:
            print("Setting initial conditions ...\n")

        if imoist == 0:
            # Dry atmosphere
            th0, exn0, prs0, z0, mtg0, s0, u0, sold, snow, uold, unow, mtg, mtgnew = makeprofile(
                sold, uold, mtg, mtgnew
            )
        else:
            if imicrophys == 0 or imicrophys == 1:
                # moist atmosphere with kessler scheme
                [
                    th0,
                    exn0,
                    prs0,
                    z0,
                    mtg0,
                    s0,
                    u0,
                    sold,
                    snow,
                    uold,
                    unow,
                    mtg,
                    mtgnew,
                    qv0,
                    qc0,
                    qr0,
                    qvold,
                    qvnow,
                    qcold,
                    qcnow,
                    qrold,
                    qrnow,
                ] = makeprofile(
                    sold,
                    uold,
                    qvold=qvold,
                    qvnow=qvnow,
                    qcold=qcold,
                    qcnow=qcnow,
                    qrold=qrold,
                    qrnow=qrnow,
                )
            elif imicrophys == 2:
                # moist atmosphere with 2-moment scheme
                [
                    th0,
                    exn0,
                    prs0,
                    z0,
                    mtg0,
                    s0,
                    u0,
                    sold,
                    snow,
                    uold,
                    unow,
                    mtg,
                    mtgnew,
                    qv0,
                    qc0,
                    qr0,
                    qvold,
                    qvnow,
                    qcold,
                    qcnow,
                    qrold,
                    qrnow,
                    ncold,
                    ncnow,
                    nrold,
                    nrnow,
                ] = makeprofile(
                    sold,
                    uold,
                    qvold=qvold,
                    qvnow=qvnow,
                    qcold=qcold,
                    qcnow=qcnow,
                    qrold=qrold,
                    qrnow=qrnow,
                    ncold=ncold,
                    ncnow=ncnow,
                    nrold=nrold,
                    nrnow=nrnow,
                )

        # Save boundary values for the lateral boundary relaxation
        if irelax == 1:
            if idbg == 1:
                print("Saving initial lateral boundary values ...\n")

            sbnd1[:] = snow[0, :]
            sbnd2[:] = snow[-1, :]

            ubnd1[:] = unow[0, :]
            ubnd2[:] = unow[-1, :]

            if imoist == 1:
                qvbnd1[:] = qvnow[0, :]
                qvbnd2[:] = qvnow[-1, :]

                qcbnd1[:] = qcnow[0, :]
                qcbnd2[:] = qcnow[-1, :]

            if imicrophys != 0:
                qrbnd1[:] = qrnow[0, :]
                qrbnd2[:] = qrnow[-1, :]

            # 2-moment microphysics scheme
            if imicrophys == 2:
                k = np.arange(0, nz)
                ncbnd1[:] = ncnow[0, :]
                ncbnd2[:] = ncnow[-1, :]

                nrbnd1[:] = nrnow[0, :]
                nrbnd2[:] = nrnow[-1, :]

            if idthdt == 1:
                dthetadtbnd1[:] = dthetadt[0, :]
                dthetadtbnd2[:] = dthetadt[-1, :]

        # Make topography
        # ----------------
        topo = maketopo(topo, nx + 2 * nb)

        # switch between boundary relaxation / periodic boundary conditions
        # ------------------------------------------------------------------
        if irelax == 1:  # boundary relaxation
            if idbg == 1:
                print("Relax topography ...\n")

            # save lateral boundary values of topography
            tbnd1 = topo[0]
            tbnd2 = topo[-1]

            # relax topography
            topo = relax(topo, nx, nb, tbnd1, tbnd2)
        else:
            if idbg == 1:
                print("Periodic topography ...\n")

            # make topography periodic
            topo = periodic(topo, nx, nb)

        # calculate geometric height (staggered)
        zhtnow = diag_height(
            prs0[np.newaxis, :], exn0[np.newaxis, :], zhtnow, th0, topo, 0.0
        )

        # Height-dependent diffusion coefficient
        # --------------------------------------
        tau = diff * np.ones(nz)

        # *** Exercise 3.1 height-dependent diffusion coefficient ***
        # *** edit here ***

        k = np.arange(nz)
        tau = diff + (diffabs - diff)*np.sin(np.pi/2*(k-nz+nab-1)/nab)**2
        tau[0:nz-nab] = diff
        # *** Exercise 3.1 height-dependent diffusion coefficient ***

        # output initial fields
        its_out = -1  # output index
        if iiniout == 1 and imoist == 0:
            its_out, Z, U, S, T = makeoutput(
                unow, snow, zhtnow, its_out, 0, Z, U, S, T)
        elif iiniout == 1 and imoist == 1:
            if imicrophys == 0 or imicrophys == 1:
                if idthdt == 0:
                    [its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC] = makeoutput(
                        unow,
                        snow,
                        zhtnow,
                        its_out,
                        0,
                        Z,
                        U,
                        S,
                        T,
                        qvnow=qvnow,
                        qcnow=qcnow,
                        qrnow=qrnow,
                        tot_prec=tot_prec,
                        prec=prec,
                        QV=QV,
                        QC=QC,
                        QR=QR,
                        TOT_PREC=TOT_PREC,
                        PREC=PREC,
                    )
                elif idthdt == 1:
                    [
                        its_out,
                        Z,
                        U,
                        S,
                        T,
                        QC,
                        QV,
                        QR,
                        TOT_PREC,
                        PREC,
                        DTHETADT,
                    ] = makeoutput(
                        unow,
                        snow,
                        zhtnow,
                        its_out,
                        0,
                        Z,
                        U,
                        S,
                        T,
                        qvnow=qvnow,
                        qcnow=qcnow,
                        qrnow=qrnow,
                        tot_prec=tot_prec,
                        prec=prec,
                        QV=QV,
                        QC=QC,
                        QR=QR,
                        TOT_PREC=TOT_PREC,
                        PREC=PREC,
                        dthetadt=dthetadt,
                        DTHETADT=DTHETADT,
                    )
            elif imicrophys == 2:
                if idthdt == 0:
                    [its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, NC, NR] = makeoutput(
                        unow,
                        snow,
                        zhtnow,
                        its_out,
                        0,
                        Z,
                        U,
                        S,
                        T,
                        qvnow=qvnow,
                        qcnow=qcnow,
                        qrnow=qrnow,
                        tot_prec=tot_prec,
                        prec=prec,
                        nrnow=nrnow,
                        ncnow=ncnow,
                        QV=QV,
                        QC=QC,
                        QR=QR,
                        TOT_PREC=TOT_PREC,
                        PREC=PREC,
                        NC=NC,
                        NR=NR,
                    )
                elif idthdt == 1:
                    [
                        its_out,
                        Z,
                        U,
                        S,
                        T,
                        QC,
                        QV,
                        QR,
                        TOT_PREC,
                        PREC,
                        NC,
                        NR,
                        DTHETADT,
                    ] = makeoutput(
                        unow,
                        snow,
                        zhtnow,
                        its_out,
                        0,
                        Z,
                        U,
                        S,
                        T,
                        qvnow=qvnow,
                        qcnow=qcnow,
                        qrnow=qrnow,
                        tot_prec=tot_prec,
                        prec=prec,
                        nrnow=nrnow,
                        ncnow=ncnow,
                        QV=QV,
                        QC=QC,
                        QR=QR,
                        TOT_PREC=TOT_PREC,
                        PREC=PREC,
                        NC=NC,
                        NR=NR,
                        dthetadt=dthetadt,
                        DTHETADT=DTHETADT,
                    )

        for i in range(1, rank_size):
            start_index = i * nx // rank_size
            end_index = (i + 1) * nx // rank_size + 2 * nb
            rank_slice_with_borders = slice(start_index, end_index)
            rank_slice_uneven_with_borders = slice(start_index, end_index + 1)

            comm.Send(sold[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 0)
            comm.Send(snow[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 1)
            comm.Send(snew[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 2)

            comm.Send(uold[rank_slice_uneven_with_borders, :],
                      dest=i, tag=i * 1000 + 3)
            comm.Send(unow[rank_slice_uneven_with_borders, :],
                      dest=i, tag=i * 1000 + 4)

            comm.Send(qvold[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 5)
            comm.Send(qvnow[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 6)
            comm.Send(qvnew[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 7)

            comm.Send(qcold[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 8)
            comm.Send(qcnow[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 9)
            comm.Send(qcnew[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 10)

            comm.Send(qrold[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 11)
            comm.Send(qrnow[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 12)
            comm.Send(qrnew[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 13)

            if imoist == 1 and imicrophys == 2:
                comm.Send(ncold[rank_slice_with_borders, :],
                          dest=i, tag=i * 1000 + 14)
                comm.Send(ncnow[rank_slice_with_borders, :],
                          dest=i, tag=i * 1000 + 15)
                comm.Send(ncnew[rank_slice_with_borders, :],
                          dest=i, tag=i * 1000 + 16)

                comm.Send(nrold[rank_slice_with_borders, :],
                          dest=i, tag=i * 1000 + 17)
                comm.Send(nrnow[rank_slice_with_borders, :],
                          dest=i, tag=i * 1000 + 18)
                comm.Send(nrnew[rank_slice_with_borders, :],
                          dest=i, tag=i * 1000 + 19)

            comm.Send(mtg[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 20)

            comm.Send(tau, dest=i, tag=i * 1000 + 21)
            comm.Send(prs0, dest=i, tag=i * 1000 + 22)
            comm.Send(prs[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 23)
            comm.Send(topo[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 24)
            comm.Send(zhtold[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 25)
            comm.Send(zhtnow[rank_slice_with_borders, :],
                      dest=i, tag=i * 1000 + 26)

        # local variables for process 0
        start_index = 0
        end_index = nx // rank_size + 2 * nb
        rank_slice_with_borders = slice(start_index, end_index)
        rank_slice_uneven_with_borders = slice(start_index, end_index + 1)

        sold_p = sold[rank_slice_with_borders, :]
        snow_p = snow[rank_slice_with_borders, :]
        snew_p = snew[rank_slice_with_borders, :]

        uold_p = uold[rank_slice_uneven_with_borders, :]
        unow_p = unow[rank_slice_uneven_with_borders, :]

        qvold_p = qvold[rank_slice_with_borders, :]
        qvnow_p = qvnow[rank_slice_with_borders, :]
        gvnew_p = qvnew[rank_slice_with_borders, :]

        qcold_p = qcold[rank_slice_with_borders, :]
        qcnow_p = qcnow[rank_slice_with_borders, :]
        qcnew_p = qcnew[rank_slice_with_borders, :]

        qrold_p = qrold[rank_slice_with_borders, :]
        qrnow_p = qrnow[rank_slice_with_borders, :]
        qrnew_p = qrnew[rank_slice_with_borders, :]

        if imoist == 1 and imicrophys == 2:
            ncold_p = ncold[rank_slice_with_borders, :]
            ncnow_p = ncnow[rank_slice_with_borders, :]
            ncnew_p = ncnew[rank_slice_with_borders, :]

            nrold_p = nrold[rank_slice_with_borders, :]
            nrnow_p = nrnow[rank_slice_with_borders, :]
            nrnew_p = nrnew[rank_slice_with_borders, :]

        mtg_p = mtg[rank_slice_with_borders, :]
        tau_p = tau
        prs0_p = prs0
        prs_p = prs[rank_slice_with_borders, :]
        topo_p = topo[rank_slice_with_borders, :]
        zhtold = zhtold[rank_slice_with_borders, :]
        zhtnow = zhtnow[rank_slice_with_borders, :]

    else:
        sold_p = np.empty((nx // rank_size + 2 * nb, nz))
        snow_p = np.empty((nx // rank_size + 2 * nb, nz))
        snew_p = np.empty((nx // rank_size + 2 * nb, nz))
        uold_p = np.empty((nx // rank_size + 1 + 2 * nb, nz))
        unow_p = np.empty((nx // rank_size + 1 + 2 * nb, nz))
        qvold_p = np.empty((nx // rank_size + 2 * nb, nz))
        qvnow_p = np.empty((nx // rank_size + 2 * nb, nz))
        qvnew_p = np.empty((nx // rank_size + 2 * nb, nz))
        qcnow_p = np.empty((nx // rank_size + 2 * nb, nz))
        qcnew_p = np.empty((nx // rank_size + 2 * nb, nz))
        qrold_p = np.empty((nx // rank_size + 2 * nb, nz))
        qrnow_p = np.empty((nx // rank_size + 2 * nb, nz))
        qrnew_p = np.empty((nx // rank_size + 2 * nb, nz))
        qcold_p = np.empty((nx // rank_size + 2 * nb, nz))
        qcnow_p = np.empty((nx // rank_size + 2 * nb, nz))
        mtg_p = np.empty((nx // rank_size + 2 * nb, nz))
        tau_p = np.empty(nz)
        prs0_p = np.zeros(nz + 1)
        prs_p = np.empty((nx // rank_size + 2 * nb, nz))
        topo_p = np.empty((nx // rank_size + 2 * nb, 1))
        zhtold_p = np.empty((nx // rank_size + 2 * nb, nz))
        zhtnow_p = np.empty((nx // rank_size + 2 * nb, nz))

        comm.Recv(sold_p, source=0, tag=rank * 1000 + 0)
        comm.Recv(snow_p, source=0, tag=rank * 1000 + 1)
        comm.Recv(snew_p, source=0, tag=rank * 1000 + 2)

        comm.Recv(uold_p, source=0, tag=rank * 1000 + 3)
        comm.Recv(unow_p, source=0, tag=rank * 1000 + 4)

        comm.Recv(qvold_p, source=0, tag=rank * 1000 + 5)
        comm.Recv(qvnow_p, source=0, tag=rank * 1000 + 6)
        comm.Recv(qvnew_p, source=0, tag=rank * 1000 + 7)

        comm.Recv(qcold_p, source=0, tag=rank * 1000 + 8)
        comm.Recv(qcnow_p, source=0, tag=rank * 1000 + 9)
        comm.Recv(qcnew_p, source=0, tag=rank * 1000 + 10)

        comm.Recv(qrold_p, source=0, tag=rank * 1000 + 11)
        comm.Recv(qrnow_p, source=0, tag=rank * 1000 + 12)
        comm.Recv(qrnew_p, source=0, tag=rank * 1000 + 13)

        if imoist == 1 and imicrophys == 2:
            ncold_p = np.empty((nx // rank_size + 2 * nb, nz))
            ncnow_p = np.empty((nx // rank_size + 2 * nb, nz))
            ncnew_p = np.empty((nx // rank_size + 2 * nb, nz))
            nrold_p = np.empty((nx // rank_size + 2 * nb, nz))
            nrnow_p = np.empty((nx // rank_size + 2 * nb, nz))
            nrnew_p = np.empty((nx // rank_size + 2 * nb, nz))

            comm.Recv(ncold_p, source=0, tag=rank * 1000 + 14)
            comm.Recv(ncnow_p, source=0, tag=rank * 1000 + 15)
            comm.Recv(ncnew_p, source=0, tag=rank * 1000 + 16)

            comm.Recv(nrold_p, source=0, tag=rank * 1000 + 17)
            comm.Recv(nrnow_p, source=0, tag=rank * 1000 + 18)
            comm.Recv(nrnew_p, source=0, tag=rank * 1000 + 19)

        comm.Recv(mtg_p, source=0, tag=rank * 1000 + 20)
        comm.Recv(tau_p, source=0, tag=rank * 1000 + 21)
        comm.Recv(prs0_p, source=0, tag=rank * 1000 + 22)
        comm.Recv(prs_p, source=0, tag=rank * 1000 + 23)
        comm.Recv(topo_p, source=0, tag=rank * 1000 + 24)

        comm.Recv(zhtold_p, source=0, tag=i * 1000 + 25)
        comm.Recv(zhtnow_p, source=0, tag=i * 1000 + 26)

    # ########## TIME LOOP #######################################################
    # ----------------------------------------------------------------------------
    # Loop over all time steps
    # ----------------------------------------------------------------------------
    if idbg == 1 and rank == 0:
        print("Starting time loop ...\n")

    t0 = tm()
    for its in range(1, int(nts + 1)):
        # calculate time
        time = its * dt

        if itime == 1:
            if idbg == 1 or idbg == 0:
                print("========================================================\n")
                print("Working on timestep %g; time = %g s; process = %g\n" %
                      (its, time, rank))
                print("========================================================\n")

        # initially increase height of topography only slowly
        topofact: float = min(1.0, float(time) / topotim)

        # Special treatment of first time step
        # -------------------------------------------------------------------------
        if its == 1:
            dtdx: float = dt / dx / 2.0
            dthetadt = None
            if imoist == 1 and idthdt == 1:
                # No latent heating for first time-step
                dthetadt = np.zeros((nx // rank_size + 2 * nb, nz1))
            if idbg == 1:
                print("Using Euler forward step for 1. step ...\n")
        else:
            dtdx: float = dt / dx

        # *** Exercise 2.1 isentropic mass density ***
        # *** time step for isentropic mass density ***
        snew_p = prog_isendens(sold_p, snow_p, unow_p,
                               dtdx, dthetadt=dthetadt, nx=nx // rank_size)
        #
        # *** Exercise 2.1 isentropic mass density ***

        # *** Exercise 4.1 / 5.1 moisture ***
        # *** time step for moisture scalars ***
        if imoist == 1:
            if idbg == 1:
                print("Add function call to prog_moisture")
            qvnew_p, qcnew_p, qrnew_p = prog_moisture(
                unow_p, qvold_p, qcold_p, qrold_p, qvnow_p, qcnow_p, qrnow_p, dtdx, dthetadt=dthetadt, nx=nx // rank_size)

            if imicrophys == 2:
                ncnew_p, nrnew_p = prog_numdens(
                    unow_p, ncold_p, nrold_p, ncnow_p, nrnow_p, dtdx, dthetadt=dthetadt, nx=nx // rank_size)

        #
        # *** Exercise 4.1 / 5.1 moisture scalars ***

        # *** Exercise 2.1 velocity ***
        # *** time step for momentum ***
        #

        # *** edit here ***
        unew_p = prog_velocity(uold_p, unow_p, mtg_p,
                               dtdx, dthetadt=dthetadt, nx=nx // rank_size)
        #
        # *** Exercise 2.1 velocity ***

        # exchange boundaries if periodic
        # -------------------------------------------------------------------------
        if irelax == 0:
            if rank == 0:
                snew_p[0:nb, :] = comm.Sendrecv(sendbuf=snew_p[nb:2*nb, :], dest=rank_size - 1,
                                                sendtag=11, recvbuf=None, source=rank_size - 1, recvtag=22)
                unew_p[0:nb, :] = comm.Sendrecv(sendbuf=unew_p[nb:2*nb, :], dest=rank_size - 1,
                                                sendtag=111, recvbuf=None, source=rank_size - 1, recvtag=222)
            elif rank == rank_size - 1:
                snew_p[-nb:, :] = comm.Sendrecv(sendbuf=snew_p[-2*nb:-nb],
                                                dest=0, sendtag=22, recvbuf=None, source=0, recvtag=11)
                unew_p[-nb:, :] = comm.Sendrecv(sendbuf=unew_p[-2*nb:-nb],
                                                dest=0, sendtag=222, recvbuf=None, source=0, recvtag=111)

            if imoist == 1:
                pass
                # qvnew_p = periodic(qvnew_p, nx, nb) # TODO
                # qcnew_p = periodic(qcnew, nx, nb) # TODO
                # qrnew_p = periodic(qrnew, nx, nb) # TODO

            # 2-moment scheme
            if imoist == 1 and imicrophys == 2:
                pass
                # ncnew = periodic(ncnew, nx, nb) # TODO
                # nrnew = periodic(nrnew, nx, nb) # TODO

        # relaxation of prognostic fields
        # -------------------------------------------------------------------------
        if irelax == 1:
            if idbg == 1:
                print("Relaxing prognostic fields ...\n")
            # snew = relax(snew, nx, nb, sbnd1, sbnd2) # TODO
            # unew = relax(unew, nx + 1, nb, ubnd1, ubnd2) # TODO
            if imoist == 1:
                pass
                # qvnew = relax(qvnew, nx, nb, qvbnd1, qvbnd2) # TODO
                # qcnew = relax(qcnew, nx, nb, qcbnd1, qcbnd2) # TODO
                # qrnew = relax(qrnew, nx, nb, qrbnd1, qrbnd2) # TODO

            # 2-moment scheme
            if imoist == 1 and imicrophys == 2:
                pass
                # ncnew = relax(ncnew, nx, nb, ncbnd1, ncbnd2) # TODO
                # nrnew = relax(nrnew, nx, nb, nrbnd1, nrbnd2) # TODO

        # Diffusion and gravity wave absorber
        # ------------------------------------

        if imoist == 0:
            [unew_p, snew_p] = horizontal_diffusion(
                tau_p, unew_p, snew_p, nx=nx // rank_size)
        else:
            if imicrophys == 2:
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
                    tau_p, unew_p, snew_p, qvnew=qvnew_p, qcnew=qcnew_p, qrnew=qrnew_p, nx=nx // rank_size
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
        exn_p, mtg_p = diag_montgomery(prs_p, mtg_p, th0, topo_p, topofact)
        #
        # *** Exercise 2.2 Diagnostic computation of Montgomery ***

        # Calculation of geometric height (staggered)
        # needed for output and microphysics schemes
        # ---------------------------------
        zhtold_p[...] = zhtnow_p[...]
        zhtnow_p = diag_height(prs_p, exn_p, zhtnow_p, th0, topo_p, topofact)

        if imoist == 1:
            # *** Exercise 4.1 Moisture ***
            # *** Clipping of negative values ***
            # *** edit here ***
            #

            if idbg == 1:
                print("Implement moisture clipping")
            qvnew_p[qvnew_p < 0] = 0
            qcnew_p[qcnew_p < 0] = 0
            qrnew_p[qrnew_p < 0] = 0

            if imicrophys == 2:
                ncnew_p[ncnew_p < 0] = 0
                nrnew_p[nrnew_p < 0] = 0

            #
            # *** Exercise 4.1 Moisture ***

        if imoist == 1 and imicrophys == 1:
            # *** Exercise 4.2 Kessler ***
            # *** Kessler scheme ***
            # *** edit here ***
            #

            if idbg == 1:
                print("Add function call to Kessler microphysics")
            [lheat, qvnew_p, qcnew_p, qrnew_p, prec_p, prec_tot] = kessler(
                snew_p, qvnew_p, qcnew_p, qrnew_p, prs_p, exn_p, zhtnow_p, th0, prec, tot_prec)

            #
            # *** Exercise 4.2 Kessler ***
        elif imoist == 1 and imicrophys == 2:
            # *** Exercise 5.1 Two Moment Scheme ***
            # *** Two Moment Scheme ***
            # *** edit here ***
            #

            if idbg == 1:
                print("Add function call to two moment microphysics")
            [lheat, qvnew_p, qcnew_p, qrnew_p, tot_prec, prec, ncnew_p, nrnew_p] = seifert(
                unew_p, th0, prs_p, snew_p, qvnew_p, qcnew_p, qrnew_p, exn_p, zhtold_p, zhtnow_p, tot_prec, prec, ncnew_p, nrnew_p, dthetadt)
            #
            # *** Exercise 5.1 Two Moment Scheme ***

        if imoist == 1 and imicrophys > 0:
            if idthdt == 1:
                # Stagger lheat to model levels and compute tendency
                k = np.arange(1, nz)
                if imicrophys == 1:
                    dthetadt[:, k] = topofact * 0.5 * \
                        (lheat[:, k - 1] + lheat[:, k]) / dt
                else:
                    dthetadt[:, k] = topofact * 0.5 * \
                        (lheat[:, k - 1] + lheat[:, k]) / (2.0 * dt)

                # force dthetadt to zeros at the bottom and at the top
                dthetadt[:, 0] = 0.0
                dthetadt[:, -1] = 0.0

                # periodic lateral boundary conditions
                # ----------------------------
                if irelax == 0:
                    dthetadt = periodic(dthetadt, nx, nb)
                else:
                    # Relax latent heat fields
                    # ----------------------------
                    dthetadt = relax(dthetadt, nx, nb,
                                     dthetadtbnd1, dthetadtbnd2)
            else:
                dthetadt = np.zeros((nx + 2 * nb, nz1))

        if idbg == 1:
            print("Preparing next time step ...\n")

        # *** Exercise 2.1 / 4.1 / 5.1 ***
        # *** exchange isentropic mass density and velocity ***
        # *** (later also qv,qc,qr,nc,nr) ***
        # *** edit here ***
        if imicrophys == 2:
            ncold = ncnow
            ncnow = ncnew

            nrold = nrnow
            nrnow = nrnew

        sold = snow
        snow = snew

        uold = unow
        unow = unew

        if imoist == 1:
            qvold = qvnow
            qvnow = qvnew

            qcold = qcnow
            qcnow = qcnew

            qrold = qrnow
            qrnow = qrnew
            if idbg == 1:
                print("exchange moisture variables")

            if imicrophys == 2:
                if idbg == 1:
                    print("exchange number densitiy variables")

        #
        # *** Exercise 2.1 / 4.1 / 5.1 ***

        # check maximum cfl criterion
        # ---------------------------------
        if iprtcfl == 1:
            u_max = np.amax(np.abs(unow))
            cfl_max = u_max * dtdx
            print("============================================================\n")
            print("CFL MAX: %g U MAX: %g m/s \n" % (cfl_max, u_max))
            if cfl_max > 1:
                print("!!! WARNING: CFL larger than 1 !!!\n")
            elif np.isnan(cfl_max):
                print("!!! MODEL ABORT: NaN values !!!\n")
            print("============================================================\n")

        # output every 'iout'-th time step
        # ---------------------------------
        if np.mod(its, iout) == 0:
            if imoist == 0:
                its_out, Z, U, S, T = makeoutput(
                    unow, snow, zhtnow, its_out, its, Z, U, S, T
                )
            elif imoist == 1:
                if imicrophys == 0 or imicrophys == 1:
                    if idthdt == 0:
                        its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC = makeoutput(
                            unow,
                            snow,
                            zhtnow,
                            its_out,
                            its,
                            Z,
                            U,
                            S,
                            T,
                            qvnow=qvnow,
                            qcnow=qcnow,
                            qrnow=qrnow,
                            tot_prec=tot_prec,
                            prec=prec,
                            QV=QV,
                            QC=QC,
                            QR=QR,
                            TOT_PREC=TOT_PREC,
                            PREC=PREC,
                        )
                    elif idthdt == 1:
                        its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, DTHETADT = makeoutput(
                            unow,
                            snow,
                            zhtnow,
                            its_out,
                            its,
                            Z,
                            U,
                            S,
                            T,
                            qvnow=qvnow,
                            qcnow=qcnow,
                            qrnow=qrnow,
                            tot_prec=tot_prec,
                            PREC=PREC,
                            prec=prec,
                            QV=QV,
                            QC=QC,
                            QR=QR,
                            TOT_PREC=TOT_PREC,
                            dthetadt=dthetadt,
                            DTHETADT=DTHETADT,
                        )
                if imicrophys == 2:
                    if idthdt == 0:
                        its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, NR, NC = makeoutput(
                            unow,
                            snow,
                            zhtnow,
                            its_out,
                            its,
                            Z,
                            U,
                            S,
                            T,
                            qvnow=qvnow,
                            qcnow=qcnow,
                            qrnow=qrnow,
                            tot_prec=tot_prec,
                            prec=prec,
                            nrnow=nrnow,
                            ncnow=ncnow,
                            QV=QV,
                            QC=QC,
                            QR=QR,
                            TOT_PREC=TOT_PREC,
                            PREC=PREC,
                            NR=NR,
                            NC=NC,
                        )
                    if idthdt == 1:
                        its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, NR, NC, DTHETADT = makeoutput(
                            unow,
                            snow,
                            zhtnow,
                            its_out,
                            its,
                            Z,
                            U,
                            S,
                            T,
                            qvnow=qvnow,
                            qcnow=qcnow,
                            qrnow=qrnow,
                            tot_prec=tot_prec,
                            prec=prec,
                            nrnow=nrnow,
                            ncnow=ncnow,
                            QV=QV,
                            QC=QC,
                            QR=QR,
                            TOT_PREC=TOT_PREC,
                            PREC=PREC,
                            NR=NR,
                            NC=NC,
                            dthetadt=dthetadt,
                            DTHETADT=DTHETADT,
                        )
        if idbg == 1:
            print("\n\n")

        # Exchange borderpoints
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

        if imoist == 1 and imicrophys == 2:
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
    if idbg > 0:
        print("\nEnd of time loop ...\n")

    tt = tm()
    print("Elapsed computation time without writing: %g s\n" % (tt - t0))

    # Write output
    # ---------------------------------
    print("Start wrtiting output.\n")
    if imoist == 0:
        write_output(nout, Z, U, S, T)
    elif imicrophys == 0 or imicrophys == 1:
        if idthdt == 1:
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
                DTHETADT=DTHETADT,
            )
        else:
            write_output(
                nout, Z, U, S, T, QV=QV, QC=QC, QR=QR, PREC=PREC, TOT_PREC=TOT_PREC
            )
    elif imicrophys == 2:
        if idthdt == 1:
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
                DTHETADT=DTHETADT,
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
    t1 = tm()

    if itime == 1:
        print("Total elapsed computation time: %g s\n" % (t1 - t0))


if __name__ == '__main__':
    main()

# END OF SOLVER.PY
