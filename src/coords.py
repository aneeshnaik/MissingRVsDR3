#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate conversion: ra / dec / distance -> (X Y Z) or (R phi Z).

Created: April 2022
Author: A. P. Naik
"""
import numpy as np
from .constants import PI, D_GC, Z_SUN, RA_GC, DEC_GC, ROLL,\
    U_SUN, V_SUN, W_SUN, KM_PER_KPC, RADS_PER_MAS, SECS_PER_YEAR


def convert_pos(ra, dec, d, rep='cartesian',
                d_GC=D_GC, z0=Z_SUN, ra_GC=RA_GC, dec_GC=DEC_GC, eta=ROLL):
    r"""
    Convert (ICRS) equatorial coordinates to galactocentric XYZ (or R phi Z).

    Given coordinates representing RAs and DECs, Cartesian (or cylindrical)
    galactocentric coordinates are calculated. The final coordinate system is a
    right-handed system (like astropy), in which the sun is approximately at
    (-8, 0, 0) kpc, and the direction of the local rotation is in the positive
    y-direction. See 'Notes' below for mathematical details of the
    transformation. Note the units below!

    Parameters
    ----------
    ra : scalar or array, shape (N)
        Right ascensions. UNITS: degrees.
    dec : scalar or array, shape (N)
        Declinations. UNITS: degrees.
    d : scalar or array, shape (N)
        Distances. UNITS: kpc.
    rep : string, ['cartesian', 'cylindrical'], optional
        Whether to output positions in Cartesian or cylindrical polar
        representation. The default is Cartesian.
    d_GC : float, optional
        Distance from Sun to Galactic Centre. UNITS: kpc. The default is
        8.122.
    z0 : float, optional
        Distance of Sun above mid-plane. UNITS: kpc. The default is 0.0208.
    ra_GC : float, optional
        RA of Galactic Centre. UNITS: degrees. The default is 266.4051.
    dec_GC : float, optional
        DEC of Galactic Centre. UNITS: degrees. The default is -28.936175.
    eta : float, optional
        'Roll' angle. UNITS: degrees. The default is 58.5986320306.

    Returns
    -------
    r : array, shape (N, 3) or (3)
        Galactocentric Cartesian (or cylindrical) positions, i.e. (X, Y, Z) or
        (R, phi, Z). UNITS: kpc (degrees for phi in the cylindrical case).

    Notes
    -----
    The transformation begins by converting a given (distance, RA, DEC) trio
    into a Cartesian position vector :math:`r_{eq}`. This vector can then be
    transformed into galactocentric coordinates :math:`r_{g}` via the formula:

    .. math:: r_{g} = H (R r_{eq} - (d_{GC},0,0)^T)

    Breaking this down, the transformation has three stages. First, the
    operator :math:`R` rotates the coordinate system to place the Galactic
    centre (GC) on the x-axis, and the north Galactic pole on the z-axis.
    :math:`R=R_{3}R_{1}R_{2}` is itself composed of three individual
    rotations:

    .. math::
        R_1 = \begin{pmatrix}
                \cos\delta_{GC}  & 0 & \sin\delta_{GC} \\
                0                & 1 & 0               \\
                -\sin\delta_{GC} & 0 & \cos\delta_{GC}
            \end{pmatrix}

    .. math::
        R_2 = \begin{pmatrix}
                \cos\alpha_{GC}  & \sin\alpha_{GC} & 0 \\
                -\sin\alpha_{GC} & \cos\alpha_{GC} & 0 \\
                0                 & 0                & 1
            \end{pmatrix}

    .. math::
        R_3 = \begin{pmatrix}
                1 & 0         & 0        \\
                0 & \cos\eta  & \sin\eta \\
                0 & -\sin\eta & \cos\eta
            \end{pmatrix}

    :math:`R_{1}` and :math:`R_{2}` re-orient the x-axis to align with the
    Sun-GC line (:math:`\alpha_{GC},\delta_{GC}` are the RA,DEC of the
    Galactic centre). Then, :math:`R_{3}` rotates the y-z plane so that the
    z-axis points towards the north Galactic pole. The angle :math:`\eta` is
    known as the 'roll'. The default value used here is around 59 degrees.

    Second, the origin is translated to the Galactic centre via the
    :math:`(d_{GC},0,0)^T` term. Here, :math:`d_{GC}` is the Sun-GC distance.

    Finally, the operator :math:`H` re-orients the x-z plane to account for the
    position of the Sun above the Galactic mid-plane. In terms of the angle
    :math:`\theta\equiv\sin^{-1}(z_0/d_{GC})`, :math:`H` is given by:

    .. math::
        H = \begin{pmatrix}
                \cos\theta  & 0 & \sin\theta \\
                0           & 1 & 0          \\
                -\sin\theta & 0 & \cos\theta
            \end{pmatrix}

    """
    # if ra, dec, d just scalars, convert to arrays length one
    scalar = False
    if not hasattr(ra, "__len__"):
        scalar = True
        ra = np.array([ra])
        dec = np.array([dec])
        d = np.array([d])

    # convert angles to radians
    ra_rads = ra * PI / 180
    dec_rads = dec * PI / 180
    ra_GC *= PI / 180
    dec_GC *= PI / 180
    eta *= PI / 180

    # create shorthands for various trig functions
    cga = np.cos(ra_GC)
    sga = np.sin(ra_GC)
    cgd = np.cos(dec_GC)
    sgd = np.sin(dec_GC)
    ce = np.cos(eta)
    se = np.sin(eta)
    ct = np.sqrt(1 - z0**2 / d_GC**2)
    st = z0 / d_GC

    # rotation matrices
    R1 = np.array([[ cgd, 0, sgd],
                   [  0,  1,   0],
                   [-sgd, 0, cgd]], dtype=ra.dtype)
    R2 = np.array([[ cga, sga,  0],
                   [-sga, cga,  0],
                   [   0,   0,  1]], dtype=ra.dtype)
    R3 = np.array([[  1,   0,  0],
                   [  0,  ce, se],
                   [  0, -se, ce]], dtype=ra.dtype)
    H = np.array([[ ct,   0, st],
                  [  0,   1,  0],
                  [-st,   0, ct]], dtype=ra.dtype)
    R = np.matmul(R3, np.matmul(R1, R2))

    # convert spherical to Cartesian
    x = d * np.cos(ra_rads) * np.cos(dec_rads)
    y = d * np.sin(ra_rads) * np.cos(dec_rads)
    z = d * np.sin(dec_rads)
    r = np.stack((x, y, z), axis=1)

    # perform transformation
    r = np.matmul(R, r.T).T - np.array([d_GC, 0, 0], dtype=ra.dtype)
    r = np.matmul(H, r.T).T

    # convert to cylindricals if requested
    if rep == 'cylindrical':
        R = np.linalg.norm(r[:, :-1], axis=-1)
        phi = np.arctan2(r[:, 1], r[:, 0])
        phi[phi < 0] = phi[phi < 0] + 2 * PI
        phi = phi * 180 / PI
        r = np.stack((R, phi, r[:, 2]), axis=1)

    # if inputs were scalars, change shape of r (1, 3) -> (3)
    if scalar:
        r = r[0]
    return r


def convert_posvel(ra, dec, d, pmra, pmdec, v_rad,
                   rep='cartesian',
                   v_sun=np.array([U_SUN, V_SUN, W_SUN]),
                   d_GC=D_GC, z0=Z_SUN, ra_GC=RA_GC, dec_GC=DEC_GC, eta=ROLL):
    r"""
    Convert 6D equatorial data into into galactocentric coords.

    Given arrays of coordinates representing RAs, DECs, distances, proper
    motions and radial velocities, Cartesian (or cylindrical) galactocentric
    positions and velocities are calculated. The final coordinate system is a
    right-handed system (like astropy), in which the sun is approximately at
    (-8, 0, 0) kpc, and the direction of the local rotation is in the positive
    y-direction. See 'Notes' below for mathematical details of the
    transformation. Note the units below!

    Parameters
    ----------
    ra : scalar or array, shape (N)
        Right ascensions. UNITS: degrees.
    dec : scalar or array, shape (N)
        Declinations. UNITS: degrees.
    d : scalar or array, shape (N)
        Distances. UNITS: kpc.
    pmra : numpy array, shape (N)
        Proper motion in right ascension
        :math:`\mu_{\alpha\*}≡\mu_\alpha\cos\delta`. UNITS: mas/yr.
    pmdec : numpy array, shape (N)
        Proper motion in declination. UNITS: mas/yr.
    v_rad : numpy array, shape (N)
        Radial velocities. UNITS: km/s.
    rep : string, ['cartesian', 'cylindrical'], optional
        Whether to output velocities in Cartesian or cylindrical polar
    v_sun : numpy array, shape (3), optional
        Velocity of Sun in (right-handed) galactocentric frame. UNITS: km/s.
        The default is [12.9, 245.6, 7.78].
    d_GC : float, optional
        Distance from Sun to Galactic Centre. UNITS: kpc. The default is
        8.122.
    z0 : float, optional
        Distance of Sun above mid-plane. UNITS: kpc. The default is 0.0208.
    ra_GC : float, optional
        RA of Galactic Centre. UNITS: degrees. The default is 266.4051.
    dec_GC : float, optional
        DEC of Galactic Centre. UNITS: degrees. The default is -28.936175.
    eta : float, optional
        'Roll' angle. UNITS: degrees. The default is 58.5986320306.

    Returns
    -------
    r : numpy array, shape (N, 3)
        Galactocentric Cartesian (or cylindrical) positions, i.e. (X, Y, Z) or
        (R, phi, Z). UNITS: kpd (radians for phi in the cylindrical case).
    v : numpy array, shape (N, 3)
        Velocities in galactocentric frame, either (VX, VY, VZ) or
        (VR, VPHI, VZ). UNITS: km/s.

    Notes
    -----
    This section documents the velocity transformation. For details about the
    position transformation, see the documentation for `convert_pos`.
    The transformation begins by converting proper motions into velocities in
    the local tangent plane. These velocities are then converted into
    Cartesians via :math:`(v_x, v_y, v_z)^T = A (v_\alpha, v_\delta, v_r)^T`.
    A is the rotation matrix:

    .. math::
        A = \begin{pmatrix}
                -\sin\alpha & -\cos\alpha\sin\delta & \cos\alpha\cos\delta \\
                \cos\alpha  & -\sin\alpha\sin\delta & \sin\alpha\cos\delta \\
                0           & \cos\delta            & \sin\delta
            \end{pmatrix}

    These Cartesian velocities are then converted into the galactocentric frame
    via the transformation

    .. math:: v_{g} = H R v_{eq} + v_\odot

    Breaking this down, the transformation has three stages. First, the
    operator :math:`R` rotates the coordinate system to place the Galactic
    centre (GC) on the x-axis, and the north Galactic pole on the z-axis.
    :math:`R=R_{3}R_{1}R_{2}` is itself composed of three individual
    rotations:

    .. math::
        R_1 = \begin{pmatrix}
                \cos\delta_{GC}  & 0 & \sin\delta_{GC} \\
                0                & 1 & 0               \\
                -\sin\delta_{GC} & 0 & \cos\delta_{GC}
            \end{pmatrix}

    .. math::
        R_2 = \begin{pmatrix}
                \cos\alpha_{GC}  & \sin\alpha_{GC} & 0 \\
                -\sin\alpha_{GC} & \cos\alpha_{GC} & 0 \\
                0                 & 0                & 1
            \end{pmatrix}

    .. math::
        R_3 = \begin{pmatrix}
                1 & 0         & 0        \\
                0 & \cos\eta  & \sin\eta \\
                0 & -\sin\eta & \cos\eta
            \end{pmatrix}

    :math:`R_{1}` and :math:`R_{2}` re-orient the x-axis to align with the
    Sun-GC line (:math:`\alpha_{GC},\delta_{GC}` are the RA,DEC of the
    Galactic centre). Then, :math:`R_{3}` rotates the y-z plane so that the
    z-axis points towards the north Galactic pole. The angle :math:`\eta` is
    known as the 'roll'. The default value used here is around 59 degrees.

    Second, the operator :math:`H` re-orients the x-z plane to account for the
    position of the Sun above the Galactic mid-plane. In terms of the angle
    :math:`\theta\equiv\sin^{-1}(z_0/d_{GC})`, :math:`H` is given by:

    .. math::
        H = \begin{pmatrix}
                \cos\theta  & 0 & \sin\theta \\
                0           & 1 & 0          \\
                -\sin\theta & 0 & \cos\theta
            \end{pmatrix}

    Finally, there is a Galilean transform :math:`v \rightarrow v+v_\odot` to
    correct for the motion of the Sun.
    """
    # if inputs just scalars, convert to arrays length one
    scalar = False
    if not hasattr(ra, "__len__"):
        scalar = True
        ra = np.array([ra])
        dec = np.array([dec])
        d = np.array([d])
        pmra = np.array([pmra])
        pmdec = np.array([pmdec])
        v_rad = np.array([v_rad])

    # convert angles to radians
    ra_rads = ra * PI / 180
    dec_rads = dec * PI / 180
    ra_GC *= PI / 180
    dec_GC *= PI / 180
    eta *= PI / 180

    # convert proper motions to velocities (in km/s)
    u = KM_PER_KPC * RADS_PER_MAS / SECS_PER_YEAR
    v_dec = d * pmdec * u
    v_ra = d * pmra * u

    # create shorthands for trig functions
    ca = np.cos(ra_rads)
    sa = np.sin(ra_rads)
    cd = np.cos(dec_rads)
    sd = np.sin(dec_rads)
    cga = np.cos(ra_GC)
    sga = np.sin(ra_GC)
    cgd = np.cos(dec_GC)
    sgd = np.sin(dec_GC)
    ce = np.cos(eta)
    se = np.sin(eta)
    ct = np.sqrt(1 - z0**2 / d_GC**2)
    st = z0 / d_GC

    # convert to Cartesians
    x = d * ca * cd
    y = d * sa * cd
    z = d * sd
    r = np.stack((x, y, z), axis=1)
    v_x = -v_ra * sa - v_dec * ca * sd + v_rad * ca * cd
    v_y = v_ra * ca - v_dec * sa * sd + v_rad * sa * cd
    v_z = v_dec * cd + v_rad * sd
    v = np.stack((v_x, v_y, v_z), axis=1)

    # rotation matrices
    R1 = np.array([[ cgd, 0, sgd],
                   [  0,  1,   0],
                   [-sgd, 0, cgd]])
    R2 = np.array([[ cga, sga,  0],
                   [-sga, cga,  0],
                   [   0,   0,  1]])
    R3 = np.array([[  1,   0,  0],
                   [  0,  ce, se],
                   [  0, -se, ce]])
    H = np.array([[ ct,   0, st],
                  [  0,   1,  0],
                  [-st,   0, ct]])
    R = np.matmul(R3, np.matmul(R1, R2))

    # perform transformation
    r = np.matmul(R, r.T).T - np.array([d_GC, 0, 0])
    r = np.matmul(H, r.T).T
    v = np.matmul(R, v.T).T
    v = np.matmul(H, v.T).T + v_sun

    # convert to cylindricals if requested
    if rep == 'cylindrical':
        R = np.linalg.norm(r[:, :-1], axis=-1)
        phi = np.arctan2(r[:, 1], r[:, 0])
        phi[phi < 0] = phi[phi < 0] + 2 * PI
        r = np.stack((R, phi, r[:, 2]), axis=1)
        vR = v[:, 0] * np.cos(phi) + v[:, 1] * np.sin(phi)
        vphi = -v[:, 0] * np.sin(phi) + v[:, 1] * np.cos(phi)
        v = np.stack((vR, vphi, v[:, 2]), axis=1)

    # if inputs were scalars, change shape of r, v (1, 3) -> (3)
    if scalar:
        r = r[0]
        v = v[0]
    return r, v
