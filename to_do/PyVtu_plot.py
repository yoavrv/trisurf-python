#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:05:51 2022.

Plotting of 3D surface by positions, bonds, triangles, and PyVtu.

@author: yoav
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
import vtu
import numba_numeric as nnf

# %% Functions for handling positions/blist/tlist surface


def points_filter(mask, points, blist, tlist):
    """Return points, blist, and tlist exclusivly for vertex kept by mask."""
    num = mask.sum()
    real_to_mask = np.arange(mask.shape[0])  # just an int vector size mask
    mask_to_real = np.arange(mask.shape[0])[mask]
    real_to_mask[mask_to_real] = np.arange(num)
    # mask_bonds = mask[blist].all(axis=1)  # all vtx are kept by the mask
    # old_idx_blist = blist[mask_bonds]  # remaining bonds
    # real_to_mask[old_idx_blist]  # change index all vtx to remaining vtx
    return (points[mask, ...],
            real_to_mask[blist[mask[blist].all(axis=1)]],
            real_to_mask[tlist[mask[tlist].all(axis=1)]],
            )


def points_filter_vtu(mask, v: vtu.PyVtu):
    """Run points_filter(mask, v.pos, v.blist, v.tlist)."""
    return points_filter(mask, v.pos, v.blist, v.tlist)


def neigh_mask(mask, blist, r=1):
    """Return new_mask, old_border for vertices kept by mask and neighbors."""
    n = mask.shape[0]
    s = {*np.arange(n)[mask]}
    inborder = []
    outborder = set()
    for i in range(r):
        inborder.append(set())
        outborder.clear()
        for (a, b) in blist:
            if a in s:
                if b not in s:
                    inborder[i].add(a)
                    outborder.add(b)
            elif b in s:
                inborder[i].add(b)
                outborder.add(a)
        s = s | outborder  # add next layer
    inborder.append(outborder)
    new_mask = np.array([i in s for i in range(n)])
    layers = [np.array([i in b for i in range(n)]) for b in inborder]
    boundary_mask, *layers_mask = layers
    return new_mask, boundary_mask, layers_mask


def getGyration(points):
    """Gyration matrix eigenvectors."""
    cm = points.mean(axis=0)
    Rg = (points-cm).T @ (points-cm) / len(points)
    e, ev = np.linalg.eig(Rg)
    return Rg, e, ev, cm


def blist_to_flatnan(blist):
    """Take [[v1 v2],[v3,v4]...] to [v1, v2, nan, v3, v4, nan...]."""
    return np.vstack((blist, np.ones(blist.shape[1])*np.nan)).T.flatten()


def unique1D_array_from_list(lst_lst):
    """Return numpy array of unique 1D vectors from list of list.

    Takes [[a,b,c],[x,y,z]...] and removes duplicates [a,c,b],[y,z,x]
    """
    # lst_lst = [[1,4,5],[1,5,4],[1,6,4]...]
    return np.array([[y for y in x] for x in
                     {frozenset(a) for a in lst_lst}  # {{1,4,5},{1,6,4}...}
                     ]  # [[1,4,5],[1,6,4]...]
                    )


def get_ellipsoid(points, v0=(0, 0, 0), B0=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    """Get ellipsoid from points based on least square (x-v)B(x-v)=1.

    Start guesse at center v0=[0,0,0] and B0=
    [1,0,0
     0,1,0
     0,0,0]l.
    """
    # reduction_factor = 1
    # def partial_mean(x):
    #     return x.reshape((x.shape[0]//reduction_factor, reduction_factor, *x.shape[1:])).mean(axis=1)

    def residual_ellipsoid_fit(x):
        v = x[:3]
        B = x[3:12].reshape(3, 3)
        r = points-v
        rBr = np.einsum("ij,jk,ik->i", r, B, r)
        return ((rBr-1)**2).sum()

    def residual_ellipsoid_fit2(x):
        v = x[:3]
        B = np.array([[x[3], x[4], x[5]],
                      [x[4], x[6], x[7]],
                      [x[5], x[7], x[8]]])
        r = points-v
        rBBr = np.einsum("ij,jk,kl,il->i", r, B, B, r)
        # lagrange multiplier
        alpha = x[9]
        BRsquare = np.linalg.inv(B@B).trace()
        Rsquare = np.einsum("ij,ij->i", r, r)/points.shape[0]
        constraint = alpha*(BRsquare-Rsquare)
        return 1-rBBr  + alpha*constraint

    def jacobi(x):
        J = np.empty((points.shape[0], 10))
        v = x[:3]
        B = np.array([[x[3], x[4], x[5]],
                      [x[4], x[6], x[7]],
                      [x[5], x[7], x[8]]])
        r = points-v
        dfvj = (  np.einsum("jl,lk,ik->ij",  B, B, r)
                + np.einsum("ik,kl,lj->ij", r, B, B))
        dfBjk = -(  np.einsum("ij,kl,il->ijk", r, B, r) 
                  + np.einsum("il,lj,ik->ijk", r, B, r))
        # dfvj = partial_mean(dfvj)
        # dfBjk = partial_mean(dfBjk)
        alpha = x[9]
        BRsquare = np.linalg.inv(B@B).trace()
        Rsquare = np.einsum("ij,ij->i", r, r)/points.shape[0]

        dfvj -= 2*alpha*r
        dfBjk -= 2*alpha*np.linalg.inv(B@B@B)

        J[:, 0] = dfvj[:, 0]
        J[:, 1] = dfvj[:, 1]
        J[:, 2] = dfvj[:, 2]
        J[:, 3] = dfBjk[:, 0, 0]
        J[:, 4] = dfBjk[:, 0, 1] + dfBjk[:, 1, 0]
        J[:, 5] = dfBjk[:, 0, 2] + dfBjk[:, 2, 0]
        J[:, 6] = dfBjk[:, 1, 1]
        J[:, 7] = dfBjk[:, 1, 2] + dfBjk[:, 2, 1]
        J[:, 8] = dfBjk[:, 2, 2]
        J[:, 9] = BRsquare-Rsquare
        return J

    def residual_ellipsoid_fit3(x):
        v = x[:3]
        B = np.array([[x[3], x[4], x[5]],
                      [x[4], x[6], x[7]],
                      [x[5], x[7], x[8]]])
        r = points-v
        rBr = np.einsum("ij,jk,ik->i", r, B, r)
        # return ((rBr-1)**2).sum()
        return ((1-rBr)**2).sum()

    v0 = np.array(v0).reshape(3)
    if len(B0)==3:
        B0 = np.array(B0).reshape(9)
    x0 = np.array([*v0, B0[0],B0[1],B0[2],
                              B0[4],B0[5],
                                    B0[8]])
    results = least_squares(residual_ellipsoid_fit3, x0,
                            max_nfev=1_000, verbose=1)

    x = results.x
    v0 = x[:3]
    B0 = np.array([[x[3], x[4], x[5]],
                   [x[4], x[6], x[7]],
                   [x[5], x[7], x[8]]])
    return v0, 0.5*(B0 + B0.T), results


def fit_polynom_2(points, r0=(0, 0, 0), D0=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                                         A0=(1,0,0,1)):
    """Get polynom from points based on least square z=xAx+constraint.
    where x,y,z = D(r-r0), leading to constraint DD^T - 1

    Start guesse at center r0=[0,0,0] and D0=
    [1,0,0
     0,1,0
     0,0,0]l.
    """
    # reduction_factor = 1
    # def partial_mean(x):
    #     return x.reshape((x.shape[0]//reduction_factor, reduction_factor, *x.shape[1:])).mean(axis=1)

    def residual_poly_fit(x):
        r0 = x[:3]
        D = x[3:12].reshape(3, 3)
        A = x[12:15]
        e, D_real = np.linalg.eig(D + D.T)

        x, y, z = D_real@(points-r0).T
        fi = z - 0.5*A[0]*(x**2) - A[1]*x*y - 0.5*A[2]*(y**2)

        return fi

    D0, A0 = np.array(D0).flatten(), np.array(A0).flatten()
    x0 = np.array([*r0, *D0, A0[0], A0[1] + A0[2], A0[3]])
    results = least_squares(residual_poly_fit, x0)

    x = results.x
    r0 = x[:3]
    D0 = x[3:12].reshape(3,3)
    A0 = x[12:15]

    _, D0 = np.linalg.eig(D0 + D0.T)
    return r0, D0, A0, results


def get_ellipsoid3(points, v0=(0, 0, 0), B0=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    """Get ellipsoid from points based on least square (x-v)B(x-v)=1.

    Start guesse at center v0=[0,0,0] and B0=
    [1,0,0
     0,1,0
     0,0,0]l.
    """
    # reduction_factor = 1
    # def partial_mean(x):
    #     return x.reshape((x.shape[0]//reduction_factor, reduction_factor, *x.shape[1:])).mean(axis=1)

    def residual_ellipsoid_fit(x):
        v = x[:3]
        B = np.array([[x[3], x[4], x[5]],
                      [x[4], x[6], x[7]],
                      [x[5], x[7], x[8]]])
        r = points-v
        rBBr = np.einsum("ij,jk,kl,il->i", r, B, B, r)
        # lagrange multiplier
        alpha = x[9]
        BRcube = np.linalg.inv(B@B).trace()**3 
        # want to minimize Volume, miminize vol^2
        # Rsquare = np.einsum("ij,ij->i", r, r)/points.shape[0]
        constraint = BRcube
        return 1-rBBr + alpha*constraint

    def jacobi(x):
        J = np.empty((points.shape[0], 10))
        v = x[:3]
        B = np.array([[x[3], x[4], x[5]],
                      [x[4], x[6], x[7]],
                      [x[5], x[7], x[8]]])
        r = points-v
        dfvj = (  np.einsum("jl,lk,ik->ij",  B, B, r)
                + np.einsum("ik,kl,lj->ij", r, B, B))
        dfBjk = -(  np.einsum("ij,kl,il->ijk", r, B, r) 
                  + np.einsum("il,lj,ik->ijk", r, B, r))
        # dfvj = partial_mean(dfvj)
        # dfBjk = partial_mean(dfBjk)
        alpha = x[9]
        BRsquare = np.linalg.inv(B@B).trace()
        # Rsquare = np.einsum("ij,ij->i", r, r)/points.shape[0]

        # dfvj -= 2*alpha*r
        dfBjk -= -3*alpha*np.linalg.inv(B@B@B)*(BRsquare**2)  # 2*alpha*np.linalg.inv(B@B@B)

        J[:, 0] = dfvj[:, 0]
        J[:, 1] = dfvj[:, 1]
        J[:, 2] = dfvj[:, 2]
        J[:, 3] = dfBjk[:, 0, 0]
        J[:, 4] = dfBjk[:, 0, 1] + dfBjk[:, 1, 0]
        J[:, 5] = dfBjk[:, 0, 2] + dfBjk[:, 2, 0]
        J[:, 6] = dfBjk[:, 1, 1]
        J[:, 7] = dfBjk[:, 1, 2] + dfBjk[:, 2, 1]
        J[:, 8] = dfBjk[:, 2, 2]
        J[:, 9] = -BRsquare**3
        return J

    x0 = np.array([*v0, B0[0],B0[1],B0[2],
                              B0[4],B0[5],
                                    B0[8], 1])
    results = least_squares(residual_ellipsoid_fit, x0, jac=jacobi)

    x = results.x
    v0 = x[:3]
    B0 = np.array([[x[3], x[4], x[5]],
                   [x[4], x[6], x[7]],
                   [x[5], x[7], x[8]]])
    return v0, 0.5*(B0@B0 + B0.T@B0.T), results


# %% Functions for 3D plotting of positions/blist/tlist surface


def setupAxe3D(fig=None, aspect=None, zoom=None):
    """Set up a 3D axe with resonable defaults."""
    if aspect is None:
        aspect = (1, 1, 1)
    if zoom is None:
        zoom = 1
    if fig is None:
        fig = plt.figure()
    axe = fig.add_subplot(projection='3d')
    axe.set_box_aspect(aspect, zoom=zoom)
    return fig, axe


def reaspectAxe(axe3D, points):
    """Reaspect a 3D axis based on the span of the points."""
    span = np.nanmax(points, axis=0)-np.nanmin(points, axis=0)
    return axe3D.set_box_aspect(span)


def reaspectAxe2(axe):
    """Automatic reaspcet based on span of xyz_lim."""
    xm, xx = axe.get_xlim()
    ym, yx = axe.get_ylim()
    zm, zx = axe.get_zlim()
    return axe.set_box_aspect((xx-xm, yx-ym, zx-zm), zoom=1)


def zoom_to_points(axe, points, ex=0.3):
    """Zoom axe to points with excess of +-ex (0.3-30% on either side)"""
    a, b = points.min(axis=0), points.max(axis=0)
    d = b-a
    mn, mx = a-ex*d, b+ex*d
    axe.set_xlim(mn[0], mx[0])
    axe.set_ylim(mn[1], mx[1])
    axe.set_zlim(mn[2], mx[2])
    return axe.set_box_aspect((1+2*ex)*d, zoom=1)


def append_axes(fig=None, ratio=1.68, favor_rows=True, **kwargs):
    """Append new Axes to Figure.

    Based on a stack overflow post, add one more axe to an existing figure,
    moving the rest in the grid in a semi-sensible way.
    ratio: """
    if fig is None:
        fig = plt.gcf()
    n = len(fig.axes) + 1
    a = int(np.floor((n**0.5)*ratio))
    b = n//a + (n % a != 0)
    if (a-1)*b >= n:
        a -= 1
    if a*(b-1) >= n:
        b -= 1
    if a*b < n:
        raise RuntimeError(f"yoav suck at basic math, grid {a},{b} does not ")
    nrows, ncols = (a, b) if favor_rows else (b, a)
    gs = mpl.gridspec.GridSpec(nrows, ncols, figure=fig)
    for i, ax in enumerate(fig.axes):
        ax.set_subplotspec(mpl.gridspec.SubplotSpec(gs, i))
    return fig.add_subplot(nrows, ncols, n, **kwargs)


def strip_3d_frame(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)


def plotMySurf(axe3D, *, points=None, p_dict=None, bonds=None, b_dict=None,
               triangles=None, t_dict=None, reaspect=None, **fmt_dict):
    """Plot a surface with points and bonds.

    points: nx3
    p_dict: {color: 'k', 'linestyle':'None', 'marker'='.', 'linewidth': 0.4}
    bonds: mx2
    b_dict: {'color': 'k', 'linestyle': '-',
             'marker': 'None', 'linewidth': 0.25}
    """
    if points is not None:
        if p_dict is None:
            p_dict = {'color': 'k',
                      'linestyle': 'None', 'marker': '.',
                      'linewidth': 0.2, }#'markersize': 2}
        p_dict.update(fmt_dict)
        x, y, z = np.vstack((points, np.ones(3)*np.nan)).T
        axe3D.scatter3D(x[:-1], y[:-1], z[:-1], **p_dict)
        if reaspect:
            reaspectAxe(axe3D, points)
    if bonds is not None:
        edges = np.hstack((bonds,
                           np.ones((bonds.shape[0], 1), dtype=int)*(len(x)-1))
                          ).flatten()
        if b_dict is None:
            b_dict = {'color': 'k', 'linestyle': '-',
                      'marker': 'None', 'linewidth': 0.25}
        b_dict.update(fmt_dict)
        if type(b_dict['color']) is np.ndarray:
            if b_dict['color'].size>4.:
                b_dict['color']=b_dict['color'].mean(axis=0)
        xen, yen, zen = x[edges], y[edges], z[edges]
        axe3D.plot3D(xen, yen, zen, **b_dict)


def plotGyration(axe3D, points, color='k'):
    """Plot gyration matrix eigenvectors."""
    Rg, e, ev, cm = getGyration(points)
    e = np.sqrt(3*e)  # ellipsoid axii
    x0, y0, z0 = cm
    x = np.array([x0, x0 + e[0]*ev[0, 0], np.nan,
                  x0, x0 + e[1]*ev[0, 1], np.nan,
                  x0, x0 + e[2]*ev[0, 2]
                  ])
    y = np.array([y0, y0 + e[0]*ev[1, 0], np.nan,
                  y0, y0 + e[1]*ev[1, 1], np.nan,
                  y0, y0 + e[2]*ev[1, 2]
                  ])
    z = np.array([z0, z0 + e[0]*ev[2, 0], np.nan,
                  z0, z0 + e[1]*ev[2, 1], np.nan,
                  z0, z0 + e[2]*ev[2, 2]
                  ])
    axe3D.plot3D(x, y, z, c=color)
    return x, y, z


def integrate_points(points, mask=None, ax='z'):
    """Integrate the number of points along ax=z."""
    if 'z' in ax:
        xs = points[:, 2]
    elif 'y' in ax:
        xs = points[:, 1]
    elif 'x' in ax:
        xs = points[:, 0]
    else:
        raise ValueError("invalid ax")
    sr = np.argsort(xs)
    if '-' in ax:
        sr = sr[::-1]
    x = xs[sr]
    if mask is None:
        curve = np.arange(len(x))
    else:
        curve = mask[sr].cumsum()
    # a more "proper" integration would take distance to account
    return x, curve


def plot_ellipsoid(axe, v, B, color=(1, 1, 0.3, 0.2), divisions=(50, 50),
                   **kwargs):
    """Plot ellipses (x-v)B(x-v)=1."""
    e, ev = np.linalg.eig(B)
    pos = np.zeros((3, *divisions))
    if (e>0).all():
        phi = np.linspace(0, 2*np.pi, divisions[0])
        theta = np.linspace(0, np.pi, divisions[1])
        pos[0] = (e[0]**-0.5) * np.outer(np.cos(phi), np.sin(theta))
        pos[1] = (e[1]**-0.5) * np.outer(np.sin(phi), np.sin(theta))
        pos[2] = (e[2]**-0.5) * np.outer(np.ones_like(phi), np.cos(theta))
    elif (e<0).sum()==1:
        roll_shift = np.array([2,1,0])[e<0][0] # the special axis
        e1,e2,e3 = np.abs(np.roll(e,roll_shift))
        i1,i2,i3 = np.abs(np.roll([0,1,2],roll_shift))
        phi = np.linspace(0, 2*np.pi, divisions[0])
        theta = np.linspace(-np.pi/4, np.pi/4, divisions[1])
        pos[i1] = (e1**-0.5) * np.outer(np.cos(phi), np.cosh(theta))
        pos[i2] = (e2**-0.5) * np.outer(np.sin(phi), np.cosh(theta))
        pos[i3] = (e3**-0.5) * np.outer(np.ones_like(phi), np.sinh(theta))
    elif (e<0).sum()==2:
        roll_shift = np.array([2,1,0])[e>0][0] # the special axis
        e1,e2,e3 = np.abs(np.roll(e,roll_shift))
        i1,i2,i3 = np.abs(np.roll([0,1,2],roll_shift))
        phi = np.linspace(0, 2*np.pi, divisions[0])
        theta = np.linspace(-np.pi/4, np.pi/4, divisions[1])
        pos[i1] = (e1**-0.5) * np.outer(np.cos(phi), np.sinh(theta))
        pos[i2] = (e2**-0.5) * np.outer(np.sin(phi), np.sinh(theta))
        pos[i3] = (e3**-0.5) * np.outer(np.ones_like(phi), np.sign(theta)*np.cosh(theta))
    x, y, z = np.einsum("ij,jkl->ikl", ev.T, pos)
    x, y, z = x+v[0], y+v[1], z+v[2]
    axe.plot_surface(x, y, z, color=color, **kwargs)
    return x,y,z


def plot_quadric(axe, Q, P, R, color=(1, 1, 0.3, 0.2), divisions=(50, 50),
                 scales=(-5,5,-5,5), **kwargs):
    """Plot ellipses (x-v)B(x-v)=1."""
    # try:
    #     v0 = np.linalg.inv(Q)@P
    # except np.linalg.LinAlgError as e:
    #     raise RuntimeError("bad Q! problematic surface") from e
    # q = Q/R
    # e, ev = np.linalg.eig(q)
    def func(x, y):
        square_part = Q[2,2]
        linear_part = Q[1,2]*y+Q[0,2]*x+Q[2,1]*y+Q[2,0]*x+P[2]
        constant_part = ( Q[0,0]*(x**2) + Q[1,1]*(y**2) + (Q[0,1]+Q[1,0])*x*y
                          + P[0]*x + P[1]*y + R)
        disc = linear_part**2 - 4*square_part*constant_part
        if square_part == 0:
            z = -constant_part/linear_part
            return z,z
        z_plus = (-linear_part + np.sqrt(disc))/(2*square_part)
        z_minus = (-linear_part - np.sqrt(disc))/(2*square_part)
        return z_plus, z_minus
    x = np.linspace(scales[0],scales[1],divisions[0])
    y = np.linspace(scales[2],scales[3],divisions[1])
    X,Y = np.meshgrid(x,y)
    Z = np.stack(func(X,Y))
    X, Y = np.stack((X,X)), np.stack((Y,Y))
    keep = ~(np.isnan(Z) | np.isinf(Z))
    x,y,z = X[keep],Y[keep],Z[keep]
    xy_delauny = Delaunay(np.stack((x,y)).T)
    try:
        # axe.plot(X[keep],Y[keep],Z[keep],'.', color=color, **kwargs)
        axe.plot_trisurf(x,y,xy_delauny.simplices,z,color=color,**kwargs)
    finally:
        return x,y,z


# %% separation into clusters?

def clusterize_vesicle(path):
    """Generate PyVtu from path and return clusterization."""
    v = vtu.PyVtu(path, load_all=False)
    nvtx = len(v.c0)
    type_blist = v.blist[v.type[v.blist[:, 0]] == v.type[v.blist[:, 1]], :]
    CC2, labeled_vtx = nnf.connected_components(nvtx, type_blist)
    return v, CC2, labeled_vtx


def clusterize_vesicle_bonding(path):
    """Generate PyVtu from path and return clusterization."""
    v = vtu.PyVtu(path, load_all=False)
    nvtx = len(v.c0)
    a, b = v.type[v.blist[:, 0]].astype(int), v.type[v.blist[:, 1]].astype(int)
    type_blist = v.blist[a & 1 == b & 1, :]
    CC2, labeled_vtx = nnf.connected_components(nvtx, type_blist)
    return v, CC2, labeled_vtx


def clusterize_cluster(bonds):
    blist = bonds.copy()
    vtx = np.unique(blist)
    n = vtx.max()
    idx = np.arange(len(vtx))
    vtx_idx = np.zeros(n+1, int)
    vtx_idx[vtx] = idx
    blist = vtx_idx[blist]
    CC, labeled_vtx = nnf.connected_components(len(vtx), blist)
    clusters = []
    for clst in CC:
        clusters.append([vtx[i] for i in clst])
    return clusters


def color_for(_type, lbl=None):
    """Color function for clusters."""
    c_bare = (0.9, 0.9, 0.9, 1.0)
    c_conv = plt.cm.tab20b(13)
    c_conc = plt.cm.tab20c(1)
    try:
        c = (
             (_type[:, np.newaxis] == 5)*c_conc
             + (_type[:, np.newaxis] == 47)*c_conv
             + (_type[:, np.newaxis] == 4)*c_bare
            )
    except (IndexError, TypeError):
        c = {5: c_conc, 47: c_conv}.get(_type, c_bare)
    return c
