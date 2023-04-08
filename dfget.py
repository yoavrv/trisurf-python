# -*- coding: utf-8 -*-
"""Dataframe series extraction df['a']=dfget.a(df)."""
from functools import partial
from typing import Tuple
import numpy as np
import pandas as pd



true_fields = ['volume', 'area', 'radius', 'lambda1', 'lambda2', 'lambda3',
               'bond_ratio', 'mean_h', 'perim', 'mean_cluster_size',
               'std_cluster_size', 'force_per_vertex']

true_fields_clst = ['type', 'cluster_size', 'bending_energy',
                    'lambda1', 'lambda2', 'lambda3', 'perim',
                    'fx', 'fy', 'fz', 'id']

get_fields = {'gyration_radius': ['lambda1', 'lambda2', 'lambda3'],
              'asphericity': ['lambda1', 'lambda2', 'lambda3'],
              'asphericity2': ['lambda1', 'lambda2', 'lambda3'],
              'acylindricity': ['lambda1', 'lambda2', 'lambda3'],
              'bounding_v_frac': ['volume', 'radius'],
              'discosity': ['lambda2', 'lambda3'],
              'adiscosity': ['lambda2', 'lambda3'],
              'force': ['fx', 'fy', 'fz'],
              'ellipsoidic_v_frac': ['lambda1', 'lambda2', 'lambda3', 'volume'],
              }


titles = {'volume': 'volume',
          'area': 'area',
          'radius': 'radius',
          'lambda1': r'$\lambda_1^2$',
          'lambda2': r'$\lambda_2^2$',
          'lambda3': r'$\lambda_3^2$',
          'bond_ratio': 'bond ratio',
          'mean_h': 'mean curvature',
          'perim': 'boundary perimeter',
          'mean_cluster_size': r'$\langle N \rangle$',
          'std_cluster_size': r'$\sigma_N$',
          'force_per_vertex': 'average force per vertex',
          'gyration_radius': r'$R_g$',
          'asphericity': 'asphericity (unnormalized)',
          'asphericity2': 'asphericity',
          'acylindricity': 'acylindricity',
          'bounding_v_frac': r'$\frac{V}{V_{sphere}}$',
          'discosity:': r'discosity',
          'ellipsoidic_v_frac': '$\\frac{V}{V_\\epsilon}$'
          }


def gyration_radius(df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe gyration radius."""
    return df["lambda1"] + df["lambda2"] + df["lambda3"]


def asphericity(df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe asphericity."""
    return df["lambda3"] - 0.5 * df["lambda2"] - 0.5 * df["lambda1"]


def normalized_asphericity(df):
    """Dataframe asphericity normalized by gyration."""
    return asphericity(df)/gyration_radius(df)


def asphericity2(df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe asphericity."""
    return ((df["lambda1"] - df["lambda2"]) ** 2
            + (df["lambda2"] - df["lambda3"]) ** 2
            + (df["lambda3"] - df["lambda1"]) ** 2
            ) / (2 * ((df["lambda1"] + df["lambda2"] + df["lambda3"]) ** 2))


def acylindricity(df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe acylindricity."""
    return df["lambda2"] - df["lambda1"]


def discosity(df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe discosity: radius of "disc" assuming l1<<l2~l3."""
    return df["lambda2"]+df["lambda3"]


def adiscosity(df: pd.DataFrame) -> pd.DataFrame:
    """Dataframe adiscosity: difference of the two largest direction."""
    return df["lambda3"]-df["lambda2"]


def bounding_v_frac(df: pd.DataFrame) -> pd.DataFrame:
    """Get ratio of volume to volume of bounding sphere."""
    return df["volume"] / ((4 * np.pi / 3) * df["radius"] ** 3)


def ellipsoidic_v_frac(df: pd.DataFrame) -> pd.DataFrame:
    """Get ratio of volume to volume of ellipsoid based on gyration tensor."""
    return (df['volume']
            /
            ((4*np.pi/3) * np.sqrt(
                 3*df['lambda1']
                *3*df['lambda2']
                *3*df['lambda3'])
                )
            )


def force_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """Get force on cluster."""
    return (df['fx']**2 + df['fy']**2 + df['fz']**2)**0.5


def mean_field_per_vertex(df: pd.DataFrame, field) -> pd.DataFrame:
    """Cluster size per vertex, so [99,1] have <N>_v=98.02 and not 50."""
    return (df[field] * df["cluster_size"] / (df["cluster_size"].sum())).sum()


def mean_of_time(df: pd.DataFrame) -> pd.DataFrame:
    """Get mean over time."""
    return df.groupby('timesteps').mean().mean()



def canonize_fields(stuff):
    """Transform to canonical form for extractors.Extractor."""
    if type(stuff) is str:
        string = stuff.replace(' ', '_').lower()
        if string in {'all', 'all_fields'}:
            return true_fields
        if string in {'asphericity2', 'normalized_asphericity',
                      'relative_anisotropy', 'anisotropy'}:
            string = 'asphericity2'
        return string
    if type(stuff) is list:
        return [canonize_fields(x) for x in stuff]
    else:
        return stuff


def fields_to_extract(fields: list):
    """Fields to extract (area, lambda1, asphericity)->(volume, lambda123)."""
    if fields is None:
        return None
    _fields = set()
    if type(fields) is not str:
        for field in fields:
            if field in true_fields:
                _fields.add(field)
            elif field in get_fields:
                _fields.update(get_fields[field])
    else:
        if fields in true_fields:
            _fields.add(fields)
        elif fields in get_fields:
            _fields.update(get_fields[fields])
    return list(_fields)


def from_strings(fields: list) -> Tuple[list, list]:
    """Find all the fields needed to load and fields to create."""
    _fields = set()
    _gets = set()
    if type(fields) is not str:
        for field in fields:
            if field in true_fields:
                _fields.add(field)
            elif field in true_fields_clst:
                _fields.add(field)
            elif field in get_fields:
                _gets.add(field)
                _fields.update(get_fields[field])
    else:
        if fields in true_fields:
            _fields.add(fields)
        elif fields in true_fields_clst:
            _fields.add(fields)
        elif fields in get_fields:
            _gets.add(fields)
            _fields.update(get_fields[fields])
    any_in_clst = any(field not in true_fields for field in _fields)
    any_in_main = any(field not in true_fields_clst for field in _fields)
    if any_in_clst and any_in_main:
        raise ValueError(f'{fields} contains fields of "cluster" and "main"')
    return list(_gets), list(_fields)


def from_strings2(df: pd.DataFrame, gets: list, fields: list) -> pd.DataFrame:
    """Load all fields and add all calculation fields."""
    if fields is None:
        return df
    if any(x not in df.columns for x in fields):
        raise IndexError(f'{fields} not all in columns of df {df.columns}')
    df = df.loc[:, fields]
    for g in gets:
        if g == 'gyration_radius':
            df['gyration_radius'] = gyration_radius(df)
        elif g == 'asphericity':
            df['asphericity'] = asphericity(df)
        elif g == 'asphericity2':
            df['asphericity2'] = asphericity2(df)
        elif g == 'acylindricity':
            df['acylindricity'] = acylindricity(df)
        elif g == 'discosity':
            df['discosity'] = discosity(df)
        elif g == 'adiscosity':
            df['adiscosity'] = adiscosity(df)
        elif g == 'bounding_v_frac':
            df['bounding_v_frac'] = bounding_v_frac(df)
        elif g == 'ellipsoidic_v_frac':
            df['ellipsoidic_v_frac'] = ellipsoidic_v_frac(df)
        elif g == 'force':
            df['force'] = (df['fx']**2 + df['fy']**2 + df['fz']**2)**0.5
    return df


def gen(fields):
    """Generate a getter f=gen('a','b'); f(df)->df[['a','b']]."""
    _fields = canonize_fields(fields)
    gets, _fields = from_strings(_fields)
    return partial(from_strings2, gets=gets, fields=_fields)
