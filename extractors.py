# -*- coding: utf-8 -*-
"""functions and classes to extract info from HDF stores."""
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd

import dfget



class AbstractExtractor(ABC):
    """Extract dataframe from HDF store."""

    @abstractmethod
    def extract(self, store_path) -> pd.DataFrame:
        """Extract information from a store."""
        pass


class MainExtractor(AbstractExtractor):
    """Extract (sub-) dataframe from 'main_data'."""

    def __init__(self, timeslice: slice = None, fields: List[str] = None):
        self.fields = fields
        self.timeslice = timeslice
        if fields is not None:
            self.func = dfget.gen(fields)

    def extract(self, store_path) -> Tuple[pd.DataFrame, dict]:
        """Extract main data from HDF store."""
        with pd.HDFStore(store_path, 'r') as store:
            folder_data = store["folder_data"]
            df = store["main_data"]
            if self.timeslice is not None:
                df = df[self.timeslice]
            if self.fields is not None:
                df = self.func(df)
        return df, folder_data

    def __call__(self, *args, **kwargs):
        """Extractors naturally call .extract."""
        return self.extract(*args, **kwargs)


class ClusterExtractor(AbstractExtractor):
    """Extract dataframe by concatenating df2_xxxxxx."""

    def __init__(self, typed=False, timeslice: slice = None,
                 fields: List[str] = None, write_timesteps=True):
        """do_timestep=True: add a timestep column."""
        if typed:
            self.ext_name = "df2_"
        else:
            self.ext_name = "df_"
        self.fields = fields
        self.timeslice = timeslice
        self.write_timesteps = write_timesteps
        if fields is not None:
            self.func = dfget.gen(fields)

    def extract(self, store_path) -> Tuple[pd.DataFrame, dict]:
        """Extract cluster data from HDF store: concatenate timesteps."""
        with pd.HDFStore(store_path, 'r') as store:
            folder_data = store["folder_data"]
            num_sims = len(store["main_data"])
            if self.timeslice is not None:
                rng = range(num_sims)[self.timeslice]
            else:
                rng = range(num_sims)

            def gen_store_with_time():
                for num in rng:
                    df = store[f"{self.ext_name}{num:06}"]
                    if self.fields is not None:
                        df = self.func(df)
                    if self.write_timesteps:
                        df["timestep"] = num
                    yield df

            df = pd.concat(gen_store_with_time(),
                           ignore_index=True,
                           )

        return df, folder_data

    def __call__(self, *args, **kwargs):
        """Extractors naturally call .extract."""
        return self.extract(*args, **kwargs)


class ClusterExtractorAcc(AbstractExtractor):
    """Extract dataframe by concatenating df2_xxxxxx."""

    def __init__(self, typed=False, timeslice: slice = None,
                 fields: List[str] = None):
        """do_timestep=True: add a timestep column."""
        if typed:
            self.ext_name = "df2_"
        else:
            self.ext_name = "df_"
        self.fields = fields
        self.timeslice = timeslice
        self.func = dfget.gen(fields)

    def extract(self, store_path) -> Tuple[pd.DataFrame, dict]:
        """Extract cluster data from HDF store: concatenate timesteps."""
        with pd.HDFStore(store_path, 'r') as store:
            folder_data = store["folder_data"]
            num_sims = len(store["main_data"])
            if self.timeslice is not None:
                rng = range(num_sims)[self.timeslice]
            else:
                rng = range(num_sims)

            def gen_store_with_time():
                for num in rng:
                    df = store[f"{self.ext_name}{num:06}"]
                    if self.fields is not None:
                        df = self.func(df)
                    yield df

            df = sum(gen_store_with_time())
            df /= len(rng)

        return df, folder_data

    def __call__(self, *args, **kwargs):
        """Extractors naturally call .extract."""
        return self.extract(*args, **kwargs)


class InfoExtractor(AbstractExtractor):
    """Extract meta information: done, and number."""

    def extract(self, store_path) -> Tuple[pd.DataFrame, dict]:
        """Extract done and number from store with name x####_xxxx/pystatistics.h5."""
        base_path = os.path.split(store_path)[0]
        num = int(os.path.split(base_path)[-1].split("_")[0][1:])
        with pd.HDFStore(store_path, 'r') as store:
            folder_data = store["folder_data"]
            df = store["done"]
        df["number"] = num
        return df, folder_data

    def __call__(self, *args, **kwargs):
        """Extractors naturally call .extract."""
        return self.extract(*args, **kwargs)
