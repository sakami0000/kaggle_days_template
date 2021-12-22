from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..cache import pickle_cache
from ..utils import timer


@pickle_cache()
def _base() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path("./input/water-water-everywhere-not-a-drop-to-drink/")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df


def id() -> Tuple[pd.Series, pd.Series]:
    train_df, test_df = _base()
    return train_df["id"], test_df["id"]


def result() -> Tuple[pd.Series, pd.Series]:
    """target"""
    train_df, test_df = _base()
    test_result = pd.Series([np.nan] * len(test_df), name="result")
    return train_df["result"], test_result


@timer()
@pickle_cache(overwrite=True)
def general_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = _base()
    train_df.drop(["id", "result"], axis=1, inplace=True)
    test_df.drop("id", axis=1, inplace=True)
    concat_df = pd.concat([train_df, test_df], sort=False, ignore_index=True)

    categorical_columns = [
        "categoryA",
        "categoryB",
        "categoryC",
        "categoryD",
        "categoryE",
        "categoryF",
        "unit",
    ]
    concat_df[categorical_columns] = concat_df[categorical_columns].astype("category")

    train_df = concat_df.iloc[: len(train_df)].reset_index(drop=True)
    test_df = concat_df.iloc[-len(test_df) :].reset_index(drop=True)
    return train_df, test_df
