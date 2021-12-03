from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..cache import pickle_cache
from ..utils import timer


@pickle_cache()
def _base() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path("./input/kaggle-days-tokyo/")
    train_df = pd.read_csv(data_dir / "train.csv", parse_dates=["ts"])
    test_df = pd.read_csv(data_dir / "test.csv")
    kiji_df = pd.read_csv(data_dir / "kiji_metadata.csv", parse_dates=["display_time"])
    return train_df, test_df, kiji_df


@pickle_cache()
def user_id() -> Tuple[pd.Series, pd.Series]:
    train_df, test_df, _ = _base()
    train_user = pd.Series(train_df["user_id"].unique(), name="user_id")
    test_user = pd.Series(test_df["user_id"].unique(), name="user_id")
    return train_user, test_user


@pickle_cache(overwrite=False)
def age() -> Tuple[pd.Series, pd.Series]:
    """target"""
    train_df, test_df, _ = _base()
    train_user, _ = user_id()
    train_age = (
        train_df.groupby("user_id")["age"]
        .first()
        .reindex(train_user)
        .reset_index(drop=True)
    )
    test_age = pd.Series([np.nan] * len(test_df), name="age")
    return train_age, test_age


@timer()
@pickle_cache(overwrite=True)
def general_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df, _ = _base()
    train_user, test_user = user_id()
    concat_df = pd.concat([train_df, test_df], sort=False, ignore_index=True)

    concat_feature_df = pd.DataFrame(index=concat_df["user_id"].unique())
    concat_feature_df["num_log"] = concat_df.groupby("user_id").size()
    concat_feature_df["num_unique_kiji"] = concat_df.groupby("user_id")[
        "kiji_id"
    ].nunique()

    for percentage in [80, 90, 100]:
        concat_feature_df[f"num_{percentage}per_read_unique_kiji"] = (
            concat_df.query(f"ig_ctx_red_viewed_percent >= {percentage}")
            .groupby("user_id")["kiji_id"]
            .nunique()
        )

    concat_feature_df = pd.concat(
        [
            concat_feature_df,
            (
                concat_df.groupby("user_id")["ig_ctx_red_viewed_percent"]
                .agg(["max", "mean", "min"])
                .add_prefix("red_viewed_percent_per_kiji_")
            ),
            (
                concat_df.groupby("user_id")["ig_ctx_red_elapsed_since_page_load"]
                .agg(["sum", "mean", "std"])
                .add_prefix("reading_time_")
            ),
            (
                concat_df.groupby("user_id")["er_geo_bc_flag"]
                .agg(["mean"])
                .add_prefix("bc_flag_")
            ),
        ],
        axis=1,
    )

    train_feature = concat_feature_df.reindex(train_user).reset_index(drop=True)
    test_feature = concat_feature_df.reindex(test_user).reset_index(drop=True)
    return train_feature, test_feature
