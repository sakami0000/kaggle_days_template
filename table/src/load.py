from typing import List, Tuple, Union

import pandas as pd

from . import features


def load_features(
    feature_names: Union[str, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load features"""
    if isinstance(feature_names, str):
        feature_name = feature_names
        train_feature, test_feature = features.__dict__[feature_name]()
        return train_feature, test_feature

    train_features = []
    test_features = []
    for feature_name in feature_names:
        train_feature, test_feature = features.__dict__[feature_name]()
        train_features.append(train_feature)
        if test_feature is not None:
            test_features.append(test_feature)

    train_df = pd.concat(train_features, axis=1)
    test_df = pd.concat(test_features, axis=1)
    return train_df, test_df
