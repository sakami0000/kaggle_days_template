import functools
import gc
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger

cached_features = {}


def pickle_cache(overwrite: bool = False):
    """Returns decorator function for caching features as pickle format.

    Parameters
    ----------
    overwrite : bool
        Whether to recreate the features.
    """

    def _pickle_cache(function: Callable):
        """Decorator function for caching features as pickle format.
        The file name will be same as the name of the decorated function.
        """

        @functools.wraps(function)
        def __pickle_cache(*args, **kwargs):
            global cached_features
            file_name = function.__name__

            if file_name in cached_features:
                result = cached_features[file_name]

                if isinstance(result, (tuple, list)):
                    result = [_result.copy() for _result in result]
                else:
                    result = result.copy()

            else:
                cache_dir = Path("./input/cache/")
                cache_dir.mkdir(exist_ok=True, parents=True)
                save_files = sorted(cache_dir.glob(f"{file_name}[0-9].pkl"))

                if len(save_files) > 0 and not overwrite:
                    result = [pd.read_pickle(save_file) for save_file in save_files]
                    logger.info(f"{file_name} loaded.")

                    if len(result) == 1:
                        result = result[0]

                else:
                    result = function(*args, **kwargs)

                    if isinstance(result, (tuple, list)):
                        for i, _result in enumerate(result):
                            _result.to_pickle(cache_dir / f"{file_name}{i}.pkl")
                    else:
                        result.to_pickle(cache_dir / f"{file_name}0.pkl")

                cached_features[file_name] = result

            gc.collect()
            return result

        return __pickle_cache

    return _pickle_cache
