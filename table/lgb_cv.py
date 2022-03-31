import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold

from src.load import load_features
from src.utils import freeze, log_evaluation, set_seed, timer, upload_to_gcs


@freeze
class config:
    features = [
        "general_features",
    ]
    id_column = "id"
    target_column = "result"
    prediction_column = "result"

    n_splits = 4
    n_models = 1

    lgb_params = {
        "objective": "rmse",
        "metric": ["rmse"],
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 16,
        "reg_lambda": 0.01,
        "reg_alpha": 0.01,
        "max_cat_threshold": 8,
        "cat_smooth": 10,
        "colsample_bytree": 0.5,
        "subsample": 0.9,
        "bagging_freq": 1,
        "seed": 72,
    }
    lgb_kwargs = {
        "num_boost_round": 100000,
    }
    early_stopping_rounds = 200

    bucket_name = "kaggledays_championship"
    bucket_path = "sakami/lightgbm/"  # CHECK HERE!!!

    seed = 1029


def generate_split(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    train_fold = pd.read_csv("./input/train_fold.csv")

    if train_fold["fold"].nunique() != config.n_splits:
        raise ValueError("The number of splits in CSV and config must be the same.")

    if len(train_fold) != len(X):
        train_fold = train_fold.sample(len(X), random_state=config.seed)

    for i in range(config.n_splits):
        train_idx = np.where(train_fold["fold"] != i)[0]
        valid_idx = np.where(train_fold["fold"] == i)[0]
        yield train_idx, valid_idx


def main(debug: bool = False):
    start_time = time.time()
    set_seed(config.seed)

    # features
    with timer("load features"):
        other_columns = [config.id_column, config.target_column]
        train_df, test_df = load_features(other_columns + config.features)

    if debug:
        train_df = train_df.sample(100, random_state=config.seed).reset_index(drop=True)
        test_df = test_df.sample(100, random_state=config.seed).reset_index(drop=True)

    logger.info(f"train shape : {train_df.shape}")
    logger.info(f"test shape  : {test_df.shape}")
    test_x = test_df.drop(other_columns, axis=1)

    # train
    with timer("train"):
        valid_preds = np.zeros((len(train_df),), dtype=np.float64)
        test_preds = np.zeros((len(test_df),), dtype=np.float64)
        cv_scores = []

        for fold, (train_idx, valid_idx) in enumerate(generate_split(train_df)):
            logger.info("-" * 40)
            logger.info(f"fold {fold +  1}")

            train_x = train_df.iloc[train_idx].drop(other_columns, axis=1)
            valid_x = train_df.iloc[valid_idx].drop(other_columns, axis=1)

            train_y = train_df[config.target_column].iloc[train_idx].values
            valid_y = train_df[config.target_column].iloc[valid_idx].values

            for i in range(config.n_models):
                logger.info(f"model {i + 1}")
                lgb_params = config.lgb_params.copy()
                lgb_kwargs = config.lgb_kwargs.copy()
                lgb_params["seed"] += i

                if debug:
                    lgb_kwargs["num_boost_round"] = 10

                dtrain = lgb.Dataset(train_x, train_y)
                dvalid = lgb.Dataset(valid_x, valid_y)

                period = lgb_kwargs.get("verbose_eval", 50)
                callbacks = [
                    lgb.early_stopping(config.early_stopping_rounds),
                    log_evaluation(period=period),
                ]

                model = lgb.train(
                    params=lgb_params,
                    train_set=dtrain,
                    valid_sets=[dtrain, dvalid],
                    valid_names=["train", "valid"],
                    callbacks=callbacks,
                    **lgb_kwargs,
                )

                valid_preds[valid_idx] = model.predict(valid_x) / config.n_models
                test_preds += model.predict(test_x) / config.n_splits / config.n_models
                cv_scores.append(model.best_score["valid"][lgb_params["metric"][0]])

                if debug:
                    break

            if debug:
                break

    # save
    save_dir = Path("./output/")
    save_dir.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            config.id_column: train_df[config.id_column],
            config.prediction_column: valid_preds,
        }
    ).to_csv(save_dir / "valid.csv", index=False)
    pd.DataFrame(
        {
            config.id_column: test_df[config.id_column],
            config.prediction_column: test_preds,
        }
    ).to_csv(save_dir / "test.csv", index=False)

    # upload to GCS
    if not debug:
        upload_to_gcs(
            save_dir, bucket_name=config.bucket_name, gcs_prefix=config.bucket_path
        )

    # best iteration
    logger.info("-" * 40)
    logger.info(f"best iteration : {model.best_iteration}")

    # best score
    logger.info("-" * 40)
    for data_name, score_dict in model.best_score.items():
        for score_name, score in score_dict.items():
            logger.info(f"{data_name}_{score_name} : {score:.5f}")

    # feature importance
    logger.info("-" * 40)
    logger.info("feature importance:")
    feature_importance = model.feature_importance(importance_type="gain")
    for i in np.argsort(feature_importance):
        logger.info(f"\t{model.feature_name()[i]:35s} : {feature_importance[i]:.2f}")

    cv_score = np.mean(cv_scores)
    logger.info(f"cv score: {cv_score:.5f}")
    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"all processes done in {elapsed_time:.1f} min.")


if __name__ == "__main__":
    # debug
    logger.info("********************** mode : debug **********************")
    main(debug=True)

    logger.info("-" * 40)
    logger.info("********************** mode : main **********************")
    main(debug=False)
