from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from src.utils import set_seed, timer, freeze, download_from_gcs, upload_to_gcs


@freeze
class config:
    id_column = "id"
    target_column = "target"
    prediction_column = "prediction"

    model_paths = [
        "sakami/bert/output/",
        "sakami/lstm/output/",
    ]
    bucket_name = "kaggledays_championship"
    bucket_path = "sakami/nelder_mead/"  # CHECK HERE!!!

    seed = 1029


@timer()
def download_models(model_paths: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_dir = Path("./input/models/")
    models = {"valid": {}, "test": {}}
    for model_path in model_paths:
        save_dir = data_dir / model_path
        if not save_dir.exists():
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            download_from_gcs(
                save_dir.parent, bucket_name=config.bucket_name, gcs_prefix=model_path
            )
        models["valid"][model_path] = pd.read_csv(save_dir / "valid.csv")
        models["test"][model_path] = pd.read_csv(save_dir / "test.csv")
    return models


def extract_preds(
    models: Dict[str, pd.DataFrame], id_column: str, prediction_column: str
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    idx = None
    preds = {}
    for model_name, df in models.items():
        if idx is None:
            idx_model_name = model_name
            idx = df[id_column].values
        else:
            if set(df[id_column].values) != set(idx):
                raise KeyError(
                    f"Indices don't match for {idx_model_name} and {model_name}."
                )
            df = df.set_index(id_column).reindex(idx).reset_index()
        preds[model_name] = df[prediction_column].values
    return idx, preds


def load_data() -> pd.DataFrame:
    data_dir = Path("./input/jigsaw-unintended-bias-in-toxicity-classification/")
    train_df = pd.read_csv(data_dir / "train.csv")
    return train_df


class Optimizer:
    def __init__(
        self,
        y_true: np.ndarray,
        valid_preds: Dict[str, np.ndarray],
        x0: Optional[List[float]] = None,
    ):
        if x0 is None:
            x0 = [1.0 / len(valid_preds)] * len(valid_preds)

        self.y_true = y_true
        self.x0 = x0

        self.model_names = list(valid_preds.keys())
        self.valid_preds = list(valid_preds.values())
        self.result = None

    def objective(self, weights: List[float]) -> float:
        preds = np.average(self.valid_preds, axis=0, weights=weights)
        score = roc_auc_score(self.y_true, preds)
        return -score

    def predict(self, test_preds: Dict[str, np.ndarray]) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Run `self.optimize()` first.")

        preds = np.average(
            [test_preds[model_name] for model_name in self.model_names],
            axis=0,
            weights=self.weights,
        )
        return preds

    def optimize(self):
        logger.info("Optimizing ensemble weights...")
        self.result = minimize(self.objective, self.x0, method="nelder-mead")

        logger.info(self.result.message)
        if self.result.success:
            logger.info(f"\tCurrent function value: {-self.result.fun:.6f}")
            logger.info(f"\tIterations: {self.result.nit}")
            logger.info(f"\tFunction evaluations: {self.result.nfev}")

            self.weights = np.array(self.result.x)
            self.weights /= self.weights.sum()

            logger.info("-" * 40)
            logger.info("Optimized weights:")
            for model_name, weight in zip(self.model_names, self.weights):
                logger.info(f"\t{model_name:30s} : {weight:.5f}")


@timer()
def main():
    set_seed(config.seed)

    logger.info("optimize weights for the following models:")
    for model_path in config.model_paths:
        logger.info(f"  {model_path}")

    logger.info("-" * 40)
    models = download_models(config.model_paths)
    valid_idx, valid_preds = extract_preds(
        models["valid"],
        id_column=config.id_column,
        prediction_column=config.prediction_column,
    )
    test_idx, test_preds = extract_preds(
        models["test"],
        id_column=config.id_column,
        prediction_column=config.prediction_column,
    )

    train_df = load_data()
    y_true = (
        train_df.set_index(config.id_column)
        .reindex(valid_idx)[config.target_column]
        .values
    )

    x0 = [1.0 / len(config.model_paths)] * len(config.model_paths)
    optimizer = Optimizer(y_true, valid_preds, x0=x0)
    optimizer.optimize()
    optimized_test_preds = optimizer.predict(test_preds)

    save_dir = Path("./output/")
    save_dir.mkdir(exist_ok=True)
    pd.DataFrame(
        {config.id_column: test_idx, config.prediction_column: optimized_test_preds}
    ).to_csv(save_dir / "submission.csv", index=False)
    upload_to_gcs(
        save_dir, bucket_name=config.bucket_name, gcs_prefix=config.bucket_path
    )


if __name__ == "__main__":
    main()
