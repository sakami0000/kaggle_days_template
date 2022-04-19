import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.nn.modules.activation import PReLU
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.load import load_features
from src.utils import freeze, set_seed, timer, upload_to_gcs


@freeze
class config:
    features = [
        "general_features",
    ]
    categorical_columns = [
        "categoryA",
        "categoryB",
        "categoryC",
        "categoryD",
        "categoryE",
        "categoryF",
        "unit",
    ]
    id_column = "id"
    target_column = "result"
    prediction_column = "result"

    n_splits = 4
    n_epochs = 30

    lr = 3e-3
    eta_min = 1e-4

    batch_size = 64
    eval_batch_size = 128

    bucket_name = "kaggledays_championship"
    bucket_path = "sakami/nn/"  # CHECK HERE!!!

    device = torch.device("cuda")
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


class WaterDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        categorical_columns: List[str],
        target: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
    ):
        self.df = df
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target

        if indices is None:
            indices = np.arange(len(self.df))

        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        index = self.indices[index]
        num_x = torch.FloatTensor(self.df[self.numerical_columns].values[index])
        cat_x = torch.LongTensor(self.df[self.categorical_columns].values[index])

        if self.target is not None:
            target = self.target[index]
            return num_x, cat_x, target

        return num_x, cat_x


class SEScale(nn.Module):
    def __init__(self, ch: int, r: int):
        super().__init__()
        self.fc1 = nn.Linear(ch, r)
        self.fc2 = nn.Linear(r, ch)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h).sigmoid()
        return h * x


class GRULayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.unsqueeze(1)
        x, _ = self.gru(x)
        x = x.squeeze(1)
        return x


class WaterModel(nn.Module):
    def __init__(self, numerical_input_size: int, category_cardinalities: List[int]):
        super().__init__()

        self.numerical_linear = nn.Sequential(
            SEScale(numerical_input_size, 64),
            nn.Linear(numerical_input_size, 128),
            nn.Dropout(0.1),
            nn.PReLU(),
        )

        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(n_cat, min(600, round(1.6 * n_cat ** 0.56))) for n_cat in category_cardinalities]
        )
        category_dimensions = [embedding_layer.embedding_dim for embedding_layer in self.category_embeddings]
        self.category_linear = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(sum(category_dimensions), 128),
            nn.Dropout(0.1),
            nn.PReLU(),
        )

        self.head = nn.Sequential(
            SEScale(256, 128),
            GRULayer(256, 128),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, num_x: torch.FloatTensor, cat_x: torch.LongTensor) -> torch.FloatTensor:
        num_x = self.numerical_linear(num_x)

        cat_x = [embedding_layer(cat_x[:, i]) for i, embedding_layer in enumerate(self.category_embeddings)]
        cat_x = torch.cat(cat_x, dim=-1)
        cat_x = self.category_linear(cat_x)

        x = torch.cat([num_x, cat_x], dim=-1)
        output = self.head(x).squeeze(1)
        return output

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            preds = [
                self(num_x.to(self.device), cat_x.to(self.device)).detach()
                for num_x, cat_x in tqdm(data_loader, desc="predict", leave=False)
            ]
        return torch.cat(preds, dim=0).cpu().numpy().ravel()


def main(debug: bool = False):
    start_time = time.time()
    set_seed(config.seed)
    save_dir = Path("./output/")
    save_dir.mkdir(exist_ok=True)

    # features
    with timer("load features"):
        other_columns = [config.id_column, config.target_column]
        train_df, test_df = load_features(other_columns + config.features)

    if debug:
        train_df = train_df.sample(100, random_state=config.seed).reset_index(drop=True)
        test_df = test_df.sample(100, random_state=config.seed).reset_index(drop=True)

    logger.info(f"train shape : {train_df.shape}")
    logger.info(f"test shape  : {test_df.shape}")

    # preprocess
    with timer("preprocess"):
        concat_df = pd.concat([train_df, test_df], sort=False, ignore_index=True)

        categorical_columns = config.categorical_columns
        numerical_columns = concat_df.drop(categorical_columns + other_columns, axis=1).columns.tolist()

        # replace non-overlapping categories with NaN
        for col in categorical_columns:
            overlapped_categories = set(train_df[col].values) & set(test_df[col].values)
            concat_df.loc[~concat_df[col].isin(overlapped_categories), col] = np.nan

        # fill NaN (categorical)
        concat_df[categorical_columns] = concat_df[categorical_columns].astype(str).fillna("__category_NaN__")

        # encode categorical columns
        for col in categorical_columns:
            concat_df[col] = LabelEncoder().fit_transform(concat_df[col])

        # input size
        numerical_input_size = len(numerical_columns)
        category_cardinalities = concat_df[categorical_columns].nunique().values

        # standard scaling
        concat_df[numerical_columns] = StandardScaler().fit_transform(concat_df[numerical_columns])

        # fill NaN (numerical)
        avg_values = concat_df[numerical_columns].mean()
        concat_df[numerical_columns] = concat_df[numerical_columns].fillna(avg_values)

        train_df = concat_df.iloc[: len(train_df)].reset_index(drop=True)
        test_df = concat_df.iloc[-len(test_df) :].reset_index(drop=True)

    train_x = train_df.drop(other_columns, axis=1)
    test_x = test_df.drop(other_columns, axis=1)

    test_dataset = WaterDataset(
        test_x,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # train
    with timer("train"):
        model_save_dir = save_dir / "best_models/"
        model_save_dir.mkdir(exist_ok=True)

        valid_preds = np.zeros((len(train_df),), dtype=np.float64)
        test_preds = np.zeros((len(test_df),), dtype=np.float64)
        cv_scores = []

        for fold, (train_idx, valid_idx) in enumerate(generate_split(train_df)):
            logger.info("-" * 40)
            logger.info(f"fold {fold +  1}")

            train_y = train_df[config.target_column].values.astype(np.float32)
            valid_y = train_y[valid_idx]

            train_dataset = WaterDataset(
                train_x,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                target=train_y,
                indices=train_idx,
            )
            valid_dataset = WaterDataset(
                train_x,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                indices=valid_idx,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            model = WaterModel(numerical_input_size, category_cardinalities)
            model.zero_grad()
            model.to(config.device)

            optimizer = optim.Adam(model.parameters(), lr=config.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.n_epochs, eta_min=config.eta_min, last_epoch=-1
            )

            model_path = model_save_dir / f"model{fold}.pth"

            loss_ema = None
            best_epoch = None
            best_score = np.inf

            for epoch in range(config.n_epochs):
                epoch_start_time = time.time()
                model.train()

                progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
                for num_x, cat_x, target in progress:
                    preds = model(num_x.to(config.device), cat_x.to(config.device))
                    loss = nn.MSELoss()(preds, target.to(config.device))

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss_ema = loss_ema * 0.9 + loss.item() * 0.1 if loss_ema is not None else loss.item()
                    progress.set_postfix(loss=loss_ema)

                scheduler.step()
                valid_fold_preds = model.predict(valid_loader)
                valid_score = mean_squared_error(valid_y, valid_fold_preds, squared=False)
                epoch_elapsed_time = (time.time() - epoch_start_time) / 60
                logger.info(
                    f"  Epoch {epoch + 1}"
                    f" \t train loss: {loss_ema:.5f}"
                    f" \t valid score: {valid_score:.5f}"
                    f" \t time: {epoch_elapsed_time:.2f} min"
                )

                if valid_score < best_score:
                    best_epoch = epoch
                    best_score = valid_score
                    torch.save(model.state_dict(), model_path)

                if debug:
                    break

            logger.info(f"best epoch: {best_epoch + 1} \t score: {best_score:.5f}")

            model.load_state_dict(torch.load(model_path))
            valid_preds[valid_idx] = model.predict(valid_loader)
            test_preds += model.predict(test_loader) / config.n_splits
            cv_scores.append(best_score)

            if debug:
                break

    # save
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
        upload_to_gcs(save_dir, bucket_name=config.bucket_name, gcs_prefix=config.bucket_path)

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
