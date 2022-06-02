import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import freeze, set_seed, timer, upload_to_gcs


@freeze
class config:
    numerical_columns = [
        "tempc",
        "feelslikec",
        "windspeed_kph",
        "windgust_kph",
        "pressure_mb",
        "humidity",
        "visibility_km",
        "cloud",
        "heatindexc",
        "dewpointc",
        "windchillc",
        "isday",
    ]
    categorical_columns = [
        "rain",
        "uvindex",
        "next_rain1",
        "next_rain2",
        "next_rain3",
        "next_rain4",
        "next_rain5",
    ]
    id_column = "chunk_id"
    target_column = "rain"
    prediction_column = "rain_prediction"
    window_size = 23

    n_splits = 5
    n_epochs = 25

    lr = 1e-3
    eta_min = 1e-5

    batch_size = 64
    eval_batch_size = 128

    bucket_name = "kaggledays_championship"
    bucket_path = "kaggledays_beijing/sakami/transformer/"  # CHECK HERE!!!

    device = torch.device("cuda")
    seed = 1029


def generate_split(
    id_column: str,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    train_fold = pd.read_csv("./input/group_k_fold_train.csv")

    if train_fold["fold"].nunique() != config.n_splits:
        raise ValueError("The number of splits in CSV and config must be the same.")

    if len(train_fold) != len(X):
        input_chunk_ids = set(X[id_column])
        train_fold = train_fold[train_fold[id_column].isin(input_chunk_ids)].reset_index(drop=True)

    for i in range(config.n_splits):
        train_chunk_idx = train_fold[train_fold["fold"] != i][id_column].unique()
        valid_chunk_idx = train_fold[train_fold["fold"] == i][id_column].unique()
        yield train_chunk_idx, valid_chunk_idx


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Dict[int, List[Any]],
        numerical_columns: List[str],
        categorical_columns: List[str],
        target: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
    ):
        self.sequences = sequences
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.target = target

        if indices is None:
            indices = list(sequences.keys())

        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        chunk_id = self.indices[index]
        sequence = self.sequences[chunk_id]

        num_x = torch.FloatTensor(np.stack([sequence[col] for col in self.numerical_columns], axis=-1))
        cat_x = torch.LongTensor(np.stack([sequence[col] for col in self.categorical_columns], axis=-1))

        if self.target is not None:
            target = torch.FloatTensor([self.target[chunk_id]])
            return num_x, cat_x, target

        return num_x, cat_x


class SpatialDropout(nn.Module):
    def __init__(self, dropout_rate: float):
        super(SpatialDropout, self).__init__()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.drop(x.unsqueeze(-1)).squeeze(-1)
        return x


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


class SequenceModel(nn.Module):
    def __init__(self, numerical_input_size: int, category_cardinalities: List[int], window_size: int):
        super().__init__()

        self.numerical_linear = nn.Sequential(
            SEScale(numerical_input_size, 16),
            nn.Linear(numerical_input_size, 128),
            nn.Dropout(0.1),
            nn.PReLU(),
        )

        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(n_cat, min(600, round(1.6 * n_cat ** 0.56))) for n_cat in category_cardinalities]
        )
        category_dimensions = [embedding_layer.embedding_dim for embedding_layer in self.category_embeddings]
        self.register_buffer("category_indices", torch.arange(len(category_cardinalities)))

        hidden_size = 128

        self.input_linear = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(sum(category_dimensions) + 128, hidden_size),
            nn.Dropout(0.1),
            nn.PReLU(),
        )

        # transformer
        self.position_embedding = nn.Parameter(torch.zeros((1, window_size, hidden_size)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # conv
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size - 1, kernel_size=1, stride=1),
            nn.Conv1d(hidden_size - 1, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_size, kernel_size=3, stride=1),
            nn.BatchNorm1d(hidden_size),
            SpatialDropout(0.1),
            nn.ReLU(),
        )

        # lstm
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        self.head = nn.Linear(hidden_size * 2, 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        num_x: torch.FloatTensor,
        cat_x: torch.LongTensor,
    ) -> torch.FloatTensor:
        num_x = self.numerical_linear(num_x)

        cat_x = [
            embedding_layer(cat_x[:, :, i])
            for i, embedding_layer in zip(self.category_indices, self.category_embeddings)
        ]
        cat_x = torch.cat(cat_x, dim=-1)

        x = torch.cat([num_x, cat_x], dim=-1)
        x = self.input_linear(x)  # (batch_size, sequence_length, hidden_size)

        # transformer
        # x = x + self.position_embedding
        # x = self.transformer_encoder(x)  # (batch_size, sequence_length, hidden_size)

        # conv
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        # lstm
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        output = self.head(x)
        return output.view(-1)

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        self.eval()
        preds = []
        with torch.no_grad():
            for num_x, cat_x in tqdm(data_loader, desc="predict", leave=False):
                y_pred = self(
                    num_x.to(self.device),
                    cat_x.to(self.device),
                ).detach()  # (batch_size, output_size)
                preds.append(y_pred)
        return torch.cat(preds, dim=0).cpu().numpy()


def main(debug: bool = False):
    start_time = time.time()
    set_seed(config.seed)
    save_dir = Path("./output/")
    save_dir.mkdir(exist_ok=True)

    # features
    with timer("load features"):
        data_dir = Path("./input/nowcastingweather/")
        train_df = pd.read_csv(data_dir / "train.csv")
        sensor_df = pd.read_csv(data_dir / "sensor.csv")
        submission_df = pd.read_csv(data_dir / "sample_submission.csv")

        sensor_df["date"] = pd.to_datetime(sensor_df["date"], infer_datetime_format=True)
        sensor_df["time_idx"] = sensor_df["date"].dt.day * 24 + sensor_df["time"]
        sensor_df["time_idx"] = sensor_df["time_idx"] - sensor_df["time_idx"].min()

        for i in range(5):
            sensor_df[f"next_rain{i + 1}"] = sensor_df.groupby("stationid")["rain"].shift(-i - 1).fillna(4)

    if debug:
        train_df = train_df.sample(100, random_state=config.seed).reset_index(drop=True)
        submission_df = submission_df.sample(100, random_state=config.seed).reset_index(drop=True)
        sensor_df = sensor_df[
            sensor_df[config.id_column].isin(set(train_df[config.id_column]) | set(submission_df[config.id_column]))
        ].reset_index(drop=True)

    train_chunk_ids = set(train_df[config.id_column])
    test_chunk_ids = set(submission_df[config.id_column])

    # preprocess
    with timer("preprocess"):
        categorical_columns = config.categorical_columns
        numerical_columns = config.numerical_columns

        train_sensor_df = sensor_df[sensor_df[config.id_column].isin(train_chunk_ids)].reset_index(drop=True)
        test_sensor_df = sensor_df[sensor_df[config.id_column].isin(test_chunk_ids)].reset_index(drop=True)

        # replace non-overlapping categories with NaN
        for col in categorical_columns:
            overlapped_categories = set(train_sensor_df[col].values) & set(test_sensor_df[col].values)
            sensor_df.loc[~sensor_df[col].isin(overlapped_categories), col] = np.nan

        # fill NaN (categorical)
        sensor_df[categorical_columns] = sensor_df[categorical_columns].astype(str).fillna("__category_NaN__")

        # encode categorical columns
        for col in categorical_columns:
            sensor_df[col] = LabelEncoder().fit_transform(sensor_df[col])

        # input size
        numerical_input_size = len(numerical_columns)
        category_cardinalities = sensor_df[categorical_columns].nunique().values

        # standard scaling
        sensor_df[numerical_columns] = StandardScaler().fit_transform(sensor_df[numerical_columns])

        # fill NaN (numerical)
        avg_values = sensor_df[numerical_columns].mean()
        sensor_df[numerical_columns] = sensor_df[numerical_columns].fillna(avg_values)

        # make sequence
        train_sensor_df = sensor_df[sensor_df[config.id_column].isin(train_chunk_ids)].reset_index(drop=True)
        test_sensor_df = sensor_df[sensor_df[config.id_column].isin(test_chunk_ids)].reset_index(drop=True)

        train_sequence = train_sensor_df.groupby(config.id_column).agg(list).to_dict(orient="index")
        test_sequence = test_sensor_df.groupby(config.id_column).agg(list).to_dict(orient="index")

    test_dataset = SequenceDataset(
        test_sequence,
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

        valid_preds = np.zeros((len(train_sequence),), dtype=np.float64)
        test_preds = np.zeros((len(test_sequence),), dtype=np.float64)
        cv_scores = []

        all_train_chunk_ids = pd.Series(list(train_sequence.keys()))
        train_targets = train_df.set_index(config.id_column)[config.target_column].to_dict()

        for fold, (train_chunk_idx, valid_chunk_idx) in enumerate(generate_split(config.id_column, train_sensor_df)):
            logger.info("-" * 40)
            logger.info(f"fold {fold +  1}")

            valid_idx = np.where(all_train_chunk_ids.isin(valid_chunk_idx))[0]
            valid_y = [train_targets[chunk_id] for chunk_id in valid_chunk_idx]

            train_dataset = SequenceDataset(
                train_sequence,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                target=train_df.set_index(config.id_column)[config.target_column].to_dict(),
                indices=train_chunk_idx,
            )
            valid_dataset = SequenceDataset(
                train_sequence,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                indices=valid_chunk_idx,
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

            model = SequenceModel(numerical_input_size, category_cardinalities, window_size=config.window_size)
            model.zero_grad()
            model.to(config.device)

            optimizer = optim.Adam(model.parameters(), lr=config.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.n_epochs, eta_min=config.eta_min, last_epoch=-1
            )

            model_path = model_save_dir / f"model{fold}.pth"

            loss_ema = None
            best_score = -np.inf

            for epoch in range(config.n_epochs):
                epoch_start_time = time.time()
                model.train()

                progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
                for num_x, cat_x, target in progress:
                    preds = model(num_x.to(config.device), cat_x.to(config.device))
                    loss = nn.MSELoss()(preds, target.to(config.device).view(-1))

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss_ema = loss_ema * 0.9 + loss.item() * 0.1 if loss_ema is not None else loss.item()
                    progress.set_postfix(loss=loss_ema)

                scheduler.step()
                valid_fold_preds = model.predict(valid_loader)
                valid_score = cohen_kappa_score(
                    valid_y, np.clip(np.round(valid_fold_preds), 0, 3), weights="quadratic"
                )
                epoch_elapsed_time = (time.time() - epoch_start_time) / 60
                logger.info(
                    f"  Epoch {epoch + 1}"
                    f" \t train loss: {loss_ema:.5f}"
                    f" \t valid score: {valid_score:.5f}"
                    f" \t time: {epoch_elapsed_time:.2f} min"
                )

                if valid_score > best_score:
                    best_score = valid_score
                    torch.save(model.state_dict(), model_path)

                if debug:
                    break

            logger.info(f"score: {best_score:.5f}")
            model.load_state_dict(torch.load(model_path))
            valid_preds[valid_idx] = model.predict(valid_loader)
            test_preds += model.predict(test_loader) / config.n_splits
            cv_scores.append(best_score)

            if debug:
                break

    # save
    valid_df = pd.DataFrame({config.id_column: list(train_sequence.keys())})
    valid_df[config.prediction_column] = valid_preds
    valid_df.to_csv(save_dir / "valid_proba.csv", index=False)

    test_df = pd.DataFrame({config.id_column: list(test_sequence.keys())})
    test_df[config.prediction_column] = test_preds
    test_df.to_csv(save_dir / "test_proba.csv", index=False)

    pd.DataFrame(
        {config.id_column: list(train_sequence.keys()), config.prediction_column: np.clip(np.round(valid_preds), 0, 3)}
    ).to_csv(save_dir / "valid.csv", index=False)
    pd.DataFrame(
        {config.id_column: list(test_sequence.keys()), config.prediction_column: np.clip(np.round(test_preds), 0, 3)}
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
