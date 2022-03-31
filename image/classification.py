import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import freeze, set_seed, timer, upload_to_gcs


@freeze
class config:
    id_column = "image_id"
    target_column = "class_6"
    prediction_column = "class_6"
    data_dir = Path("./input/dont-stop-until-you-drop")
    num_labels = 6

    model_name = "tf_efficientnet_b0_ns"
    image_size = 512
    fp16 = False

    n_splits = 5
    n_epochs = 3

    batch_size = 32
    eval_batch_size = 64

    lr = 1e-4
    weight_decay = 1e-6
    max_grad_norm = 1000
    T_max = 3
    eta_min = 1e-6

    bucket_name = "kaggledays_championship"
    bucket_path = "sakami/efficientnet_b0_v1/"  # CHECK HERE!!!

    device = torch.device("cuda")
    seed = 1029


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    submission_df = pd.read_csv(data_dir / "sample_submission.csv")
    return train_df, submission_df


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


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        image_ids: np.ndarray,
        target: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        transform: Optional[A.Compose] = None,
    ):
        if indices is None:
            indices = np.arange(len(image_ids))

        self.data_dir = Path(data_dir)
        self.image_ids = image_ids
        self.target = target
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        idx = self.indices[i]
        image_id = self.image_ids[idx]
        file_path = self.data_dir / image_id

        if not file_path.exists():
            raise ValueError(f"{str(file_path)} does not exist.")

        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.target is not None:
            target = torch.LongTensor([self.target[idx]])
            return image, target

        return image


def get_transforms(mode: str, image_size: int) -> A.Compose:
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(image_size, image_size, scale=(0.85, 1.0)),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif mode == "test":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class ImageModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.fc = nn.Linear(self.n_features, num_labels)

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        feature = self.model(image)
        output = self.fc(feature)
        return output

    def predict(self, data_loader: DataLoader, fp16: bool = False) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        preds = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="predict", leave=False):
                with autocast(enabled=fp16):
                    preds.append(self(batch.to(device)).detach())

        preds = torch.cat(preds, dim=0)
        preds = preds.float().softmax(dim=1).cpu().numpy()
        return preds


def main(debug: bool = False):
    start_time = time.time()
    set_seed(config.seed)

    save_dir = Path("./output/")
    save_dir.mkdir(exist_ok=True)

    model_save_dir = save_dir / "models"
    model_save_dir.mkdir(exist_ok=True)

    # load data
    train_df, submission_df = load_data(config.data_dir)

    if debug:
        train_df = train_df.sample(100, random_state=config.seed).reset_index(drop=True)
        submission_df = submission_df.sample(100, random_state=config.seed).reset_index(drop=True)

    logger.info(f"train shape : {train_df.shape}")
    logger.info(f"submission shape  : {submission_df.shape}")

    test_transform = get_transforms(mode="test", image_size=config.image_size)
    test_dataset = ImageDataset(
        config.data_dir / "images/test_images", submission_df[config.id_column].values, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # train
    with timer("train"):
        valid_preds = np.zeros((len(train_df), config.num_labels), dtype=np.float64)
        test_preds = np.zeros((len(submission_df), config.num_labels), dtype=np.float64)
        cv_scores = []

        for fold, (train_idx, valid_idx) in enumerate(generate_split(train_df)):
            logger.info("-" * 40)
            logger.info(f"fold {fold + 1}")

            # model
            model = ImageModel(config.model_name, num_labels=config.num_labels, pretrained=True)
            model.zero_grad()
            model.to(config.device)

            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, amsgrad=False)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.T_max, eta_min=config.eta_min, last_epoch=-1
            )
            scaler = GradScaler()

            # data
            train_transform = get_transforms(mode="train", image_size=config.image_size)
            valid_transform = get_transforms(mode="test", image_size=config.image_size)

            train_dataset = ImageDataset(
                config.data_dir / "images/train_images",
                train_df[config.id_column].values,
                target=train_df[config.target_column].values,
                indices=train_idx,
                transform=train_transform,
            )
            valid_dataset = ImageDataset(
                config.data_dir / "images/train_images",
                train_df[config.id_column].values,
                indices=valid_idx,
                transform=valid_transform,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.eval_batch_size,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            )

            valid_y = train_df[config.target_column].values[valid_idx]
            best_score = 0.0
            best_fold_preds = 0.0
            loss_ema = None

            for epoch in range(config.n_epochs):
                epoch_start_time = time.time()
                model.train()

                progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
                for x_batch, y_batch in progress:
                    with autocast(enabled=config.fp16):
                        y_preds = model(x_batch.to(config.device))
                        y_batch = y_batch.to(config.device).squeeze(1)
                        loss = nn.CrossEntropyLoss()(y_preds, y_batch)

                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    loss_ema = loss_ema * 0.9 + loss.item() * 0.1 if loss_ema is not None else loss.item()
                    progress.set_postfix(loss=loss_ema)

                scheduler.step()
                valid_fold_preds = model.predict(valid_loader)
                valid_score = f1_score(valid_y, valid_fold_preds.argmax(axis=1), average="micro")
                epoch_elapsed_time = (time.time() - epoch_start_time) / 60
                logger.info(
                    f"  Epoch {epoch + 1}"
                    f" \t train loss: {loss_ema:.5f}"
                    f" \t valid score: {valid_score:.5f}"
                    f" \t time: {epoch_elapsed_time:.2f} min"
                )
                torch.save(model.state_dict(), model_save_dir / f"model{epoch}.pth")

                if valid_score > best_score or epoch == 0:
                    best_score = valid_score
                    best_fold_preds = valid_fold_preds
                    torch.save(model.state_dict(), model_save_dir / "best_model.pth")

                if debug:
                    break

            model.load_state_dict(torch.load(model_save_dir / "best_model.pth"))
            test_fold_preds = model.predict(test_loader)

            valid_preds[valid_idx] = best_fold_preds
            test_preds += test_fold_preds / config.n_splits
            cv_scores.append(best_score)

            if debug:
                break

    # save
    pd.DataFrame(
        {
            config.id_column: train_df[config.id_column],
            config.prediction_column: valid_preds.argmax(axis=1),
        }
    ).to_csv(save_dir / "valid.csv", index=False)
    pd.DataFrame(
        {
            config.id_column: submission_df[config.id_column],
            config.prediction_column: test_preds.argmax(axis=1),
        }
    ).to_csv(save_dir / "test.csv", index=False)

    proba_columns = [f"pred{i}" for i in range(valid_preds.shape[1])]

    valid_preds_df = pd.DataFrame({config.id_column: train_df[config.id_column]})
    valid_preds_df[proba_columns] = valid_preds
    valid_preds_df.to_csv(save_dir / "valid_proba.csv", index=False)

    test_preds_df = pd.DataFrame({config.id_column: submission_df[config.id_column]})
    test_preds_df[proba_columns] = test_preds
    test_preds_df.to_csv(save_dir / "test_proba.csv", index=False)

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
