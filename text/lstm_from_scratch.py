import itertools
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec, KeyedVectors
from loguru import logger
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from tqdm import tqdm

from src.data import BucketSampler
from src.ema import ModelEma
from src.preprocess import preprocess_text
from src.utils import set_seed, freeze, timer, upload_to_gcs


@freeze
class config:
    id_column = "id"
    text_column = "comment_text"
    target_column = "target"
    prediction_column = "prediction"

    max_len = 220
    max_vocab_size = 100000
    vector_size = 300

    word2vec_kwargs = {
        "epochs": 5,
        "min_count": 1,
        "workers": 8,
        "seed": 2127,
    }

    n_splits = 4
    n_epochs = 10

    lstm_kwargs = {
        "embedding_dropout": 0.2,
        "lstm_hidden_size": 120,
        "gru_hidden_size": 60,
        "out_size": 20,
        "out_dropout": 0.1,
    }
    ema_kwargs = {
        "decay": 0.9,
        "n": 1,
    }

    batch_size = 512
    eval_batch_size = 1024
    bucket_size = 100

    warmup = 0.2
    scheduler = "linear"
    lr = 5e-3

    bucket_name = "kaggledays_championship"
    bucket_path = "sakami/lstm/"  # CHECK HERE!!!

    device = torch.device("cuda")
    seed = 1029


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path("./input/jigsaw-unintended-bias-in-toxicity-classification/")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_df["target"] = (train_df["target"] >= 0.5).astype(float)
    return train_df, test_df


def generate_split(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Generator[np.ndarray, None, None]:
    kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    return kf.split(X)


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[str, Dict[str, int]],
        max_len: int,
        max_vocab_size: int = 100000,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.max_vocab_size = max_vocab_size

    @classmethod
    def build_from_text(
        cls, texts: List[str], max_len: int, max_vocab_size: int = 100000
    ) -> "Tokenizer":
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        vocab = {"<PAD>": 0, "<UNK>": max_vocab_size + 1}
        vocab.update(
            {
                token: _id + 1
                for _id, (token, _) in enumerate(counter.most_common(max_vocab_size))
            }
        )
        tokenizer = cls(vocab, max_len=max_len, max_vocab_size=max_vocab_size)
        return tokenizer

    def encode(self, text: List[str]) -> List[int]:
        return [
            self.vocab.get(token, self.max_vocab_size - 1)
            for token in text[: self.max_len]
        ]

    def batch_encode(self, texts: List[List[str]]) -> List[List[int]]:
        return [self.encode(text) for text in texts]


class TextDataset(Dataset):
    def __init__(
        self,
        encodings: List[List[int]],
        target: Optional[np.ndarray] = None,
        indices: Optional[List[int]] = None,
    ):
        if indices is None:
            indices = np.arange(len(encodings))

        self.encodings = encodings
        self.target = target
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[List[int], float]:
        data = (
            idx,
            self.encodings[self.indices[idx]],
        )
        if self.target is not None:
            data += (self.target[self.indices[idx]],)
        return data


@dataclass
class PaddingCollator:
    device: Union[torch.device, str] = "cpu"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batches = list(zip(*features))
        seqs = batches[1]
        max_len = max([len(seq) for seq in seqs])

        x_batch = torch.zeros(len(seqs), max_len).long()
        for i, seq in enumerate(seqs):
            x_batch[i, -len(seq) :] = torch.LongTensor(seq)

        x_batch = x_batch.to(self.device)

        if len(batches) == 3:
            y_batch = torch.tensor(batches[2]).to(self.device)
            return x_batch, y_batch

        indices = list(batches[0])
        return indices, x_batch


def load_embeddings(
    vocab: Dict[str, int], wv: KeyedVectors, vector_size: int = 100
) -> np.ndarray:
    ps = PorterStemmer()
    lc = LancasterStemmer()
    sb = SnowballStemmer("english")
    embedding_matrix = np.zeros((len(vocab), vector_size), dtype=np.float64)

    for key, i in tqdm(vocab.items(), desc="load embeddings"):
        word = key
        if word in wv:
            embedding_matrix[i] = wv[word]
            continue
        word = key.lower()
        if word in wv:
            embedding_matrix[i] = wv[word]
            continue
        word = key.upper()
        if word in wv:
            embedding_matrix[i] = wv[word]
            continue
        word = key.capitalize()
        if word in wv:
            embedding_matrix[i] = wv[word]
            continue
        word = ps.stem(key)
        if word in wv:
            embedding_matrix[i] = wv[word]
            continue
        word = lc.stem(key)
        if word in wv:
            embedding_matrix[i] = wv[word]
            continue
        word = sb.stem(key)
        if word in wv:
            embedding_matrix[i] = wv[word]
            continue

    return embedding_matrix


class LstmModel(nn.Module):
    def __init__(
        self,
        embedding_shape: Tuple[int, int],
        embedding_dropout: float = 0.2,
        lstm_hidden_size: int = 120,
        gru_hidden_size: int = 60,
        out_size: int = 20,
        out_dropout: float = 0.1,
    ):
        super().__init__()
        self.gru_hidden_size = gru_hidden_size

        self.embedding = nn.Embedding(*embedding_shape)
        self.embedding_dropout = nn.Dropout2d(embedding_dropout)

        self.lstm = nn.LSTM(
            embedding_shape[1],
            lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.gru = nn.GRU(
            lstm_hidden_size * 2, gru_hidden_size, bidirectional=True, batch_first=True
        )

        self.linear = nn.Linear(gru_hidden_size * 6, out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(out_dropout)
        self.out = nn.Linear(out_size, 1)

    @classmethod
    def from_pretrained(
        cls,
        embedding_matrix: np.ndarray,
        embedding_dropout: float = 0.2,
        lstm_hidden_size: int = 120,
        gru_hidden_size: int = 60,
        out_size: int = 20,
        out_dropout: float = 0.1,
    ) -> "LstmModel":
        model = cls(
            embedding_shape=embedding_matrix.shape,
            embedding_dropout=embedding_dropout,
            lstm_hidden_size=lstm_hidden_size,
            gru_hidden_size=gru_hidden_size,
            out_size=out_size,
            out_dropout=out_dropout,
        )
        model.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32)
        )
        model.embedding.weight.requires_grad = False
        return model

    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        h_embedding = self.embedding(x)
        h_embedding = self.apply_spatial_dropout(h_embedding)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, hh_gru = self.gru(h_lstm)
        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((hh_gru, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out.view(-1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def predict(model: nn.Module, data_loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = torch.zeros(
        (len(data_loader.dataset),), dtype=torch.float, device=model.device
    )
    with torch.no_grad():
        for idx, batch in tqdm(data_loader, desc="predict", leave=False):
            preds[idx] = model(batch).detach()
    return preds.sigmoid().cpu().numpy().ravel()


def main(debug: bool = False):
    start_time = time.time()
    set_seed(config.seed)

    # load data
    train_df, test_df = load_data()

    if debug:
        train_df = train_df.sample(100, random_state=config.seed).reset_index(drop=True)
        test_df = test_df.sample(100, random_state=config.seed).reset_index(drop=True)

    logger.info(f"train shape : {train_df.shape}")
    logger.info(f"test shape  : {test_df.shape}")

    # train word2vec
    with timer("train word2vec"):
        texts = list(
            itertools.chain(train_df[config.text_column], test_df[config.text_column])
        )
        fill_blank = lambda text: text if len(text.strip()) > 0 else "<BLANK>"
        tokens = [fill_blank(preprocess_text(text)).split() for text in texts]
        word_freq = Counter(itertools.chain.from_iterable(tokens))

        if not debug:
            # log training progress
            logging.basicConfig(
                format="%(levelname)s - %(asctime)s: %(message)s",
                datefmt="%H:%M:%S",
                level=logging.INFO,
            )

        w2v_model = Word2Vec(
            **config.word2vec_kwargs,
            vector_size=config.vector_size,
            max_vocab_size=config.max_vocab_size,
        )
        w2v_model.build_vocab_from_freq(word_freq)
        w2v_model.train(tokens, total_examples=len(tokens), epochs=w2v_model.epochs)
        wv = w2v_model.wv

        vocab = {
            **{"<PAD>": 0, "<UNK>": config.max_vocab_size + 1},
            **{key: idx + 1 for key, idx in wv.key_to_index.items()},
        }

    # prepare data
    with timer("prepare data"):
        # encode text
        tokenizer = Tokenizer(
            vocab, max_len=config.max_len, max_vocab_size=config.max_vocab_size
        )
        encodings = []
        sort_keys = []
        for text in tqdm(tokens, desc="encode text"):
            encoding = tokenizer.encode(text)
            encodings.append(encoding)
            sort_keys.append(len(encoding))

        train_encodings = encodings[: len(train_df)]
        train_sort_keys = sort_keys[: len(train_df)]
        test_encodings = encodings[len(train_df) :]
        test_sort_keys = sort_keys[len(train_df) :]

        collator = PaddingCollator(device=config.device)
        test_dataset = TextDataset(test_encodings)
        test_sampler = BucketSampler(
            test_dataset,
            test_sort_keys,
            bucket_size=None,
            batch_size=config.eval_batch_size,
            shuffle_data=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            sampler=test_sampler,
            num_workers=0,
            collate_fn=collator,
        )

    # token length summary
    train_sort_keys = np.array(train_sort_keys)
    test_sort_keys = np.array(test_sort_keys)

    logger.info("token length statistics")
    logger.info(f"  max   : {train_sort_keys.max()}")
    logger.info(f"  mean  : {train_sort_keys.mean()}")
    logger.info(f"  50%   : {np.percentile(train_sort_keys, 50)}")
    logger.info(f"  75%   : {np.percentile(train_sort_keys, 75)}")
    logger.info(f"  90%   : {np.percentile(train_sort_keys, 90)}")
    logger.info(f"  95%   : {np.percentile(train_sort_keys, 95)}")
    logger.info(f"  99%   : {np.percentile(train_sort_keys, 99)}")
    logger.info(f"  99.5% : {np.percentile(train_sort_keys, 99.5)}")
    logger.info(f"  99.9% : {np.percentile(train_sort_keys, 99.9)}")

    # load embeddings
    with timer("load embeddings"):
        embedding_matrix = load_embeddings(vocab, wv, vector_size=config.vector_size)

    # train
    with timer("train"):
        valid_preds = np.zeros((len(train_df),), dtype=np.float64)
        test_preds = np.zeros((len(test_df),), dtype=np.float64)
        cv_scores = []

        for fold, (train_idx, valid_idx) in enumerate(generate_split(train_encodings)):
            logger.info(f"fold {fold + 1}")

            # model
            model = LstmModel.from_pretrained(embedding_matrix, **config.lstm_kwargs)
            model.zero_grad()
            model.to(config.device)

            ema = ModelEma(model, **config.ema_kwargs)
            optimizer = optim.Adam(model.parameters(), lr=config.lr)

            num_training_steps = (
                len(train_encodings) * config.n_epochs // config.batch_size
            )
            num_warmup_steps = int(config.warmup * num_training_steps)
            scheduler = get_scheduler(
                config.scheduler, optimizer, num_warmup_steps, num_training_steps
            )

            # data
            train_dataset = TextDataset(
                train_encodings, train_df[config.target_column], indices=train_idx
            )
            valid_dataset = TextDataset(train_encodings, indices=valid_idx)

            train_sampler = BucketSampler(
                train_dataset,
                train_sort_keys[train_idx],
                bucket_size=config.bucket_size,
                batch_size=config.batch_size,
                shuffle_data=True,
            )
            valid_sampler = BucketSampler(
                valid_dataset,
                train_sort_keys[valid_idx],
                bucket_size=None,
                batch_size=config.eval_batch_size,
                shuffle_data=False,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=train_sampler,
                num_workers=0,
                collate_fn=collator,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.eval_batch_size,
                sampler=valid_sampler,
                num_workers=0,
                collate_fn=collator,
            )

            valid_y = train_df[config.target_column].values[valid_idx]
            loss_ema = None

            for epoch in range(config.n_epochs):
                epoch_start_time = time.time()
                model.train()

                progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
                for x_batch, y_batch in progress:
                    y_preds = model(x_batch)
                    loss = nn.BCEWithLogitsLoss()(y_preds, y_batch)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    ema.update(model)

                    loss_ema = (
                        loss_ema * 0.9 + loss.item() * 0.1
                        if loss_ema is not None
                        else loss.item()
                    )
                    progress.set_postfix(loss=loss_ema)

                valid_fold_preds = predict(model, valid_loader)
                valid_score = roc_auc_score(valid_y, valid_fold_preds)
                epoch_elapsed_time = (time.time() - epoch_start_time) / 60
                logger.info(
                    f"  Epoch {epoch + 1}"
                    f" \t train loss: {loss_ema:.5f}"
                    f" \t valid score: {valid_score:.5f}"
                    f" \t time: {epoch_elapsed_time:.2f} min"
                )

                if debug:
                    break

            # EMA
            ema.ema_model.lstm.flatten_parameters()
            ema.ema_model.gru.flatten_parameters()

            valid_fold_preds = predict(ema.ema_model, valid_loader)
            valid_score = roc_auc_score(valid_y, valid_fold_preds)
            epoch_elapsed_time = (time.time() - epoch_start_time) / 60
            logger.info(f"  EMA model \t valid score: {valid_score:.5f}")

            valid_preds[valid_idx] = valid_fold_preds
            test_preds += predict(ema.ema_model, test_loader) / config.n_splits
            cv_scores.append(valid_score)

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
