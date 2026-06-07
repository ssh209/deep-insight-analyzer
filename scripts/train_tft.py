"""
TFT(Temporal Fusion Transformer) NVI 예측 모델 학습 스크립트.

학습 데이터(여러 위기 사례의 시간당 NVI 시계열)로 모델을 학습하고,
models/tft_nvi.ckpt 체크포인트로 저장합니다.

이후 TFTForecasterAgent는 이 체크포인트를 로드하여 예측합니다.

사용법:
  python scripts/train_tft.py                          # 기본 학습
  python scripts/train_tft.py --data-dir data/cases/    # 커스텀 데이터 경로
  python scripts/train_tft.py --epochs 30 --gpus 1      # GPU 학습
  python scripts/train_tft.py --fast-dev-run             # 빠른 검증 (1 batch)

데이터 형식:
  data/cases/ 디렉토리에 CSV 파일들 (각 파일이 하나의 위기 사례)
  컬럼: Datetime, Hours_Since_Start, Company_Action_Type, Influencer_Impact,
        Negative_Ratio, Mockery_Index, Advocate_Ratio, Negative_Momentum,
        Actual_NVI, crisis_type
"""
import sys
import os
import argparse
import glob
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# 설정
# ==========================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CHECKPOINT_NAME = "tft_nvi"

TARGET = "Actual_NVI"
MAX_ENCODER_LENGTH = 72
MAX_PREDICTION_LENGTH = 168

KNOWN_FUTURE_REALS = ["Hours_Since_Start"]
KNOWN_FUTURE_CATEGORICALS = ["Company_Action_Type", "Influencer_Impact"]
OBSERVED_REALS = ["Negative_Ratio", "Mockery_Index", "Advocate_Ratio", "Negative_Momentum"]
STATIC_CATEGORICALS = ["crisis_type"]

CRISIS_TYPE_MAP = {"victim": "0", "accidental": "1", "preventable": "2"}


def load_training_data(data_dir: str) -> pd.DataFrame:
    """data_dir 내 모든 CSV를 읽어 하나의 DataFrame으로 합침.

    각 CSV는 하나의 위기 사례이며, series_id로 구분됩니다.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"학습 데이터를 찾을 수 없습니다: {data_dir}/*.csv")

    frames = []
    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)
        df["series_id"] = f"case_{i:03d}"

        # crisis_type이 없으면 기본값
        if "crisis_type" not in df.columns:
            df["crisis_type"] = "1"  # accidental
        else:
            df["crisis_type"] = df["crisis_type"].map(
                lambda x: CRISIS_TYPE_MAP.get(str(x), str(x))
            )

        # 시간 인덱스 (사례 내 순차)
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            issue_start = df["Datetime"].min()
            df["Hours_Since_Start"] = (
                (df["Datetime"] - issue_start).dt.total_seconds() / 3600
            )

        df["time_idx"] = range(len(df))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f">> 학습 데이터: {len(csv_files)}개 사례, {len(combined)} rows 로드")

    # 카테고리컬 타입 변환
    for col in KNOWN_FUTURE_CATEGORICALS + STATIC_CATEGORICALS:
        combined[col] = combined[col].astype(int).astype(str)
    combined["series_id"] = combined["series_id"].astype(str)

    return combined


def create_datasets(df: pd.DataFrame, val_ratio: float = 0.2):
    """Train/Val TimeSeriesDataSet 생성."""
    # 각 시계열의 마지막 20%를 validation으로 사용
    training_cutoff = df.groupby("series_id")["time_idx"].transform(
        lambda x: x.max() - int(len(x) * val_ratio)
    )

    train_df = df[df["time_idx"] <= training_cutoff].copy()
    val_df = df.copy()  # validation은 전체 데이터에서 decoder 부분이 자동 분리

    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["series_id"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_categoricals=KNOWN_FUTURE_CATEGORICALS,
        time_varying_known_reals=KNOWN_FUTURE_REALS,
        time_varying_unknown_reals=[TARGET] + OBSERVED_REALS,
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=True, stop_randomization=True
    )

    print(f"   Train: {len(training_dataset)} samples")
    print(f"   Val:   {len(validation_dataset)} samples")

    return training_dataset, validation_dataset


def train(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    gpus: int = 0,
    fast_dev_run: bool = False,
):
    """TFT 모델 학습 및 체크포인트 저장."""
    # 1. 데이터 로드
    df = load_training_data(data_dir)

    # 2. Dataset 생성
    training_dataset, validation_dataset = create_datasets(df)

    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    # 3. TFT 모델 구성
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,  # 7 quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"\n>> TFT 모델 파라미터: {tft.size() / 1e3:.1f}K")
    print(f"   hidden_size: 32, attention_heads: 2")
    print(f"   encoder_length: {MAX_ENCODER_LENGTH}, prediction_length: {MAX_PREDICTION_LENGTH}")

    # 4. 학습
    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(
            dirpath=MODEL_DIR,
            filename=CHECKPOINT_NAME,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
    ]

    accelerator = "gpu" if gpus > 0 and torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
    )

    print(f"\n>> 학습 시작 (epochs={epochs}, accelerator={accelerator})")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 5. 최적 모델 경로 출력
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        # 표준 경로로 복사

        target_path = os.path.join(MODEL_DIR, "tft_nvi.ckpt")
        shutil.copy2(best_model_path, target_path)
        print(f"\n[OK] Best model saved: {target_path}")
        print(f"   Val loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    else:
        print("\n[WARN] No checkpoint saved (fast_dev_run?)")

    # 6. 간단한 검증
    if not fast_dev_run and best_model_path:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)
        print(f"   Prediction shape: {predictions.output.prediction.shape}")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="TFT NVI 예측 모델 학습")
    parser.add_argument(
        "--data-dir", default="data/cases/",
        help="학습 데이터 디렉토리 (CSV 파일들, 기본: data/cases/)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="학습 에폭 수 (기본: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기 (기본: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률 (기본: 0.001)")
    parser.add_argument("--gpus", type=int, default=0, help="GPU 수 (기본: 0, CPU)")
    parser.add_argument("--fast-dev-run", action="store_true", help="1 batch만 실행 (검증용)")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gpus=args.gpus,
        fast_dev_run=args.fast_dev_run,
    )
