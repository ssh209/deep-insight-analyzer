"""
LightGBM NVI 예측 모델 사전 학습 & 저장 스크립트.

720h 학습 데이터로 모델을 학습하고 models/nvi_forecaster.pkl로 저장합니다.
이후 ForecasterAgent는 매번 학습 없이 이 모델을 로드합니다.

실행: python scripts/train_model.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import joblib
from lightgbm import LGBMRegressor

TRAIN_CSV = "data/pr_crisis_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "nvi_forecaster.pkl")

FEATURES = [
    'Hours_Since_Start', 'Company_Action_Type', 'Influencer_Impact',
    'Negative_Ratio', 'Mockery_Index', 'Advocate_Ratio', 'Negative_Momentum'
]

def train_and_save():
    print(f">> Loading training data: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    issue_start = df['Datetime'].min()
    df['Hours_Since_Start'] = (df['Datetime'] - issue_start).dt.total_seconds() / 3600
    
    X_train = df[FEATURES]
    y_train = df['Actual_NVI']
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Features: {FEATURES}")
    
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        min_data_in_leaf=3
    )
    model.fit(X_train, y_train)
    
    # 모델 저장
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    
    print(f"\n[OK] Model saved: {MODEL_PATH}")
    print(f"   File size: {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")
    
    # 간단한 검증
    y_pred = model.predict(X_train)
    from sklearn.metrics import mean_absolute_error, r2_score
    print(f"\n-- Train Set Metrics --")
    print(f"   MAE:  {mean_absolute_error(y_train, y_pred):.4f}")
    print(f"   R2:   {r2_score(y_train, y_pred):.4f}")

if __name__ == "__main__":
    train_and_save()
