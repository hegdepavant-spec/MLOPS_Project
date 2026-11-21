"""
Compatibility wrapper for src/ml/train.py
"""
from src.ml.train import train_and_evaluate_all as train_models

if __name__ == "__main__":
    train_models()
