"""
File: main.py
Author(s): Kayvon Khosrowpour
Date created: 11/26/18

Description:
Given a desired model and CSV, will perform training on the given
data using that model. Includes support for a random forest
classifier, an AdaBoost classifier, and XGBoost classifier.

For RandomForestClassifier:
    python3 main.py -c configs/rfc_config.ini
For AdaBoostClassifier:
    python3 main.py -c configs/adab_defaults.ini
For XGBoostClassifier:
    python3 main.py -c configs/xgb_defaults.ini
"""

from configs import parse
from train import build_model

def main():
    config = parse()
    model = build_model(config)
    model.train()
    model.save()

if __name__ == '__main__':
    main()