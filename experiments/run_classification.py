#!/usr/bin/env python3
"""
Classification and evaluation pipeline for FuseCC
- Reads a classification config YAML
- Loads features CSV (results produced by run_experiment.py)
- Runs baseline CV (XGBoost, RandomForest, LogisticRegression)
- Runs randomized hyperparameter search for XGBoost and RandomForest
- Trains a stacking ensemble and evaluates with stratified CV
- Saves models, OOF predictions, and metrics to output_dir/classification/

Usage:
  python experiments/run_classification.py config/classify_bcb.yaml

The script is intentionally conservative about resources (uses n_jobs from config or available CPUs).
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as fh:
        return yaml.safe_load(fh)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def report_scores(name: str, estimator, X, y, cv):
    scorers = {
        'f1': lambda est, X, y: cross_val_score(est, X, y, cv=cv, scoring='f1'),
        'precision': lambda est, X, y: cross_val_score(est, X, y, cv=cv, scoring='precision'),
        'recall': lambda est, X, y: cross_val_score(est, X, y, cv=cv, scoring='recall'),
    }
    res = {}
    logging.info('Evaluating %s with CV (%d folds)', name, cv.get_n_splits())
    for k, fn in scorers.items():
        scores = fn(estimator, X, y)
        res[k] = {'mean': float(scores.mean()), 'std': float(scores.std())}
        logging.info('%s %s: mean=%.4f std=%.4f', name, k, scores.mean(), scores.std())
    return res


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    features_path = Path(cfg.get('features_path', 'results/real_fusecc/results.csv'))
    output_dir = Path(cfg.get('output_dir', 'results/real_fusecc/classification'))
    n_splits = int(cfg.get('n_splits', 5))
    random_state = int(cfg.get('random_state', 12345))
    n_iter = int(cfg.get('n_iter', 30))
    n_jobs = int(cfg.get('n_jobs', os.cpu_count() or 1))

    ensure_dir(output_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', handlers=[logging.FileHandler(output_dir / 'run_classification.log'), logging.StreamHandler(sys.stdout)])
    logging.info('Classification run – features: %s, output: %s', features_path, output_dir)

    df = pd.read_csv(features_path)
    if 'label' not in df.columns:
        logging.error('No label column found in features CSV')
        sys.exit(1)
    df = df.dropna()
    y = df['label'].astype(int)
    feat_cols = [c for c in df.columns if c.startswith('score_')]
    X = df[feat_cols].values
    logging.info('Loaded features X shape=%s, y distribution=%s', X.shape, y.value_counts().to_dict())

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results: Dict[str, Any] = {'baseline': {}, 'tuned': {}, 'stacking': {}}

    # Baseline estimators
    estimators = {}
    if XGB_AVAILABLE:
        xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=n_jobs, random_state=random_state)
        estimators['xgb'] = xgb_base
        logging.info('XGBoost available: including xgb baseline')
    else:
        logging.info('XGBoost not available; skipping xgb baseline')

    rf_base = RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
    lr_base = LogisticRegression(max_iter=2000)

    if 'xgb' in estimators:
        results['baseline']['xgb'] = report_scores('xgb', estimators['xgb'], X, y, cv)
    results['baseline']['rf'] = report_scores('rf', rf_base, X, y, cv)
    results['baseline']['lr'] = report_scores('lr', lr_base, X, y, cv)

    # Hyperparameter tuning (RandomizedSearchCV)
    # XGBoost param grid
    best_models = {}
    if XGB_AVAILABLE:
        xgb_param_dist = {
            'n_estimators': [100, 200, 400],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        logging.info('Running RandomizedSearchCV for XGBoost (%d iters)', n_iter)
        rsearch_xgb = RandomizedSearchCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=n_jobs, random_state=random_state),
                                         xgb_param_dist, n_iter=n_iter, scoring='f1', cv=cv, random_state=random_state, n_jobs=1)
        rsearch_xgb.fit(X, y)
        best_models['xgb'] = rsearch_xgb.best_estimator_
        results['tuned']['xgb'] = {'best_params': rsearch_xgb.best_params_, 'best_score': float(rsearch_xgb.best_score_)}
        logging.info('XGBoost best score %.4f', rsearch_xgb.best_score_)

    # RandomForest tuning
    rf_param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [None, 10, 20, 40],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10]
    }
    logging.info('Running RandomizedSearchCV for RandomForest (%d iters)', n_iter)
    rsearch_rf = RandomizedSearchCV(RandomForestClassifier(n_jobs=n_jobs, random_state=random_state),
                                    rf_param_dist, n_iter=n_iter, scoring='f1', cv=cv, random_state=random_state, n_jobs=1)
    rsearch_rf.fit(X, y)
    best_models['rf'] = rsearch_rf.best_estimator_
    results['tuned']['rf'] = {'best_params': rsearch_rf.best_params_, 'best_score': float(rsearch_rf.best_score_)}
    logging.info('RandomForest best score %.4f', rsearch_rf.best_score_)

    # Save tuned models
    for name, m in best_models.items():
        joblib.dump(m, output_dir / f'{name}_best.pkl')
        logging.info('Saved tuned model: %s', output_dir / f'{name}_best.pkl')

    # Stacking ensemble using tuned models (and logistic regression as meta)
    estimators_list = []
    if 'xgb' in best_models:
        estimators_list.append(('xgb', best_models['xgb']))
    estimators_list.append(('rf', best_models['rf']))
    estimators_list.append(('lr', LogisticRegression(max_iter=2000)))

    stacking = StackingClassifier(estimators=estimators_list, final_estimator=LogisticRegression(max_iter=2000), n_jobs=n_jobs, passthrough=False)
    # Evaluate stacking via CV
    results['stacking'] = report_scores('stacking', stacking, X, y, cv)

    # Fit stacking on full data and save
    stacking.fit(X, y)
    joblib.dump(stacking, output_dir / 'stacking_model.pkl')
    logging.info('Saved stacking model to %s', output_dir / 'stacking_model.pkl')

    # Produce OOF predictions using cross_val_predict (probabilities)
    logging.info('Generating OOF predicted probabilities (stacking)')
    oof_probs = cross_val_predict(stacking, X, y, cv=cv, method='predict_proba', n_jobs=n_jobs)
    # save OOF predictions
    oof_df = pd.DataFrame({'id': df['id'], 'oof_prob': oof_probs[:, 1], 'label': y})
    oof_df.to_csv(output_dir / 'oof_predictions.csv', index=False)
    logging.info('Saved OOF predictions to %s', output_dir / 'oof_predictions.csv')

    # Final evaluation on OOF
    oof_pred_labels = (oof_probs[:, 1] >= 0.5).astype(int)
    final_metrics = {
        'f1': float(f1_score(y, oof_pred_labels)),
        'precision': float(precision_score(y, oof_pred_labels)),
        'recall': float(recall_score(y, oof_pred_labels))
    }
    results['final_oof'] = final_metrics
    logging.info('Final OOF metrics: %s', final_metrics)

    # Save results JSON and feature list
    with open(output_dir / 'classification_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    with open(output_dir / 'feature_columns.json', 'w') as fh:
        json.dump(feat_cols, fh, indent=2)

    logging.info('Classification pipeline finished. Results and models saved to %s', output_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python experiments/run_classification.py config/classify_bcb.yaml')
        sys.exit(1)
    main(sys.argv[1])
