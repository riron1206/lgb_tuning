# -*- coding: utf-8 -*-
"""
CatBoost + cross-validation で回帰モデル作成
Usage:
    $ conda activate py37
    $ python run_catboost_reg.py
"""
import argparse
import os

import numpy as np
import pandas as pd

# from model_nn import ModelNN
from model_catboost import ModelCatBoost
from runner import Runner
from util import Submission


def get_args() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output_dir", type=str, default="../model/v1")
    # ap.add_argument("-i", "--input_dir", type=str, default='../input/v1')
    ap.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=r"C:\Users\yokoi.shingo\my_task\Bike_Shareing\data\add_feature_v4",
    )
    args = vars(ap.parse_args())

    # 特徴量の指定
    # num_feature = ['temp', 'atemp', 'humidity', 'windspeed', 'elapsed_day', 'count_month_mean', 'count_month_std']
    # cat_feature_name = ['season', 'holiday', 'workingday', 'weather', 'day', 'year', 'month', 'dayofweek', 'hour', 'windspeed_category', 'month_class', 'dayofweek_class', 'hour_class']
    num_feature = [
        "temp",
        "atemp",
        "humidity",
        "windspeed",
        "elapsed_day",
        "count_month_mean",
        "count_month_std",
        "weather_am_pm",
        "discomfort",
    ]
    categorical_feature = [
        "season",
        "holiday",
        "workingday",
        "weather",
        "year",
        "month",
        "dayofweek",
        "hour",
        "windspeed_category",
        "month_class",
        "dayofweek_class",
        "hour_class",
        "am_pm",
    ]
    args["num_feature"] = num_feature
    args["categorical_feature"] = categorical_feature
    args["features"] = [*num_feature, *categorical_feature]
    args["target"] = "count"
    args["transform"] = np.expm1  # submission csvの後処理（add_feature_v4は対数化してるので元戻すため）

    return args


def run_cv(args, params_catboost, name="catboost", n_fold=4) -> None:
    """cross-validation"""
    # モデル学習予測をとりまとめるRunnerオブジェクト作成
    runner = Runner(
        name,
        ModelCatBoost,
        args["features"],
        params_catboost,
        target=args["target"],
        n_fold=n_fold,
        train_csv=os.path.join(args["input_dir"], "train.csv"),
        test_csv=os.path.join(args["input_dir"], "test.csv"),
        output_dir=args["output_dir"],
    )

    # cross-validationで学習
    _, va_cv_scores_mean = runner.run_train_cv()  # 学習したモデルも返すが使わないので _ にしてる
    print("va_cv_scores_mean", va_cv_scores_mean)

    # test_csv予測
    runner.run_predict_cv()  # 予測結果はpklファイルに保存するのでreturnはなし

    # submit csv作成
    _ = Submission.output_submit(
        os.path.join(args["output_dir"], f"{name}-test.pkl"),
        os.path.join(args["output_dir"], f"{name}-submission.csv"),
        transform=args["transform"],
    )

    # 0番目のfoldのモデルでfeature importance可視化
    ModelCatBoost("", {}).save_plot_importance(
        os.path.join(args["output_dir"], f"{name}-0.model"),
        args["features"],
        png_path=os.path.join(args["output_dir"], f"{name}-0-plot_importance.png"),
    )


def run_all(args, params_catboost, name="catboost") -> None:
    """
    学習データ全体を使う場合（validation set無いからearly stoppingは使えない。cvの時より精度下がるが可能性あり）
    """
    # モデル学習予測をとりまとめるRunnerオブジェクト作成
    runner = Runner(
        name,
        ModelCatBoost,
        args["features"],
        params_catboost,
        target=args["target"],
        train_csv=os.path.join(args["input_dir"], "train.csv"),
        test_csv=os.path.join(args["input_dir"], "test.csv"),
        output_dir=args["output_dir"],
    )

    # 学習
    _ = runner.run_train_all()  # 学習したモデル返すが使わないので _ にしてる

    # test_csv予測
    runner.run_predict_all()  # 予測結果はpklファイルに保存するのでreturnはなし

    # submit csv作成
    _ = Submission.output_submit(
        os.path.join(args["output_dir"], f"{name}-all-test.pkl"),
        os.path.join(args["output_dir"], f"{name}-all-submission.csv"),
        transform=args["transform"],
    )

    # feature importance可視化
    ModelCatBoost("", {}).save_plot_importance(
        os.path.join(args["output_dir"], f"{name}-all.model"),
        args["features"],
        png_path=os.path.join(args["output_dir"], f"{name}-all-plot_importance.png"),
    )


if __name__ == "__main__":
    args = get_args()

    # catboostのパラメータ
    params_catboost = {
        "loss_function": "RMSE",
        "depth": 12,
        "random_strength": 0.1,
        "l2_leaf_reg": 10,
        "subsample": 0.9,
        "random_state": 71,
        "num_boost_round": 10,  # 350# 早く終わらすために小さくしてる
        "early_stopping_rounds": 10,
    }
    params_catboost["cat_features"] = list(
        range(len(args["num_feature"]), len(args["features"]))
    )

    # cross-validation
    run_cv(args, params_catboost, name="catboost", n_fold=4)

    # 学習データ全体を使う場合（validation set無いからearly stoppingは使えない。cvの時より精度下がるが可能性あり）
    run_all(args, params_catboost, name="catboost")
