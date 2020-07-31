# -*- coding: utf-8 -*-
"""
LightGBM + cross-validation で回帰モデル作成
Usage:
    $ conda activate py37
    $ python run_lgb_reg.py -c config/run_lgb_reg.py.casual.yml  # ymlでパラメータ指定
    $ python run_lgb_reg.py --is_pred_only  # 予測のみ実行する居合
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

# from model_nn import ModelNN
from model_lgb import ModelLGB
from runner import Runner
import util
from util import Submission


def get_args() -> dict:
    """入出力ファイルのパスや特徴量のカラム名を設定"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c", "--config", type=str, default="../tests/config/run_lgb_reg.py.casual.yml",
    )
    ap.add_argument(
        "-is_p_o",
        "--is_pred_only",
        action="store_const",
        const=True,
        default=False,
        help="予測のみ実行する場合",
    )
    args = vars(ap.parse_args())
    args["input_cv_dir"] = None
    args["del_features"] = None
    args["orig_submit_csv"] = None

    # 特徴量とlgmのパラメなど指定(yamlでロード)
    with open(args["config"]) as f:
        config = yaml.load(f)
    args.update(config)

    if "num_feature" in config and "categorical_feature" in config:
        args["features"] = [*config["num_feature"], *config["categorical_feature"]]
    else:
        args["features"] = None

    args["transform"] = np.expm1  # submission csvの後処理（add_feature_v4は対数化してるので元戻すため）
    os.makedirs(args["output_dir"], exist_ok=True)
    print(args)
    return args


def run_cv(args, params_lgb, name="lgb", n_fold=4) -> None:
    """cross-validation"""
    # モデル学習予測をとりまとめるRunnerオブジェクト作成
    runner = Runner(
        name,
        ModelLGB,
        args["features"],
        params_lgb,
        target=args["target"],
        n_fold=n_fold,
        train_csv=os.path.join(args["input_dir"], "train.csv"),
        test_csv=os.path.join(args["input_dir"], "test.csv"),
        output_dir=args["output_dir"],
        split_type=args["split_type"],
        val_eval=params_lgb["metric"],
        input_cv_dir=args["input_cv_dir"],
        del_features=args["del_features"],
    )

    # cross-validationで学習
    _, va_cv_scores_mean = runner.run_train_cv()  # 学習したモデルも返すが使わないので _ にしてる
    print("va_cv_scores_mean", va_cv_scores_mean)

    # test_csv予測
    runner.run_predict_cv()  # 予測結果はpklファイルに保存するのでreturnはなし

    # submit csv作成
    if args["orig_submit_csv"] is None:
        _ = Submission.output_submit(
            os.path.join(args["output_dir"], f"{name}-test.pkl"),
            os.path.join(args["output_dir"], f"{name}-submission.csv"),
            transform=args["transform"],
        )
    else:
        _ = Submission.output_submit(
            os.path.join(args["output_dir"], f"{name}-test.pkl"),
            os.path.join(args["output_dir"], f"{name}-submission.csv"),
            transform=args["transform"],
            orig_submit_csv=args["orig_submit_csv"],
        )

    # 各foldのモデルでfeature importance可視化
    for i in range(n_fold):
        ModelLGB("", {}).save_plot_importance(
            os.path.join(args["output_dir"], f"{name}-{i}.model"),
            png_path=os.path.join(
                args["output_dir"], f"{name}-{i}-plot_importance.png"
            ),
        )


def run_all(args, params_lgb, name="lgb") -> None:
    """
    学習データ全体を使う場合（validation set無いからearly stoppingは使えない。cvの時より精度下がるが可能性あり）
    """
    # モデル学習予測をとりまとめるRunnerオブジェクト作成
    runner = Runner(
        name,
        ModelLGB,
        args["features"],
        params_lgb,
        target=args["target"],
        train_csv=os.path.join(args["input_dir"], "train.csv"),
        test_csv=os.path.join(args["input_dir"], "test.csv"),
        output_dir=args["output_dir"],
        split_type=args["split_type"],
        val_eval=params_lgb["metric"],
        input_cv_dir=args["input_cv_dir"],
        del_features=args["del_features"],
    )
    # 学習
    _ = runner.run_train_all()  # 学習したモデル返すが使わないので _ にしてる

    # test_csv予測
    runner.run_predict_all()  # 予測結果はpklファイルに保存するのでreturnはなし

    # submit csv作成
    if args["orig_submit_csv"] is None:
        _ = Submission.output_submit(
            os.path.join(args["output_dir"], f"{name}-all-test.pkl"),
            os.path.join(args["output_dir"], f"{name}-all-submission.csv"),
            transform=args["transform"],
        )
    else:
        _ = Submission.output_submit(
            os.path.join(args["output_dir"], f"{name}-all-test.pkl"),
            os.path.join(args["output_dir"], f"{name}-all-submission.csv"),
            transform=args["transform"],
            orig_submit_csv=args["orig_submit_csv"],
        )

    # feature importance可視化
    ModelLGB("", {}).save_plot_importance(
        os.path.join(args["output_dir"], f"{name}-all.model"),
        png_path=os.path.join(args["output_dir"], f"{name}-all-plot_importance.png"),
    )

    # train_csv予測 横軸時系列でtrain setの正解と予測plot
    runner.run_predict_all_train()


def run_predict_only(args, name="lgb", n_fold=4) -> None:
    """
    モデルロードして予測のみ行う
    ※昔作成したモデルファイルから予測したいとき用
    """
    # モデル学習予測をとりまとめるRunnerオブジェクト作成
    runner = Runner(
        name,
        ModelLGB,
        args["features"],
        {},
        target=args["target"],
        n_fold=n_fold,
        train_csv="",
        test_csv=os.path.join(args["input_dir"], "test.csv"),
        output_dir=args["output_dir"],
        split_type=args["split_type"],
    )

    if n_fold != "all":
        pkl = os.path.join(args["output_dir"], f"{name}-test.pkl")
        csv = os.path.join(args["output_dir"], f"{name}-submission.csv")
        model = os.path.join(args["output_dir"], f"{name}-0.model")
        png_path = os.path.join(args["output_dir"], f"{name}-0-plot_importance.png")

        # test_csv予測
        runner.run_predict_cv()  # 予測結果はpklファイルに保存するのでreturnはなし

    else:
        pkl = os.path.join(args["output_dir"], f"{name}-all-test.pkl")
        csv = os.path.join(args["output_dir"], f"{name}-all-submission.csv")
        model = os.path.join(args["output_dir"], f"{name}-all.model")
        png_path = os.path.join(args["output_dir"], f"{name}-all-plot_importance.png")

        # test_csv予測
        runner.run_predict_all()  # 予測結果はpklファイルに保存するのでreturnはなし

    # submit csv作成
    if args["orig_submit_csv"] is None:
        _ = Submission.output_submit(pkl, csv, transform=args["transform"],)
    else:
        _ = Submission.output_submit(
            pkl,
            csv,
            transform=args["transform"],
            orig_submit_csv=args["orig_submit_csv"],
        )

    # 0番目のfoldのモデルでfeature importance可視化
    ModelLGB("", {}).save_plot_importance(
        model, png_path=png_path,
    )


if __name__ == "__main__":
    args = get_args()

    # lgbのパラメータ
    params_lgb = args["params_lgb"]
    if "categorical_feature" in args:
        params_lgb["categorical_feature"] = args["categorical_feature"]

    if args["is_pred_only"]:
        # 予測のみ実行
        run_predict_only(args, name="lgb", n_fold=4)
        run_predict_only(args, name="lgb")
    else:
        # cross-validation
        run_cv(args, params_lgb, name="lgb", n_fold=4)
        # 学習データ全体を使う場合（validation set無いからearly stoppingは使えない。cvの時より精度下がるが可能性あり）
        run_all(args, params_lgb, name="lgb")
