# -*- coding: utf-8 -*-
"""
LightGBM + cross-validation + Optunaで回帰モデルチューニング
Usage:
    $ conda activate py37
    $ python run_lgb_reg_tuning.py -c params/run_lgb_reg_tuning.py.casual.yml  # ymlでパラメータ指定
    $ python run_lgb_reg_tuning.py --is_pred_only  # 予測のみする場合
"""
import argparse
import csv
import os
import traceback
import shutil

import numpy as np
import pandas as pd
import optuna
import yaml

import run_lgb_reg
from model_lgb import ModelLGB
from runner import Runner
from util import Submission


def write_csv(file, save_dict):
    """
    dictを読み書き
    https://qiita.com/nabenabe0928/items/fcdf2c81e5fff3364c6f
    """
    save_row = {}

    with open(file, "w") as f:
        writer = csv.DictWriter(
            f, fieldnames=save_dict.keys(), delimiter=",", quotechar='"'
        )
        writer.writeheader()

        k1 = list(save_dict.keys())[0]
        length = len(save_dict[k1])

        for i in range(length):
            for k, vs in save_dict.items():
                save_row[k] = vs[i]

            writer.writerow(save_row)


def check_n_sample(args, n_fold, split_type="Kfold"):
    """
    train setのサンプル数(行数)計算
    lgmはサンプル数に依存するパラメ(min_child_samples)がある
    min_data_in_leaf < n_sample / num_leaves でないと、[LightGBM] [Warning] No further splits with positive gain, best gain: -inf が止まらなくなる
    """
    runner = Runner(
        None,
        None,
        args["features"],
        None,
        args["target"],
        n_fold=n_fold,
        train_csv=os.path.join(args["input_dir"], "train.csv"),
        test_csv=os.path.join(args["input_dir"], "test.csv"),
        split_type=split_type,
    )
    # train_x = runner.load_x_train()
    # return train_x.shape[0]
    tr_idx, va_idx = runner.load_index_fold(0)
    return len(tr_idx)
    # validation setの数にしないとvalidation set評価時に[LightGBM] [Warning] No further splits with positive gain, best gain: -inf　が出る？
    # return len(va_idx)
    # validation setの数でもちょこちょこ[LightGBM] [Warning]  出るので0.7倍する？これしても最後の全データで学習時にWarning出ることあり。。。
    # return int(len(va_idx) * 0.7)


def get_args() -> dict:
    """入出力ファイルのパスや特徴量のカラム名を設定"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        type=str,
        default="../tests/config/run_lgb_reg_tuning.py.casual.yml",
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

    args["transform"] = np.expm1  # submission csvの後処理（対数化してるので元戻すため）
    print("\nargs:\n", args, "\n")
    os.makedirs(args["output_dir"], exist_ok=True)
    return args


class Objective(object):
    def __init__(self):
        self.args = get_args()
        self.best_score = 10000.0
        self.n_fold = 4
        self.n_sample = check_n_sample(self.args, self.n_fold)
        self.params = args["params_lgb"].copy()
        self.best_dir = os.path.join(self.args["output_dir"], "best")
        os.makedirs(self.best_dir, exist_ok=True)

    def get_lgm_params(self, trial) -> dict:
        """
        Get parameter sample for Boosting (like XGBoost, LightGBM)
        https://github.com/Y-oHr-N/OptGBM/blob/master/optgbm/sklearn.py#L194-L221
        https://nykergoto.hatenablog.jp/entry/2019/03/29/%E5%8B%BE%E9%85%8D%E3%83%96%E3%83%BC%E3%82%B9%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0%E3%81%A7%E5%A4%A7%E4%BA%8B%E3%81%AA%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E6%B0%97%E6%8C%81%E3%81%A1
        Args:
            trial(trial.Trial):
        Returns:
            dict: parameter sample generated by trial object
        """
        params = self.params.copy()

        params["feature_fraction"] = trial.suggest_discrete_uniform(
            "feature_fraction", 0.1, 1.0, 0.05
        )

        params["max_depth"] = trial.suggest_int("max_depth", 1, 7)

        params["num_leaves"] = trial.suggest_int(
            "num_leaves", 2, 2 ** params["max_depth"]
        )
        # See https://github.com/Microsoft/LightGBM/issues/907
        params["min_data_in_leaf"] = trial.suggest_int(
            "min_data_in_leaf", 1, max(1, int(self.n_sample / params["num_leaves"])),
        )

        params["lambda_l1"] = trial.suggest_loguniform("lambda_l1", 1e-09, 10.0)
        params["lambda_l2"] = trial.suggest_loguniform("lambda_l2", 1e-09, 10.0)

        if params["boosting_type"] != "goss":
            params["bagging_fraction"] = trial.suggest_discrete_uniform(
                "bagging_fraction", 0.5, 0.95, 0.05
            )
            params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 10)

        if params["objective"] == "regression":
            params["reg_sqrt"] = trial.suggest_categorical("reg_sqrt", [True, False])

        ## 分類の時用
        # if params["objective"] in ["multiclassova", "binary"]:
        #    params["is_unbalance"] = trial.suggest_categorical(
        #        "is_unbalance", [True, False]
        #    )
        # if params["is_unbalance"] is False:
        #    params["scale_pos_weight"] = trial.suggest_discrete_uniform(
        #        "scale_pos_weight", 0.1, 10.0, 0.05
        #    )
        print("params:", params)
        return params

    def trial_train(self, trial):
        """ Train function for optuna """
        # Hyper parameter setting
        params_lgb = self.get_lgm_params(trial)
        if "categorical_feature" in self.args:
            params_lgb["categorical_feature"] = self.args["categorical_feature"]

        # モデル学習予測をとりまとめるRunnerオブジェクト作成
        name = "lgb"
        runner = Runner(
            name,
            ModelLGB,
            self.args["features"],
            params_lgb,
            target=self.args["target"],
            n_fold=self.n_fold,
            train_csv=os.path.join(self.args["input_dir"], "train.csv"),
            test_csv=os.path.join(self.args["input_dir"], "test.csv"),
            output_dir=self.args["output_dir"],
            split_type=self.args["split_type"],
            val_eval=params_lgb["metric"],
            input_cv_dir=args["input_cv_dir"],
            del_features=args["del_features"],
        )

        # cross-validationで学習
        models, va_cv_scores_mean = runner.run_train_cv()
        print("va_cv_scores_mean", va_cv_scores_mean)

        if self.best_score > va_cv_scores_mean:
            # best_score更新
            self.best_score = va_cv_scores_mean

            # モデルファイルコピー
            for i in range(self.n_fold):
                shutil.copy(
                    os.path.join(self.args["output_dir"], f"{name}-{str(i)}.model"),
                    os.path.join(self.best_dir, f"{name}-{str(i)}.model"),
                )

            # 一応tsvファイルでもモデルのパラメと評価結果の値(rmseとかの値)残しておく
            # モデルのパラメ
            pd.DataFrame.from_dict(models[-1].params, orient="index").to_csv(
                os.path.join(self.best_dir, f"{name}-cv_param.tsv"), sep="\t",
            )
            # 評価結果の値
            pd.DataFrame.from_dict(
                {"va_cv_scores_mean": va_cv_scores_mean}, orient="index"
            ).to_csv(os.path.join(self.best_dir, f"{name}-cv_score.tsv"), sep="\t")

            # test_csv予測
            runner.run_predict_cv()  # 予測結果はpklファイルに保存するのでreturnはなし

            # submit csv作成
            if args["orig_submit_csv"] is None:
                _ = Submission.output_submit(
                    os.path.join(self.args["output_dir"], f"{name}-test.pkl"),
                    os.path.join(self.best_dir, f"{name}-submission.csv"),
                    transform=args["transform"],
                )
            else:
                _ = Submission.output_submit(
                    os.path.join(self.args["output_dir"], f"{name}-test.pkl"),
                    os.path.join(self.best_dir, f"{name}-submission.csv"),
                    transform=args["transform"],
                    orig_submit_csv=args["orig_submit_csv"],
                )

            # 各foldのモデルでfeature importance可視化
            for i in range(self.n_fold):
                ModelLGB("", {}).save_plot_importance(
                    os.path.join(self.args["output_dir"], f"{name}-{i}.model"),
                    png_path=os.path.join(
                        self.best_dir, f"{name}-{i}-plot_importance.png"
                    ),
                )

        return va_cv_scores_mean  # Minimize metrics.

    def __call__(self, trial):
        """ Objective function for optuna """
        try:  # optuna v0.18以上だとtryで囲まないとエラーでtrial落ちる
            min_eval_metric = self.trial_train(trial)
            return min_eval_metric
        except Exception as e:
            traceback.print_exc()  # Exceptionが発生した際に表示される全スタックトレース表示
            return e  # 例外を返さないとstudy.csvにエラー内容が記載されない


if __name__ == "__main__":
    args = get_args()
    study_name = args["study_name"]  # Unique identifier of the study.
    # study = optuna.create_study()
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{args['output_dir']}/{study_name}.db",
        load_if_exists=True,
    )

    if args["is_pred_only"] is False:
        # 最適化実行
        study.optimize(Objective(), n_trials=args["n_trials"])

    study.trials_dataframe().to_csv(
        f"{args['output_dir']}/{study_name}_history.csv", index=False
    )
    print(f"\nstudy.best_params:\n{study.best_params}")
    print(f"\nstudy.best_trial:\n{study.best_trial}")

    # bestディレクトリのモデルを予測のみ実行
    _args = args.copy()
    _args["output_dir"] = os.path.join(_args["output_dir"], "best")
    run_lgb_reg.run_predict_only(_args, name="lgb", n_fold=4)

    # bestパラメータロード
    params_lgb = args["params_lgb"].copy()
    params_lgb.update(study.best_params)
    params_lgb.update(study.best_params)
    print("\nparams_lgb:\n", params_lgb, "\n")

    # best paramでmodel再作成
    args["output_dir"] = os.path.join(args["output_dir"], "best_retrain")
    os.makedirs(args["output_dir"], exist_ok=True)
    # cross-validation
    run_lgb_reg.run_cv(args, params_lgb, name="lgb", n_fold=4)
    # 学習データ全体を使う場合（validation set無いからearly stoppingは使えない。cvの時より精度下がるが可能性あり）
    run_lgb_reg.run_all(args, params_lgb, name="lgb")
