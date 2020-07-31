import os
import pathlib
from statistics import mean

import numpy as np
import pandas as pd
import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback
import matplotlib.pylab as plt

from model import Model
from util import Util


def run_lgb_cv(df, input_cols, target_col, lgb_params, cv_params):
    """lgbでcross-validation"""
    train_set = lgb.Dataset(df[input_cols], label=df[target_col])
    scores = lgb.cv(lgb_params, train_set, **cv_params)
    score = scores[f"{lgb_params['metric']}-mean"][-1]
    return score


class ModelLGB(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # パラメータの設定
        params = dict(self.params)

        train_params = {
            "num_boost_round": params.pop("num_round"),
            "verbose_eval": params.pop("verbose_eval")
            if "verbose_eval" in params
            else True,
        }
        silent = (params.pop("silent") if "silent" in params else False,)
        categorical_feature = (
            params.pop("categorical_feature")
            if "categorical_feature" in params
            else "auto"
        )

        # データのセット
        validation = va_x is not None
        dtrain = lgb.Dataset(
            tr_x, tr_y, silent=silent, categorical_feature=categorical_feature
        )
        if validation:
            dvalid = lgb.Dataset(
                va_x, va_y, silent=silent, categorical_feature=categorical_feature
            )

        # LightGBMPruningCallbackがあるとvalid_names指定できないしtrainのスコアも出せない
        callbacks = params.pop("callbacks") if "callbacks" in params else None
        if callbacks is not None and isinstance(callbacks[0], LightGBMPruningCallback):
            train_params["valid_sets"] = dvalid
        else:
            if validation:
                train_params["valid_names"] = ["train", "valid"]
                train_params["valid_sets"] = [dtrain, dvalid]
            else:
                train_params["valid_names"] = ["train"]
                train_params["valid_sets"] = [dtrain]

        # 学習
        evaluation_results = {}
        if validation:
            early_stopping_rounds = params.pop("early_stopping_rounds")
            self.model = lgb.train(
                params,
                dtrain,
                categorical_feature=categorical_feature,  # Datasetとtrain両方でcategorical_feature指定しないとWarningでる
                evals_result=evaluation_results,
                early_stopping_rounds=early_stopping_rounds,  # early_stopping_rounds回以上精度が改善しなければ中止
                callbacks=callbacks,
                **train_params,
            )
            # 打ち切ったiterationの数保持
            if "optimum_boost_rounds" in self.params:
                self.params["optimum_boost_rounds"].append(self.model.best_iteration)
            else:
                self.params["optimum_boost_rounds"] = [self.model.best_iteration]
            self.plot_train_metric(evaluation_results, metric=params["metric"])
        else:
            params.pop("early_stopping_rounds")
            if "optimum_boost_rounds" in self.params:
                # 各foldで打ち切ったiterationの数の平均まで学習する
                train_params["num_boost_round"] = int(
                    mean(params.pop("optimum_boost_rounds"))
                )
            print("num_boost_round:", train_params["num_boost_round"])
            self.model = lgb.train(
                params,
                dtrain,
                categorical_feature=categorical_feature,
                callbacks=callbacks,
                **train_params,
            )

    def predict(self, te_x):
        return self.model.predict(te_x)

    def save_model(self, model_dir="../model"):
        model_path = os.path.join(model_dir, f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        Util.dump(self.model, model_path)

    def load_model(self, model_dir="../model"):
        model_path = os.path.join(model_dir, f"{self.run_fold_name}.model")
        self.model = Util.load(model_path)

    def plot_train_metric(
        self,
        evaluation_results,
        metric="rmse",
        model_dir=None,
        is_Agg=True,
        figsize=(15, 4),
    ):
        """Plot metric during training"""
        if is_Agg:
            import matplotlib

            matplotlib.use("Agg")
        plt.figure(figsize=figsize)
        if "train" in evaluation_results:
            plt.plot(evaluation_results["train"][metric], label="train")
        if "valid" in evaluation_results:
            plt.plot(evaluation_results["valid"][metric], label="valid")
        plt.ylabel(metric)
        plt.xlabel("Boosting round")
        plt.title("Training performance")
        plt.legend()

        # 指定なければカレントディレクトリに画像出力
        model_dir = pathlib.Path.cwd() if model_dir is None else model_dir
        plt.savefig(
            os.path.join(model_dir, "training_performance.png"),
            bbox_inches="tight",
            pad_inches=0,
        )  # bbox_inchesなどは余白削除オプション

        plt.show()
        plt.clf()
        plt.close()

    @classmethod
    def save_plot_importance(
        cls, model_path, png_path=None, is_Agg=True, height=0.5, figsize=(8, 16),
    ):
        """lgbのモデルファイルからモデルロードしてfeature importance plot"""
        model = Util.load(model_path)
        if is_Agg:
            import matplotlib

            matplotlib.use("Agg")

        lgb.plot_importance(model, height=height, figsize=figsize)
        if png_path is not None:
            plt.savefig(
                png_path, bbox_inches="tight", pad_inches=0,
            )  # bbox_inchesなどは余白削除オプション
        plt.show()
        plt.clf()
        plt.close()
