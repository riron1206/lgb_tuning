import os

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
import matplotlib.pylab as plt

from model import Model
from util import Util


class ModelCatBoost(Model):
    def train(self, tr_x, tr_y, va_x=None, va_y=None):

        # ハイパーパラメータの設定
        params = dict(self.params)
        cat_features = params.pop("cat_features")
        print(cat_features.index)

        # データのセット
        validation = va_x is not None
        dtrain = Pool(tr_x, tr_y, cat_features=cat_features)
        if validation:
            dvalid = Pool(va_x, va_y, cat_features=cat_features)

        # 学習
        self.model = CatBoost(params)
        if validation:
            self.model.fit(
                dtrain, eval_set=dvalid, use_best_model=True,  # 最も精度が高かったモデルを使用するかの設定
            )
        else:
            self.model.fit(
                dtrain, use_best_model=True,  # 最も精度が高かったモデルを使用するかの設定
            )

    def predict(self, te_x):
        return self.model.predict(te_x,)

    def save_model(self, model_dir="../model/model"):
        model_path = os.path.join(model_dir, f"{self.run_fold_name}.model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # best_ntree_limitが消えるのを防ぐため、pickleで保存することとした
        Util.dump(self.model, model_path)

    def load_model(self, model_dir="../model/model"):
        model_path = os.path.join(model_dir, f"{self.run_fold_name}.model")
        self.model = Util.load(model_path)

    @classmethod
    def save_plot_importance(
        cls,
        model_path,
        feature_names,
        png_path=None,
        is_Agg=True,
        height=0.5,
        figsize=(8, 16),
    ):
        """catboostのモデルファイルからモデルロードしてfeature importance plot"""
        model = Util.load(model_path)
        if is_Agg:
            import matplotlib

            matplotlib.use("Agg")

        # 特徴量の重要度を取得する
        feature_importance = model.get_feature_importance()
        # print(feature_importance)
        # print(feature_names)
        # 棒グラフとしてプロットする
        plt.figure(figsize=figsize)
        plt.barh(
            range(len(feature_importance)),
            feature_importance,
            tick_label=feature_names,
            height=height,
        )
        plt.xlabel("importance")
        plt.ylabel("features")
        plt.grid()
        if png_path is not None:
            plt.savefig(
                png_path, bbox_inches="tight", pad_inches=0,
            )  # bbox_inchesなどは余白削除オプション
        plt.show()
