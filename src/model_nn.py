"""
ニューラルネットのモデル
"""
import os
import random
import joblib

# tensorflowの警告抑制
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    ReLU,
    PReLU,
    Activation,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import *
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ModelNN:
    def __init__(self, run_fold_name="", params={}) -> None:
        """コンストラクタ
        :param run_fold_name: ランの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None
        self.scaler = None

    def build_model(self, input_shape):
        """モデル構築"""
        model = Sequential()
        model.add(Dense(self.params["units"][0], input_shape=input_shape))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(self.params["dropout"][0]))

        for l_i in range(1, self.params["layers"] - 1):
            model.add(Dense(self.params["units"][l_i]))
            model.add(PReLU())
            model.add(BatchNormalization())
            model.add(Dropout(self.params["dropout"][l_i]))

        model.add(Dense(self.params["nb_classes"]))
        model.add(Activation(self.params["pred_activation"]))
        if self.params["optimizer"] == "adam":
            opt = Adam(learning_rate=self.params["learning_rate"])
        else:
            opt = SGD(
                learning_rate=self.params["learning_rate"], momentum=0.9, nesterov=True
            )

        model.compile(
            loss=self.params["loss"], metrics=self.params["metrics"], optimizer=opt,
        )
        self.model = model

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # 乱数固定
        ModelNN().set_tf_random_seed()

        # 出力ディレクトリ作成
        os.makedirs(self.params["out_dir"], exist_ok=True)

        # データのセット・スケーリング
        validation = va_x is not None
        scaler = self.params["scaler"]  # StandardScaler()
        scaler.fit(tr_x)
        tr_x = scaler.transform(tr_x)
        # ラベルone-hot化
        tr_y = to_categorical(tr_y, num_classes=self.params["nb_classes"])

        # モデル構築
        self.build_model((tr_x.shape[1],))

        hist = None
        if validation:
            va_x = scaler.transform(va_x)
            va_y = to_categorical(va_y, num_classes=self.params["nb_classes"])

            cb = []
            cb.append(
                ModelCheckpoint(
                    filepath=os.path.join(
                        self.params["out_dir"], f"best_val_loss_{self.run_fold_name}.h5"
                    ),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                )
            )
            # cb.append(ModelCheckpoint(filepath=os.path.join(self.params["out_dir"], f"best_val_acc_{self.run_fold_name}.h5"),
            #        monitor="val_acc",
            #        save_best_only=True,
            #        verbose=1,
            #        mode="max",
            #    )
            # )
            cb.append(
                EarlyStopping(
                    monitor="val_loss", patience=self.params["patience"], verbose=1
                )
            )
            hist = self.model.fit(
                tr_x,
                tr_y,
                epochs=self.params["nb_epoch"],
                batch_size=self.params["batch_size"],
                verbose=2,
                validation_data=(va_x, va_y),
                callbacks=cb,
            )
        else:
            hist = self.model.fit(
                tr_x,
                tr_y,
                epochs=self.params["nb_epoch"],
                batch_size=self.params["batch_size"],
                verbose=2,
            )

        # スケーラー保存
        self.scaler = scaler
        joblib.dump(
            self.scaler,
            os.path.join(self.params["out_dir"], f"{self.run_fold_name}-scaler.pkl"),
        )

        # history plot
        self.plot_hist_acc_loss(hist)

        return hist

    def predict_binary(self, te_x):
        """2値分類の1クラスのみ取得"""
        te_x = self.scaler.transform(te_x)
        pred = self.model.predict(te_x)[:, 1]
        return pred

    def load_model(self):
        model_path = os.path.join(
            self.params["out_dir"], f"best_val_loss_{self.run_fold_name}.h5"
        )
        # model_path = os.path.join(self.params['out_dir'], f'best_val_acc_{self.run_fold_name}.h5')
        scaler_path = os.path.join(
            self.params["out_dir"], f"{self.run_fold_name}-scaler.pkl"
        )
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"INFO: \nload model:{model_path} \nload scaler: {scaler_path}")

    def plot_hist_acc_loss(self, history):
        """学習historyをplot"""
        acc = history.history["acc"]
        val_acc = history.history["val_acc"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(len(acc))

        # 1) Accracy Plt
        plt.plot(epochs, acc, "bo", label="training acc")
        plt.plot(epochs, val_acc, "b", label="validation acc")
        plt.title("Training and Validation acc")
        plt.legend()
        plt.savefig(
            f"{self.params['out_dir']}/{self.run_fold_name}-acc.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()
        plt.close()

        # 2) Loss Plt
        plt.plot(epochs, loss, "bo", label="training loss")
        plt.plot(epochs, val_loss, "b", label="validation loss")
        plt.title("Training and Validation loss")
        plt.legend()
        plt.savefig(
            f"{self.params['out_dir']}/{self.run_fold_name}-loss.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()
        plt.close()

    @staticmethod
    def set_tf_random_seed(seed=0):
        """
        tensorflow v2.0の乱数固定
        https://qiita.com/Rin-P/items/acacbb6bd93d88d1ca1b
        ※tensorflow-determinism が無いとgpuについては固定できないみたい
         tensorflow-determinism はpipでしか取れない($ pip install tensorflow-determinism)ので未確認
        """
        ## ソースコード上でGPUの計算順序の固定を記述
        # from tfdeterminism import patch
        # patch()
        # 乱数のseed値の固定
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)  # v1.0系だとtf.set_random_seed(seed)


def test_func():
    """
    テスト駆動開発での関数のテスト関数
    test用関数はpythonパッケージの nose で実行するのがおすすめ($ conda install -c conda-forge nose などでインストール必要)
    →noseは再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行する
    $ cd <このモジュールの場所>
    $ nosetests -v -s --nologcapture <本モジュール>.py  # 再帰的にディレクトリ探索して「Test」や「test」で始まるクラスとメソッドを実行
      -s付けるとprint()の内容出してくれる
      --nologcapture付けると不要なログ出さない
    """
    import pandas as pd
    import seaborn as sns

    def _test_train(df, feats, target_col, params):
        num_folds = 2
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        for n_fold, (train_idx, valid_idx) in enumerate(
            folds.split(df[feats], df[target_col])
        ):
            t_fold_df = df.iloc[train_idx]
            v_fold_df = df.iloc[valid_idx]
            train_x, train_y = (
                t_fold_df[feats],
                t_fold_df[target_col],
            )
            valid_x, valid_y = (
                v_fold_df[feats],
                v_fold_df[target_col],
            )
            model_cls = ModelNN(n_fold, params)
            # 学習
            hist = model_cls.train(train_x, train_y, valid_x, valid_y)
            # 予億
            print(model_cls.predict_binary(valid_x))

    def _test_pred(df, feats):
        # ファイルからロードして予億
        params = dict(out_dir="tmp")
        model_cls = ModelNN(0, params)
        model_cls.load_model()
        model_cls.predict_binary(df[feats])

    df = sns.load_dataset("titanic")
    df = df.drop(["alive"], axis=1)
    # 欠損最頻値で補完
    for col in [col for col in df.columns if df[col].isnull().any()]:
        column_mode = df[col].mode()[0]
        df[col].fillna(column_mode, inplace=True)

    for col in [
        "sex",
        "embarked",
        "who",
        "embark_town",
        "class",
        "adult_male",
        "alone",
        "deck",
    ]:
        df[col], uni = pd.factorize(df[col])
    target_col = "survived"
    feats = df.columns.to_list()
    feats.remove(target_col)

    params = dict(
        out_dir="tmp",
        scaler=StandardScaler(),
        layers=3,
        units=[128, 64, 32],
        dropout=[0.3, 0.3, 0.3],
        nb_classes=2,
        pred_activation="softmax",
        loss="categorical_crossentropy",
        optimizer="adam",
        learning_rate=0.001,
        metrics=["acc"],
        nb_epoch=10,
        patience=5,
        batch_size=256,
    )

    _test_train(df, feats, target_col, params)
    _test_pred(df, feats)


# test_func()
