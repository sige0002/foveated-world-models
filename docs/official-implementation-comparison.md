# 公式LaWAMとの実装・学習条件の比較

追記：この比較後に[公式配布重み自体も評価した](native-posterior-investigation.md)。公式では平均定数zへの差替えでtest MSEが43.3%悪化し、今回のローカルモデルとは異なるz利用を確認した。以下の「未評価」は本比較時点の記録。

その後に確認したカメラ解像度・姿勢規約の留保・保存形式と、前処理変更をまだ実施しないという作業境界は[差分と説明の訂正](local-official-data-contract-gaps.md)を参照。

2026-11-13。**同じローカル重み・実画像・乱数で、公式コードとローカルのdirectモデルはz・予測・損失・勾配が完全一致した。一方、状態の前処理と学習条件は公式と異なる。** 現時点ではモデル本体の移植ミスを支持する証拠はなく、データアダプタの仕様と学習条件を優先して切り分ける。

これは公式**コード**との比較である。使用した重みは`stage1-robomind-main/checkpoints/final.ckpt`（10,000更新、SHA256 `7212d2db040ec4bed5e998287cbff01e59ac516ce14c489433b56f2945974200`）。公式配布重み自体のz定数化評価は未実施であり、以前の「元モデル」の診断を公式配布モデルの結果として扱わない。

## モデル本体の一致

[公式リポジトリ](https://github.com/RLinf/LaWAM)を新しく取得。取得時HEADは固定済みの`c12168d9077edea67fe7c1b70d08b80b8e65a9c8`と同じだった。取り込んだ12ファイルを再照合し、10ファイルは完全一致。残り2ファイルの差分は追加decoderの選択肢、flowの損失分岐、評価指標の追加で、既定のdirect経路は公式と同じ。

公式checkoutとローカルvendorを**別プロセス**で読み込み、実際の学習設定・同一checkpointをstrict loadした。モジュールの読込元も検査した。同じGPUで、validationの2ペアをFP32推論、trainの別の2ペアをBF16学習モードで実行。後者では画像augmentation・VAEサンプリングを有効にしてbackwardまで実施し、optimizer更新は行っていない。再現性のため決定的アルゴリズムとfloat32 matmul precision=`highest`を共通指定した。

| 比較対象 | 結果 |
|---|---|
| augmentation後のencoder/decoder入力 | 推論・学習とも要素ごとに完全一致 |
| encoder出力、z、現在特徴、予測 | 推論・学習とも最大絶対差0 |
| 総損失・共通の損失ログ | 完全一致。推論0.2050842345、学習0.1918053478 |
| 学習時の勾配 | 449個のパラメータテンソルすべてのSHA256が一致、欠落0 |
| ローカルだけの出力 | 評価ログ`feature_mse`と`copy_feature_mse` |

この確認は固定した2バッチの計算同値性であり、全入力・全学習過程や両データローダーの同値性を証明するものではない。LayerNorm、VAEのKL係数既定値5e-5、学習時のサンプリング、推論時のposterior mean、zのdecoder条件入力は公式と同じ。これらをローカル固有の実装不具合として説明する根拠はない。

## 学習条件は一致していない

公式リポジトリの[YAML](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/config/dino_large_vae.yaml)と、公式配布モデルに付属する[YAML](https://huggingface.co/jialei02/lawam_lam/blob/bd993da2a0861afaac5a95ac86d2555b1313ab8c/dino_large_vae.yaml)は異なる。後者も実際の学習履歴を独立に証明する記録ではない。

| 項目 | 公式リポジトリ設定 | 公式配布付属設定 | 今回のローカル学習 |
|---|---|---|---|
| データ指定 | `lam_plus_human` | `lam_plus_human` | RoboMIND FR3 dualのみ |
| batch/process | 64 | 64 | 2、単一GPU、蓄積1 |
| 学習期間 | 10 epoch | 40 epoch | 10,000更新、約20,000ペア提示 |
| 学習率 | 3e-4 | 1e-4 | 3e-4 |
| warmup | 10,000更新 | 10,000更新 | 5,000更新 |
| 画像の時間差 | robot 1.6秒、human 0.4秒 | 同左 | 1.2秒、末尾は短縮 |
| 状態の最大次元 | 32 | 14 | 32、有効20次元 |
| validation上限 | 50バッチ | 100バッチ | `1.0`＝全validation |

ローカルtrainは9,201ペア、1,444 episode、54動画グループ。約20,000ペア提示は約2.17周相当だが、異なるデータ集合の公式epoch数とは直接比較できない。公式の`lam_plus_human`という名前は公開YAMLにあるが、このcommitの`DATASET_NAMED_MIXTURES`には登録がなく、リポジトリ全体の検索でもYAML以外に定義を確認できなかった。公開コードだけから混合内訳・重み・実際のサンプル提示量を確定できない。

## 原因候補になるデータ前処理の差

**状態の表現と正規化が違う。** 公式の[FR3 dual設定](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/starVLA/dataloader/gr00t_lerobot/data_config.py#L1470)は、各腕の位置3・姿勢3（フィールド名`orientation_rotvec`）・gripper1を用い、各stateをデータ統計のmin–maxで[-1,1]へ正規化する。LAM用transformにもこのstate処理が引き継がれる。ローカルは位置をメートルのまま、姿勢をRPY由来のrot6D、gripperを[0,1]として各腕10次元に変換し、統計による正規化をしない。

補助損失は有効次元で平均した状態差分のL2。したがって`lambda_aux=1`が同じでも、単位・次元数・姿勢表現の違いにより各成分の重みとzへの学習信号は異なる。さらに状態予測器は現在状態も受け取るため、zを利用せずに小さい差分を予測する経路もある。この経路自体は公式と共通である。**前処理差は具体的な原因候補だが、それでzがほぼ定数になったという因果確認はまだない。** 正規化の介入を行うならtrainだけで統計を推定し、姿勢表現の変更とは分ける。

**画像の前処理も違う。** 公式FR3経路は横長画像を最大4:3へ中央crop後、linear補間で256×256にresizeする。ローカルは画像全体を保持し、bicubic縮小と黒paddingで256×256にする。同じ後段augmentationでも、対象物の占有率・背景・形状は一致しない。

**欠損補完状態の混入は今回の説明にならない。** ローカルcollateが`original_valid`をstate maskへ反映しない潜在的な問題はあるが、実際の全train 9,201ペア・validation 950ペアの状態端点を調べると、補完されたraw state値はどちらも0件だった。末尾短縮ペアはtrain 1,407、validation 155件。

## 解釈と次の切り分け

「データが悪い」か「モデル実装のバグ」かという二択では絞れない。今回確認したモデル計算は一致し、差は**データをモデルへ渡す実装・学習条件**に存在する。まず状態正規化だけの対照と、有効batch・学習量の対照を分け、通常zが定数zを上回るか評価するのが妥当。LayerNormやKLを変える場合も別条件にする。公式配布重みの通常z／train平均定数z／交換z比較は別途必要であり、今回のコード比較からその成否を推定しない。

## 成果物と再実行

- [比較結果](../research/results/official-implementation-comparison-20261113/comparison.json)
- [checkpoint・source hash・全ペアの欠損検査](../research/results/official-implementation-comparison-20261113/inputs.json)
- [公式の損失・全勾配hash](../research/results/official-implementation-comparison-20261113/official.json)／[ローカル](../research/results/official-implementation-comparison-20261113/local.json)
- [外部設定とデータ処理コードの取得元・hash](../research/results/official-implementation-comparison-20261113/external-sources.json)
- [比較スクリプト](../research/compare_official_stage1.py)

大きい実画像・出力tensorは`/data2/yuki_sadasue/runs/foveated-world-models/official-implementation-comparison-20261113/`に保存し、Gitには含めない。公式checkoutの作成後、リポジトリrootから順に実行する。

```sh
.venv/bin/python research/compare_official_stage1.py prepare --source /tmp/lawam-official-compare-20261113
.venv/bin/python research/compare_official_stage1.py run --source /tmp/lawam-official-compare-20261113 --label official
.venv/bin/python research/compare_official_stage1.py run --source "$PWD/vendor/lawam" --label local
.venv/bin/python research/compare_official_stage1.py compare
```

以前のsource auditには別のGB10 checkpointの記述がある。この報告の対象は上記SHA256の`stage1-robomind-main`。また今回の実設定にある`limit_val_batches: 1.0`は「1バッチ」ではなく全validationを意味する。
