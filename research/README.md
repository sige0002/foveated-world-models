# Stage1予測改善の研究コード

現時点で精度が改善した方式は、固定した現在画像のみの予測に、参照差分IDMのコードを使う補正器を追加する直接MSE回帰である。画像そのものではなく、未来の画像特徴を予測する。元モデルへの本番統合や、実行時の行動条件付き予測の改善は未完了。

**比較基準は公式配布モデルではなく、公式由来のdirect構造をRoboMINDのみで独自に10000更新学習したモデルである。** 公式設定の混合データ・時間間隔・バッチサイズ・学習スケジュールを再現していない。この基準への改善やzの小さな追加効果を、公式LaWAM全般の結果と解釈しない。[ソース・設定の監査](idm/official-source-audit-20261113/report.md)。

## 改善した処理

現在特徴を C、正解未来特徴を Y とすると、処理は次のようになる。

1. 学習済みdecoderの行動入力をゼロにして、土台の予測 `b(C)=decoder10000(C,0)` を作る。このdecoderは固定する。
2. 固定IDM encoderの「現在・未来」のqueryから「現在・現在」のqueryを引き、訓練データで決めた尺度で校正する。別途学習したheadから32次元のposterior mean `z` を得る。
3. 現在特徴、土台の予測、zを補正器へ渡し、`Y_hat=b(C)+correction(C,b(C),z)` を直接MSEで学習する。

元の大きな予測器の知識を維持しながら、追加ネットワークにzの使い方を学ばせる構成である。補正器の出力は全特徴次元を持つ。flowの反復生成は使用しない。IDMの変更と補正器の追加を合わせた結果なので、改善全体を一方だけの効果とは呼ばない。

主要な実装は [prior_correction.py](prior_correction.py)、[wide_head_flow.py](wide_head_flow.py)、[train_cached_decoders.py](train_cached_decoders.py)、[IDM head学習](idm/train_reference_heads.py)。`wide_head_flow.py` は直接回帰とflowの共通構造を提供しており、名前だけで今回の方式をflowと解釈しない。

## 確認済みの結果

| 方法 | 既観測96例のMSE |
|---|---:|
| 元のfinal10000モデル | .103631 |
| 参照差分IDM＋補正器、train512・400更新、3 head/decoder seed平均 | .086709（16.33%減） |
| 同方式、train512・1600更新、seed42のみ | .083304（19.61%減） |

[精度の集計](results/nested512-direct-scaling-summary-20261113.json)。追加のhead学習・補正器学習を含むため、総学習費用を揃えた比較ではない。seed平均は予測ensembleではない。全て正解未来から作るzを使ったStage1再構成であり、64+96例は研究で繰り返し観測済み。カメラ運動の除去、操作意味の識別、未使用データへの汎化、会議論文としての新規性は未証明。

CPU2スレッド・float32・batch1での予測器部分の直接測定は、元モデル338.77ms、補正込み361.65ms（6.75%増）。各12回のwarm測定中央値。画像特徴抽出・IDM・読み込みは含めず、GPUは未測定。[各回の測定値](results/original-decoder-timing-20261113.json)、[ベンチコード](benchmark_original_decoder.py)。

## 内部統合版の実装と検証

[layerwise_condition.py](layerwise_condition.py) は、元の12層の画像処理を一度だけ使い、各層へ `c_l = b_old + U_l z` を渡す研究用decoderである。外付け補正器を包んだ実装ではない。32次元zはそのまま、層別の写像は一つのLinearで計算する。学習対象は393,216パラメータ。元の重みは固定し、初期予測と学習後のz=0予測は元のzero-code priorを保つ。元の通常のz付き予測を初期状態で保存する方式ではない。

これは外付け補正器の改善を等価に移植したものではなく、新しい比較候補である。内部版の精度・速度優位は未確認。本番Stage1/Stage2の既定構造は変更していない。既知の層別条件付けを使うため、この変更だけを論文の新規性とは主張しない。[設計と対照条件](idm/internal-layerwise-condition-design-20261113.md)。

実モデルでの確認は [check_layerwise_condition.py](check_layerwise_condition.py) で訓練例1件・破棄する1更新を実行する。全層の非ゼロ勾配、元重みの不変性、z=0の予測一致を検査する。これは精度評価用の学習ではない。

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/check_layerwise_condition.py \
  --output research/results/reproduce-layerwise-smoke.json
```

同じcacheを使う400更新実験は次のコマンドで選択できる。`--train` を外すと初期状態の確認だけを行う。`--architecture shared` は以前の全層共通写像、`--arm rms_constant` はコードを固定した対照である。層別400更新はCPUの初期検証後、CUDA実行を開始した。完了結果が出るまでは精度未確認として扱う。

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python research/train_early_condition.py \
  --source . --output runs/reproduce-layerwise-real400 \
  --architecture layerwise --arm real --threads 2 --train
```

[複数指標評価](evaluate_prediction_metrics.py) は固定済みの外付け補正器をMSE、元の視覚損失Smooth L1、MAE、cosine similarity、特徴変化量で選んだ領域のMSEで比較する。変化量上位領域は物理的なロボットのマスクではない。対象は既観測64/96例で、真の未来から作ったzを使う。

固定seed42・既観測96例での追加評価は以下の通り。全体MSEだけでなく、元の視覚損失と特徴変化が大きい領域でも改善した。ただし外付け版の結果であり、内部統合版の結果ではない。[全指標・例別値・条件付きbootstrap区間](results/prediction-multimetric-20261113/results.json)。

| 指標 | 元モデル | 外付け400更新 | 外付け1600更新 |
|---|---:|---:|---:|
| Smooth L1（beta=.1、低いほど良い） | .138106 | .125132 | .123954 |
| MAE（低いほど良い） | .177759 | .163787 | .162724 |
| Cosine similarity（高いほど良い） | .944639 | .953753 | .955456 |
| 特徴変化上位10%領域のMSE | .469058 | .376009 | .358206 |

[実チェックポイントの1更新検査](results/layerwise-condition-smoke-20261113.json)は成功し、[回帰テスト](results/layerwise-validation-20261113.json)は142件通過。これらは内部統合版の実装検査であり、汎化精度・速度の改善を示すものではない。

CUDAが利用できる環境では学習コマンドに `--device cuda` を追加する（`CUDA_VISIBLE_DEVICES=''` は外す）。float32・highest設定を維持し、20更新ごとに経過時間を記録、100更新ごとにadapterとoptimizerを `latest.pt` へ保存する。旧CPU実行には途中保存がなく、CUDA実行は初期状態から開始した。CUDAの[実モデル検査](results/layerwise-cuda-smoke-20261113.json)でも初期・更新後zero-code予測の一致と各層への勾配を確認した。他用途のGPU常駐プロセスは停止していない。

## 再実行と成果物

学習済みcheckpoint、画像特徴cache、元動画、論文全文はGitへ含めない。JSON/CSVは研究時点の測定・出典・ハッシュを保存する記録であり、記載されたローカル絶対パスが別環境で存在するとは限らない。リポジトリのcloneだけでは実データ実験を再実行できない。

元データから抽出してハッシュ整合性を確認したcacheとcheckpointを配置した環境では、次のように直接補正器を学習できる。出力先は新規ディレクトリを指定する。

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python research/train_cached_decoders.py \
  --cache runs/reference-idm42-nested512-prior10000 \
  --output runs/reproduce-prior-residual400 \
  --variants prior_residual --steps 400 --batch-size 8 \
  --dim 128 --layers 2 --threads 2 --seeds 42
```

速度比較には、さらに元final10000のweights/config/native-code cacheとFFN4補正器のcheckpointが必要。

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python research/benchmark_original_decoder.py \
  --source . --output research/results/reproduce-original-decoder-timing.json
```

速度比較時は競合する計算を避ける。別worktreeで実行したソースの同一コピーと出所は [source manifest](results/published-worktree-sources-20261113.json) に記録した。研究履歴・未採用案・進行中の実験は [詳細記録](../docs/stage1-latent-flow.md)、100件の論文整理は [サーベイ](../surveys/flow-architecture/survey.csv) を参照。追加adapterとbounded innovationのソースは検証用であり、上表の改善を得た方式と混同しない。

小規模な実装検査はCPUで実行できる。

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests research --ignore=research/source_snapshots
```
