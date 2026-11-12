# Stage1予測改善の研究コード

現時点で精度が改善した方式は、固定した現在画像のみの予測に、参照差分IDMのコードを使う補正器を追加する直接MSE回帰である。画像そのものではなく、未来の画像特徴を予測する。元モデルへの本番統合や、実行時の行動条件付き予測の改善は未完了。

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
