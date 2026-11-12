# Stage1の潜在flow予測実験

研究ブランチ: `research/stage1-latent-flow-matching`。
目的は未来のDINO特徴の予測精度を改善すること。中間画像の生成や行動チャンクの導入は、この実験の前提にしない。現段階の実装は比較基盤であり、新手法としての独自性や改善は未証明。

## 実装

[潜在flow](../vendor/lawam/latent_action_model/core/utils/latent_flow.py) は、現在特徴 `h` と潜在行動 `z` を条件に速度場を学習する。`x0 = h + sigma * epsilon`、`x1 = future`、`x_tau = (1-tau)*x0 + tau*x1` とし、速度教師は `x1-x0`。生成計算の時間 `tau` と、データの物理時間幅は異なる。

- `direct`: 従来の一回のTransformer予測。既存設定の既定値。
- `refinement`: 現在特徴から同じ更新ネットワークを反復する決定論的予測。終点回帰で学習する。
- `flow` / `current`: 現在特徴の周囲のノイズから未来特徴へ輸送。
- `flow` / `noise`: ガウスノイズから未来特徴へ輸送。現在特徴は条件入力として保持。

flowの学習ではランダムな生成時刻の速度MSEを最適化し、IDMのKLと状態補助損失を加える。推論はEuler積分。速度の大きさを潰さないよう、速度出力にはLayerNormをかけない。検証では未来特徴を速度場に渡さず、実際にサンプリングした終点を評価する。

学習時の一段の終点推定は、正解を含む補間点を使う。そこで `train/interpolation_*` と表示し、真の未来予測精度として扱わない。`val/feature_mse` は生成した終点の誤差、`val/copy_feature_mse` は現在特徴をそのままコピーした誤差。`val/recon_loss` は従来と同じ特徴損失。生成モデルでbest-of-Nだけを報告しない。

IDMと状態予測器の初期値は、同じseedなら全方式で一致するようにした。ただし共同学習後のIDMは方式ごとに異なる。現在のStage1評価は正解未来を見たIDMコードを条件にするため、実行時の予測能力の証明ではない。

[UPSTREAM.json](../vendor/lawam/UPSTREAM.json) は公開コードの元ハッシュを保持し、変更した2ファイルのハッシュを [RESEARCH_PATCHES.json](../vendor/lawam/RESEARCH_PATCHES.json) に記録する。ハッシュ検査は元版と研究用変更の対応を検証する。

## 実行

リポジトリのルートで実行する。`/path/to/...` は既存のローカルモデルと準備済みデータに置き換える。ダウンロードは行わない。

```sh
.venv/bin/python -m pytest -q tests

.venv/bin/python research/run_latent_flow_pilot.py \
  --vision-model /path/to/dinov3-vitb16-pretrain-lvd1689m \
  --prepared-root /path/to/preprocessed \
  --output runs/latent-flow-pilot-seed42 \
  --accelerator cpu --steps 100 --val-batches 32 --seed 42
```

[pilotスクリプト](../research/run_latent_flow_pilot.py) は小型構成（IDM/予測器各1層、幅64）、同じデータ分割、augmentationなしで5方式（1段の残差回帰を含む）を順番に学習する。出力ディレクトリの再利用を拒否する。各方式の設定・ログ・checkpointと、集約した `comparison.json` を保存する。100 step・検証先頭32ペア・1 seedは予備実験にすぎず、方式の優劣を確定しない。

通常サイズの設定は [現在特徴flow](../configs/stage1_flow.yaml)、[ノイズflow](../configs/stage1_flow_noise.yaml)、[決定論的反復](../configs/stage1_refinement.yaml)。`train_stage1.py` の `--vision-model`、`--train-manifest`、`--val-manifest`、`--output` で環境に合わせる。本学習は別途設定・実行する。接続確認専用の [flow smoke設定](../configs/stage1_flow_smoke.yaml) もある。

## 論文として成立させるための判断基準

単なるflow化を新規性にしない。次の既存研究は直接関連する。

- [VFMF, §3–4](https://arxiv.org/html/2512.11225v1): VFM特徴の確率的予測と特徴圧縮。高次元DINO特徴をそのまま生成する難しさも議論。現在の直接特徴flowは、圧縮を使う強い方式との比較が必要。
- [RLA-WM, §3](https://arxiv.org/html/2605.07079v1): DINO差分から圧縮表現を学び、その表現を行動条件付きflowで予測する。残差化と低次元化自体を独自性にできない。
- [Flow-JEPA, §4](https://arxiv.org/html/2608.29029v1): 現在観測と行動系列を条件に将来の潜在系列をflowで共同生成する。未来系列全体のflow生成も既存。

### 現在特徴始点の位置づけ

条件 `h` を固定して `y_tau = x_tau-h` と置けば、現在特徴始点のflowは、`y0 = sigma*epsilon` から `y1 = future-h` へのflowと等価な輸送問題になる。したがって「現在特徴から開始する」ことだけは、新しい確率モデルの証明にならない。実装上のパラメータ化、最適化、計算効率が変わる可能性を実測する。

### 検証する仮説

1. **残差の予測をしやすくする効果**: current-flowの優位性が、確率的生成ではなく現在特徴の保存に由来する可能性。コピー、1段の残差回帰、反復回帰と比較する。
2. **不確実性を表現する効果**: 条件が十分なら直接回帰で足り、条件が不十分なときにflowが効く可能性。IDMの潜在幅、固定したIDM、未来を使わない行動条件を別々に評価する。単一サンプルMSEだけで多峰性を評価しない。
3. **局所操作を予測する効果**: 全特徴の平均誤差が下がっても、静的背景の再現だけかもしれない。独立に定義した操作領域・物体位置・接触結果で改善を測る。予測誤差が大きい画素を選んで操作領域と呼ばない。

これらは研究仮説であり、新規性の確定ではない。採択水準の主張には、既存方式との差、失敗機序、独立した操作評価、複数データ・seedでの再現が必要。

### 本比較の未完了項目

- 同じ学習済みIDMを凍結した予測器単独比較と、共同学習の比較を分ける。
- SmoothL1/MSEの損失差、最終LayerNormの差、モデル容量を揃える対照を追加する。
- 同じ推論時間／計算量で比較し、積分1/2/4/8/16回を測る。反復回帰は逆伝播も反復するため学習計算量も異なる。
- ノイズ強度と始点の種類を交差して比較する。現在のpilotはcurrent=0.1、noise=1.0なので始点だけの比較ではない。
- 時間幅別・episode別に評価する。固定seedの複数サンプル、平均予測、分布評価を分離し、episode単位の信頼区間を出す。
- 現在の先頭32ペアから全validation・未見タスクへ広げる。test splitは選択に使わない。
- 実行可能な行動条件で評価し、未来画像からの情報による改善と区別する。

## 検証記録

2026-11-13: CPU上で実RoboMIND＋凍結DINOを用いたflowの2 step学習、サンプリング評価、checkpoint保存、3 step目へのoptimizer込み再開が成功。23件のテストが成功。GPUはCUDA初期化時のメモリ不足で失敗し、既存の本学習には介入していない。単体テストは一定速度の厳密輸送、行動条件への勾配、固定ノイズでの再現、未来教師を受け取らない生成API、不正設定、IDM初期値の一致を検査する。

接続テストは性能改善の証拠ではない。

100 step、seed 42、validation先頭32ペアの初期結果（小さいほどよい）:

| 方式 | 特徴MSE |
|---|---:|
| 現在特徴をコピー | 0.24791 |
| 直接予測 | 0.86478 |
| 1段残差回帰 | 0.25714 |
| 4段反復回帰 | 0.25629 |
| 現在特徴からのflow、4段 | 0.26676 |
| ノイズからのflow、4段 | 1.50916 |

この予備結果はflowの優位性を支持しない。現在特徴を保存する3方式は近く、いずれもコピーを上回らない。直接予測も未収束と考えられ、これを本サイズモデルの性能と解釈しない。始点保存の効果と確率的学習の効果を分離する必要がある。

設定・分割ハッシュ・測定値を含む [保存した実験記録](../research/results/latent-flow-pilot-20261113.json)。

詳細ログ: `runs/latent-flow-pilot-20261113/comparison.json`、`runs/latent-flow-residual-pilot-20261113/comparison.json`。

追加診断は [評価スクリプト](../research/evaluate_latent_flow.py) で行う。validation全域から等間隔の64ペアを選び、同じノイズを使って正しいz・ゼロz・次のペアのzを比較する。サンプル平均のMSEと各サンプルのMSEを分離する。ゼロz／別zは条件依存の診断であり、実行可能な反実仮想の正解を与える実験ではない。

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/evaluate_latent_flow.py \
  --runs runs/latent-flow-pilot-seed42/direct \
         runs/latent-flow-pilot-seed42/flow_current \
  --output runs/latent-flow-pilot-seed42/paired-evaluation.json \
  --pairs 64 --samples 4
```


### validation全域の64ペア診断

全方式で同じペア・同じ教師特徴を使用したことを検査済み。flowは固定seedで4サンプル。

| 方式 | サンプルMSE | 4サンプル平均のMSE | 別ペアのzでのMSE |
|---|---:|---:|---:|
| direct | 0.860298 | — | 0.860301 |
| refinement | 0.213813 | — | 0.213814 |
| flow_current | 0.224591 | 0.216648 | 0.224591 |
| flow_noise | 1.500026 | 0.733001 | 1.500026 |
| residual | 0.214481 | — | 0.214481 |

コピーMSEは0.205827。全方式で別ペアのzへの交換による誤差差はごく小さい。これは初期モデルが動画ごとのzの違いを十分に利用していない可能性を示すが、未収束の小型モデルなので、本サイズStage1やflow一般の欠陥とは結論しない。

[ペア別の測定値](../research/results/latent-flow-paired-20261113.json) を保存した。サンプル誤差＝平均予測誤差＋サンプル分散の恒等式も数値的に検査済み。現在特徴flowの平均予測もコピーを上回らないため、分散を表現したことだけで精度改善とは言えない。CPU所要時間は他プロセスと共存した診断値であり、正式な速度比較ではない。

次の判断に必要なのは、十分に学習した同一IDMを固定した予測器比較。現時点ではこの案を有効な新手法として採用しない。

## 固定IDMによる予測器単独比較

[特徴抽出](../research/cache_stage1_features.py) と [予測器の学習](../research/train_cached_decoders.py) を追加した。動いている本学習から名前付きstep checkpointを読み出し、元ファイルが削除されても再利用できる重みスナップショットを保存する。`last.ckpt` は書換え中の可能性があるため入力として拒否する。checkpoint・manifest・キャッシュのハッシュを記録する。

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/cache_stage1_features.py \
  --run /path/to/existing-stage1-run \
  --checkpoint /path/to/existing-stage1-run/checkpoints/epoch=0-step=2900.ckpt \
  --output runs/frozen-idm-step2900 --train-pairs 128 --val-pairs 64

CUDA_VISIBLE_DEVICES='' .venv/bin/python research/train_cached_decoders.py \
  --cache runs/frozen-idm-step2900 --output runs/frozen-idm-decoder-comparison \
  --steps 400 --batch-size 8 --dim 128 --layers 2 --seeds 42
```

- trainとvalidationはepisodeが重ならないことを検査する。両者とも全時間位置から等間隔に選び、augmentationなし・観測済みendpointだけを使う。
- DINOとIDMは凍結し、IDMのGaussian meanを保存する。同じキャッシュを全予測器に与え、ミニバッチ選択用乱数を生成モデルの乱数から分離する。
- 直接回帰は最終LayerNormあり・なしの2種類。1段残差回帰・4段反復回帰・現在特徴flowと比較する。残差／反復／flowには同じ初期ネットワーク重みを与える。
- 回帰は終点MSE、flowは速度MSE。IDMを含まないためKLや状態補助損失はこの段階では使わない。
- 100 stepごとのvalidation、最後のtrain再構成誤差、z交換診断、パラメータ数、実行時間を保存する。flowは4サンプルの平均予測と各サンプルの誤差を別記する。best-of-Nは使わない。
- 128ペア・小型予測器・1 seedは機序を調べる予備実験であり、実データ全体や本サイズへの汎化を証明しない。元の本サイズ予測器の値は参考診断であり、学習量・容量が違う新予測器の公平な対照ではない。

使用開始した2,900 step checkpointはwarmup 5,000 stepの途中であり、IDMの収束を仮定しない。100 stepの共同学習との違いを確かめる段階で、最終判断には成熟したcheckpointと複数seedが必要。固定IDMの有効性自体をz交換診断で検査する。

### 固定IDM比較の結果（2026-11-13）

2,900 stepのIDMを凍結。train 128 episode／128ペア、validation 64 episode／64ペア。新予測器は幅128・2層、400更新、batch 8。主要3方式は各910,848パラメータで、同じ初期値・同じ入力・同じミニバッチ順序を使った。

| 方式 | seed数 | validation MSEのseed平均 | seed間標準偏差 |
|---|---:|---:|---:|
| コピー | — | 0.205827 | — |
| 直接回帰、最終LNあり | 1 | 0.231413 | — |
| 直接回帰、最終LNなし | 1 | 0.212655 | — |
| 1段残差回帰 | 3 | **0.181039** | 0.000515 |
| 4段反復回帰 | 3 | 0.181222 | 0.001728 |
| flow、4段・個々のサンプル | 3 | 0.204777 | 0.001777 |
| flow、4サンプル平均 | 3 | 0.194197 | 0.001719 |

平均化したflowは16回のネットワーク評価を必要とし、1回の残差回帰より計算量が多い。flowと残差回帰のMSE差は全3 seedで正（flowが悪い）。学習したseedに条件づけたepisode bootstrapでも、単一サンプルの差の95%区間は[0.02119, 0.02641]、4サンプル平均の差は[0.01068, 0.01581]。この区間は未知の学習seed全体や別データへの汎化を保証しない。

seed 42のflowについて積分回数1/2/4/8/16も評価した。4回から16回に増やすと、個々のサンプルMSEは0.20448から0.20976へ悪化した。積分回数不足だけでこの差を説明する証拠はない。

**判断:** この小規模・固定IDM・特徴MSEの条件では、直接DINO特徴空間のcurrent-flowを採用する根拠は得られなかった。残差回帰を次の強い対照にする。flow一般の否定、最適化の収束、データ全体での優劣は主張しない。特徴圧縮や幾何を考慮した既存flow方式ともまだ比較していない。

**別の重要な観測:** 固定IDMのvalidationコードは、平均ベクトルの二乗ノルム6.262に対し、動画間の分散の合計が0.00000911だった。元の本サイズ予測器も、正しいzでのMSE 0.143712と別episodeのzでのMSE 0.143711がほぼ同じ。このcheckpointと評価条件では、動画ごとのコードの違いが予測に寄与する証拠がない。

ゼロzでは元の予測器のMSEが0.212439に悪化するが、これは「行動情報を使っている」証拠にならない。ほぼ一定の非ゼロコードがAdaLNの共通条件として働き、ゼロ化がその共通条件を壊す可能性がある。これは構造からの解釈であり、学習全体を通じた潜在崩壊を確定したものではない。

[集計・区間](../research/results/frozen-idm-summary-20261113.json)、[seed 42の全記録](../research/results/frozen-idm-seed42-20261113.json)、[seed 43/44](../research/results/frozen-idm-seeds43-44-20261113.json)、[積分回数](../research/results/frozen-idm-integration-20261113.json)、[元予測器のz交換](../research/results/frozen-idm-reference-20261113.json)、[コード分散](../research/results/frozen-idm-diagnostic-20261113.json)、[キャッシュ来歴](../research/results/frozen-idm-cache-manifest-20261113.json)。

追加した集計・診断の実行入口は [paired集計](../research/summarize_cached_comparison.py)、[元予測器の再評価](../research/evaluate_cached_reference.py)、[積分回数比較](../research/sweep_cached_flow.py)。固定IDM比較を含む25件のテストが成功。
