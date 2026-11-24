# RoboTwinの差分IDMを学習する

同じEncoderで現在・未来の画像ペアと現在・現在の画像ペアを処理し、出力の差をVAEに渡す。通常規模の基準設定はEncoder 24層、Decoder 12層、batch 64、seed 42、30,000更新。検証は1,000更新ごとで、検証総損失が最小の重みと最新の重みを保存する。[実装](../research/full_difference.py)・[設定](../configs/stage1_difference_30k.yaml)

## このワークステーションでの準備

この手順はGB10とセットアップ済みのSSD1を使う。コマンドはリポジトリのルートで実行する。別のcheckoutで作業する場合は、学習環境のPythonを `LAWAM_PYTHON` に絶対パスで指定できる。既定はcheckout内の `.venv-lawam-stage1/bin/python`。環境に必要なパッケージの版は[依存一覧](../research/requirements-lawam-stage1.lock.txt)にある。

`scripts/difference-stage1 check` が確認する主な入力は、SSD1上の `foveated-world-models` 以下にある。

| 入力 | SSD1上の相対位置 |
|---|---|
| DINOv3の重み | `models/dinov3-vitb16-pretrain-lvd1689m` |
| 四元数補正済みRoboTwinと参照先の動画 | `research/official-data-contract-audit-20261113/robotwin-qfix-data/robotwin_merged` |
| 公式LaWAMのGit checkout、補助依存 | 同じauditディレクトリ内の `LaWAM`、`python-deps`、`pytorch3d` |
| 既定のエピソード分割 | `runs/robotwin-loss-ablation-b4-20261114/kl_mse/episode_split.json` |

各起動で公式コミット `c12168d9077edea67fe7c1b70d08b80b8e65a9c8` から必要なコードを取り出し、状態出力を `2 * tanh` にした版をrunの `upstream/` に保存する。全データとモデル重みはSSD1で管理する。

## 進捗を読む

```bash
scripts/difference-stage1 status
```

引数を省略すると11月24日に始めた本学習の記録とsystemdの稼働状態を表示する。自分のrunでは `scripts/difference-stage1 status my-difference-30k` のように名前を渡す。表示される更新数は最後にログへ記録した更新で、定期検証のCSVは最初の1,000更新を `step=999` と記録する。

## 自分のrunを開始する

```bash
scripts/difference-stage1 check
scripts/difference-stage1 preflight my-difference-check
scripts/difference-stage1 train my-difference-30k --seed 42
```

`check` のGPU利用状況と空き容量を確認する。この構成の過去の単独試運転はピーク約57 GiBで、フルcheckpointは1件約7.5 GiB。共有GPUでは本学習を追加する余裕がある時に試運転する。`preflight` は本番と同じモデル・batchで3更新し、検証2バッチと保存まで実行する。損失・メモリ・更新時間は `status my-difference-check` で確認する。`train` は新しく初期化して開始する。各コマンドには新しいrun名を使う。

設定を変える場合は上記YAMLをコピーし、`preflight` と `train` の両方に `--config 自分の設定.yaml` を渡す。出力、CSVログ、checkpointの保存先はrun名から設定する。実際に使用した設定をrun直下の `training.yaml` に残す。

ログアウト後も続けたい場合は、前景の `train` の代わりに次を使う。

```bash
systemd-run --user --unit=my-difference-30k --setenv="LAWAM_PYTHON=${LAWAM_PYTHON:-$PWD/.venv-lawam-stage1/bin/python}" --working-directory="$PWD" "$PWD/scripts/difference-stage1" train my-difference-30k --seed 42
journalctl --user -u my-difference-30k -f
```

## 中断した学習を再開する

```bash
scripts/difference-stage1 resume my-difference-continued --from-run my-difference-30k
```

元runの解決済み設定と `checkpoints/last.ckpt` を使い、重み、optimizer状態、更新数を復元する。再開先には新しいrun名を指定する。元のログと最良checkpointは元runで参照でき、再開先には以降のログとcheckpointが保存される。総更新数を延ばす場合は `--max-steps 40000` のように**再開前を含む総数**を指定する。データローダーの次のバッチまでの完全一致は未検証。

各runの保存先は `/media/sadasue/SSD1/foveated-world-models/runs/<run名>/`。進捗は `progress.jsonl`、検証結果は `metrics/version_*/metrics.csv`、重みは `checkpoints/` にある。[研究結果と過去の比較資料](README.md)
