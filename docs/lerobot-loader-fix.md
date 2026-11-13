# LeRobot原本を使うStage 1ローダーへの修正

2026-11-13。ユーザーの修正指示を受け、既定の学習経路を `data.format: lerobot_v3` に変更した。問題は、公式モデルのz計算そのものではなく、独自NPZの作成・読み込み時に導入したデータ処理のずれ。**そのずれがzの弱さの原因だったかは、まだ確認できていない。**

## 修正内容

| 項目 | 旧実験 | 修正後の既定経路 |
|---|---|---|
| 入力 | 独自JSONL＋状態NPZ、原本MP4 | LeRobot v3のmeta・Parquet・MP4を直接読む |
| 手先状態 | xyz＋rot6D＋gripper、両腕20次元 | 原本のxyz[m]＋RPY[rad]＋gripper、両腕14次元 |
| 状態正規化 | なし | train全観測だけでmin/maxを計算し、validation/testへ固定適用 |
| 画像 | bicubic resize＋黒帯で256×256 | uint8のまま最大横縦比4:3の中心crop、bilinear antialias resizeで256×256 |
| 欠損 | 派生NPZ作成時に補完・区間分割 | 不正状態・同期失敗はエラー。暗黙に補完・除外しない |
| 再開 | データ表現の違いを検査しない | checkpointのデータ契約を照合。不一致・旧契約なしの再開を拒否 |

原本の列・meta・動画は変更していない。読み出し時だけ左右それぞれ `[x,y,z,roll,pitch,yaw,gripper]` を結合し、min-max変換する。rot6D化、NPZ生成、Lanceへの変換、追加ダウンロードは不要。モデルの最大状態容量32は維持し、公式collatorが有効14次元だけを損失のmaskに含める。

min-maxは公式と同じ `2*(x-min)/(max-min)-1`、定数成分は0、範囲外のclipなし。正規化した始点・終点の差分を既存の状態補助損失へ渡す。式・画像変換の参照は公式コミット `c12168d9077edea67fe7c1b70d08b80b8e65a9c8` の [state_action.py](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/starVLA/dataloader/gr00t_lerobot/transform/state_action.py) と [video.py](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/starVLA/dataloader/gr00t_lerobot/transform/video.py)。

使用中のtop動画は実際には1280×720。これを中央960×720へcropし、256×256へresizeする。640×480は未取得の左右カメラに関するmetaの値。カメラは従来と同じtopだけを選び、metaに列挙された未取得カメラを暗黙に取得しない。

## 維持した条件と実装の範囲

データ分割は従来の動画シャード単位・タスク層化・SHA256順を維持。配布timestampに基づく1.2秒の最近傍と短い末尾ペアも維持した。train 1,444エピソード／9,201ペア、validation 160／950、test 170／1,081。

モデル本体、z、損失式、KL係数、補助損失係数は変更していない。batch 2、LR 3e-4、warmup 5,000、10,000更新など、完了済みfinal10000の実行設定を維持する。従来の設定ファイルが短時間用テンプレートのwarmup 10,000／1,000更新を継承していたため、RoboMIND設定に実行済みの値を明示した。

新実装は **LeRobot保存形式を読むLaWAM向けアダプター**。Hugging Faceの `LeRobotDataset` クラスや公式StarVLA `LeRobotSingleDataset` クラスへ置き換えたという意味ではない。既存の時刻ペア選択を維持し、部分取得されたtop映像を読むため、ParquetはPyArrow、MP4はPyAVで読む。学習用の別データ形式は作らない。

公式FR3設定には姿勢キー名 `_rotvec` とRPYを示すコメントが混在し、原本RPYと同じ意味かは未確認。そのため別の姿勢表現へ再変換せず、原本のRPYとして明示した。公式の全データ混合・姿勢意味・学習手順まで完全再現したとは扱わない。RPYの角度境界の不連続も残る。

旧 `robomind.py` と `prepare_robomind.py` は過去の診断結果を再現するため保持し、`configs/stage1_robomind_legacy.yaml` で明示的に選ぶ。旧NPZ前提のresearch評価スクリプトは旧実験用であり、修正後のcheckpointへそのまま適用しない。既存flow pilotも旧NPZ入力を明示するよう修正した。

## 検証

- `pytest -q tests` は49件成功（timmの既存非推奨警告1件）。
- 合成LeRobot原本で、RPYとgripperの保持、train限定統計、範囲外非clip、定数成分0、有効14次元mask、状態差分、黒帯の不在を検査。
- 全1,774エピソードの時刻列、全11,232ペア、episode/group集合が旧NPZ経路と完全一致。旧データに欠損・補完がなかったことも確認。
- 各splitから8ペア、計48画像を抽出。旧監査で記録した正確な動画PTSから独立にデコードし、公式crop/resizeを適用した画像との一致を検査。
- GPU 1上で実DINO＋小型IDM/LaWMを2更新し、保存後3更新目へ再開。全パラメータの有限性、optimizer状態、データ契約保存を確認。
- 単体テストで旧契約・異なる契約からのresumeを拒否することを確認。

実行コマンドと結果は [verification.json](../research/results/lerobot-loader-fix-20261113/verification.json) と以下を参照。smoke用checkpointは一時ディレクトリに作成し、検証後削除する。これは接続検査であり、zの改善や学習収束の検証ではない。修正後の本学習はまだ開始していない。

```sh
.venv/bin/python -m pytest -q tests
CUDA_VISIBLE_DEVICES=1 .venv/bin/python tests/smoke_robomind.py \
  --legacy-root /data2/yuki_sadasue/datasets/RoboMIND-lerobot/preprocessed \
  --report research/results/lerobot-loader-fix-20261113/verification.json
```

## 次回の実行

```sh
CUDA_VISIBLE_DEVICES=1 uv run --locked python train_stage1.py \
  --config configs/stage1_robomind.yaml \
  --output /data2/yuki_sadasue/runs/foveated-world-models/stage1-lerobot-rpy14-minmax
```

新規runとして開始する。旧final10000からの `--resume` は入力表現・画像分布が違うため拒否する。出力には `data_contract.json` と `resolved_config.yaml` を保存し、checkpointにも同じ契約を埋め込む。契約には原本Parquet・info・episode metaのhash、splitのepisode ID、正規化統計、画像処理、時刻選択条件を含む。動画の内容hashまでは再開時に照合しない。
