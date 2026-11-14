# RoboTwin Stage 1：学習経路とデータの独立監査

2026-11-14。未改造公式commit `c12168d9077edea67fe7c1b70d08b80b8e65a9c8`、ローカルのg出力幅patch、旧10000更新run、実行中の4条件のbaseline、準備されたcausal v2 runnerを確認した。GPU・optimizer更新・学習コードの変更は行っていない。[機械可読な結果・ファイルhash・8組の詳細](robotwin-training-path-audit-20261114.json)

**この監査でz不使用を説明する新たなdetach・凍結・optimizer除外・画像ペアの入れ替えは見つからなかった。** 一方、時間差の混在とIDM/decoder間の独立した画像拡張は実在する。公開実装にもある条件なので、原因としては独立実験が必要である。

## 学習経路の確認

| 項目 | ソースと確認結果 |
|---|---|
| 勾配・train/eval | `core/lam_model.py:412–460` はencoder→FP32 VAE→画像decoder/gを直接接続。DINOだけが `requires_grad=False`、eval、no_grad。`core/vjepa_encoder.py:287–392`。IDM全体を凍結する経路は見つからない。 |
| optimizer | `core/lam_lightinng.py:380–440,901–923` は全trainable parameterの割当漏れ・重複を検査。biasとnormはweight decay対象外。runnerは同じ関数を使用する。 |
| 数値・clip | 公開YAMLとローカルともBF16 mixed、VAE内部FP32、global norm clip=1。初回の記録ではIDM/g/画像decoderとも勾配あり、DINOは0。clipは全成分へ共通係数を掛け、画像勾配だけを除く処理ではない。 |
| 画像と状態の平均 | `core/lam_lightinng.py:643–706`、`core/utils/utils.py:40–116`。画像は1例196,608要素を平均。gは有効16成分を平均し、残り16成分のpaddingはmaskで除外。同じ公開実装であり、要素数比をIDM勾配比に読み替えない。 |
| mask | `core/utils/modules.py:67–81`。queryは全contextへattention可能。これは時間ごとの遮断ではなくmodalityのmask。`enc_add_state=False` なので状態の誤りが入る経路は主にg損失である。 |
| ローカル差分 | [runner](../train_robotwin_stage1.py:110)はcheckoutのdiffを[保存patch](../patches/official-state-output-scale.patch)と照合。モデルの変更はgの `tanh` を `2*tanh` にする箇所。出力微分も変わるため、学習への影響は未排除。 |

上記の公式ファイルはすべて `LaWAM/latent_action_model/` 以下。DINO以外の全パラメータの実更新をこの監査で再実行したわけではない。[固定commit](https://github.com/RLinf/LaWAM/tree/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model)

## 実際の500 batchを再構成

`starVLA/dataloader/gr00t_lerobot/datasets.py:2243–2252,2563–2590` のhashとNumPy乱数、保存episode split、metadataのframe範囲から先頭2000組を再構成した。各4組のidentity hashは実行中baselineの `training_stream.jsonl` と **500/500 batchで一致**。2000組すべて別のペアで、1904エピソードから取得されていた。

2000組の両端を実際のparquetでも照合し、episode IDとframe indexは全件一致した。timestamp差と `frame差/30` の最大ずれは約1.53マイクロ秒で、浮動小数点の丸めの範囲だった。

現在・未来のindexは `base` と `min(base+48, length-1)`。動画と絶対状態が同じ末尾処理を使う。ソースは `latent_action_model/data_loader/lerobot_dataset.py:206–232`、`starVLA/dataloader/gr00t_lerobot/datasets.py:1541–1559,1616–1635`。

| 測定 | 結果 |
|---|---:|
| 最初2000組で48フレーム未満 | 23.15% |
| 同一フレームindex | 17/2000 |
| train全体を一様抽出した場合の短いhorizon比率 | 21.71% |
| train全体の同一index比率 | 0.4523% |

したがって固定1.6秒だけの学習ではない。ただし「ほぼ全例が現在→現在」という破損ではなく、公開loaderにも同じ仕様がある。修復判定を時間差や動きの大きさで分ける価値はある。

## 動画・状態・quaternion

最初の8組について、元parquetのtimestamp・状態とcam_high動画を読み、固定commitのtimestamp decoderで確認した。表示MAEは0–255画素での現在/未来差。画像の意味とロボット状態の厳密な同期を全データで検証したものではない。

| episode | current→future | 時間差(s) | 画像MAE | 生の状態差norm |
|---:|---:|---:|---:|---:|
| 3410 | 161→209 | 1.6000 | 23.5297 | 1.18444 |
| 17677 | 121→129 | 0.2667 | 1.9724 | 0.17086 |
| 5492 | 88→136 | 1.6000 | 36.4691 | 0.79833 |
| 16726 | 89→137 | 1.6000 | 16.0190 | 0.36929 |
| 23897 | 70→118 | 1.6000 | 15.8566 | 0.20593 |
| 21716 | 185→233 | 1.6000 | 3.5939 | 0.10814 |
| 19584 | 38→86 | 1.6000 | 4.7861 | 1.00012 |
| 18318 | 139→179 | 1.3333 | 3.5748 | 0.92463 |

すべて状態は有限、quaternion normは約1、動画fpsは約30。timestamp差は選択したindex差と一致した。動画読込は `starVLA/dataloader/gr00t_lerobot/video.py:138–203` で、相対timestampにepisodeの動画開始時刻を足してnearest frameを選ぶ。

さらに実際の2000組のquaternionを調べた。`dot(q_current,q_future)<0` は左右それぞれ4組。該当例の物理的な姿勢差は約135–176度で、負の内積かつ姿勢差10度未満の例は0。したがって、この集合には「ほぼ同じ姿勢のqと−qを引いて巨大な差分を作る」徴候はない。負の内積だけでは、連続した大きな回転と符号不連続を区別できない。既存の全6,077,247フレームのqfix監査では隣接フレームの負の内積は左右とも0。個別の8件はJSONに保存した。

状態の並びは左XYZ・左quat・左gripper・右XYZ・右quat・右gripper。XYZ/gripperをmin-max正規化し、gripperの符号を反転、quaternionはそのまま使う。`data_config.py:1042–1110` とmetadataが一致し、`collate.py:39–55` は正規化後にfuture−currentを計算する。

## 画像拡張

正確な `video_aug.py:60–115` をCPUの同一画像 `[B=2,T=2,3,256,256]` で確認した。現在と未来は同じcrop/jitter、batch内も同じ変換を共有し、IDM用viewとdecoder用viewは独立。評価時の2viewは同一になる。時間ごとに無関係な変換を掛けるバグではない。

独立viewは公開設定の `dual_view_aug=True` に従う。少ない更新でこの不変性を学ぶ難しさは未検証であり、`dual_view_aug=False`だけを変える比較で分離できる。拡張を原因と確定した結果ではない。

## causal v2 runnerの独立レビュー

対象：[train_robotwin_stage1_causal_v2.py](../train_robotwin_stage1_causal_v2.py)。レビューしたhashはJSONの `causal_v2_helper_review.runner_sha256` に記録した。

- `install_sampling_control:26–71` は元のVAEを1回実行し、元のepsilonも消費する。乱数の再現用forwardはforkしたRNG/no_grad内。μ-onlyではdetachしていない実際のμを返し、KLは維持する。
- `on_before_backward:492–542` は同じgraphに `autograd.grad(..., retain_graph=True)` を使い、その後通常のbackwardを行う。nodeとzを区別し、sampled zに対するKLのNoneを許容する。geometry hookは返り値を変更せず、replay中のno_grad出力も除外する。
- β=0とg=0は損失を0倍するので、nodeの経路は接続したまま勾配0になる。g=0でもAdamWのweight decayで未使用headの値が動く可能性はあるが、gからIDM/画像への学習信号は0。未学習gの状態MSEを修復指標にはしない。
- μ-onlyのlog分散は画像・gから学ばず、KLで学ぶ。ログのnoise fractionは使われていない事後ノイズの指標であり、この条件の実際のdecoder入力のノイズ量ではない。
- official_initは439テンソルのencoder/VAE/画像decoderをstrict loadし、local gと乱数を維持する。optimizerは新規。公式14次元g・optimizer履歴を復元した条件ではなく、新しいgの干渉で悪化する可能性もある。

公式VAEを使ったCPU・B4/Q1/D32の小規模検査では、sampled hook＋診断用勾配計算を加えても、元のforward/backwardとz・総parameter/node勾配が完全一致した。全条件のforward後CPU乱数状態も一致し、μ-onlyの画像→log分散勾配はNone、μへの勾配は非ゼロ、β=0/g=0の対応するnode勾配は厳密な0だった。これはhelperの確認であり、GPU/full-modelのbaseline replay検査を代替しない。

静的レビューで実行を妨げる問題は見つからなかった。実機preflightではfresh条件の初期重み、全条件の入力・拡張・epsilon、baselineの数値一致を照合する必要がある。旧runnerの診断時 `train_loss` 表示をhash文字列で上書きする問題は既知で、学習計算には影響せず、v2では修正されている。
