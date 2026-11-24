# RoboTwin LaWAM Stage 1 新規学習：30,000更新

2026-11-15〜16の実行。ユーザーの指示により、各checkpointの評価結果を記録した上で、最終30,000更新以外のcheckpoint重みを削除する。削除後、中間checkpointからの再開・再評価はできない。

## データと分割

- RoboTwin `robotwin_merged`。データはSSD1の `foveated-world-models/research/official-data-contract-audit-20261113/robotwin-qfix-data`。動画は元データを参照し、状態の四元数の符号連続性を補正済み。
- 1.6秒間隔の現在・未来2フレーム、256×256、30fpsでは48フレーム間隔。ロボットID 1。状態は実16成分を最大32次元にパディングし、損失は有効成分のみ。
- episode分割：訓練26,125、検証1,375、重複なし。評価512例と訓練用probe 2,048例について、補正済みデータから再生成した状態・差分との一致を確認。詳細と参照先ハッシュは[data-contract.json](data-contract.json)。
- データの実体と過去の評価キャッシュはこの文書への同梱物ではない。正確な保存先とSHA256はデータ契約を参照。

## 学習条件

[実行時に解決された設定](resolved-training-config.yaml)を保存した。主な条件：seed 42、batch 64、30,000更新、AdamW、学習率最大3e-4、線形warmup 10,000、weight decay 0.01、bf16 mixed、勾配clip 1.0、GB10。DINOv3 ViT-Bは事前学習済み重みを読み固定し、IDM・VAE・画像予測器・gは新規初期化。公式コミット `c12168d9077edea67fe7c1b70d08b80b8e65a9c8` のモデルコードを使用し、gの出力のみ `2 * tanh` に変更。

未来DINO特徴の損失はSmoothL1（beta 0.1）、状態差分gは有効成分のMSE、係数1、VAEのKL係数は5e-5。四元数補正はデータ側の処理。実行時の依存関係は[environment-freeze.txt](environment-freeze.txt)。step 0からの再起動は[launch.json](launch.json)参照。進捗と損失の記録は[progress.jsonl](progress.jsonl)。

## 評価と結果

- [checkpoint-summary.csv](checkpoint-summary.csv)：0〜30,000の1,000更新ごと、全31件。各介入の損失、差、episode単位bootstrap 95%区間、PC1、SNR、KL、checkpointのSHA256を記録。
- [results/](results/)：各checkpointの詳細JSON。512例それぞれの画像・状態誤差、z介入、全体統計、幾何、checkpoint・評価コードのハッシュを保存。5,000更新ごとの外部線形probeも該当stepに含む。
- [manifest.json](manifest.json)：元結果JSONとcheckpointのハッシュ、削除対象・実行結果。元結果とコピーの整合を検証するための記録。

FP32、同一512例で、z=μ、ゼロ、平均、別エピソードz、現在画像のみからのμを比較。画像特徴と高変化patchのMSE、持続予測を記録。gは有効状態成分で評価し、位置・姿勢・グリッパー別の結果も残す。5,000更新ごとの外部線形probeは別の訓練2,048例でfitし検証512例で測定する。公式配布gは状態表現が異なるため直接比較しない。

最終30,000更新では第一主成分21.49%、SNR>1が11方向、画像特徴MSE 0.21807。zゼロで34.94%、別episodeへ交換で87.98%の誤差増。配布重みの同じ画像512例における画像MSE 0.31343、第一主成分22.23%、z交換46.60%増は参考値。zが未来画像を見て得られるStage 1の条件付き評価であり、制御成功率や未観測データへの汎化を示すものではない。99%集中の改善を単一条件に帰属できる実験設計ではない。

最終重みは実行runの `checkpoints/step030000.ckpt` に保持。SHA256 `17ad6ec1786af67f572daecb127b9a06a90c18088c3e8f540af634cc9d203877`。学習途中の各重みは削除対象だが、各詳細評価JSONと評価キューは残す。
