# Stage 1のz利用と外付け／共同学習の比較

2026-11-13。実行前に固定。ユーザー依頼：同条件の外付け版・共同学習版・定数z対照を試し、zの効果が小さい理由を検証する。

## 比較条件

- 4条件×seed42/43/44。direct-real（元decoderと差分IDMの共同学習）、direct-constant（同じ元decoder＋学習可能な共通head(0)）、external-real（固定zero-code prior＋外付け補正器と差分IDMの共同学習）、external-constant（同じ外付け補正器＋学習可能な共通head(0)）。定数条件のencoderは使わず固定する。
- 共通のfinal10000 encoder/decoder、差分のtrain512 RMS校正、新規head初期値、train512、batch8、seedごとのミニバッチ順、MSE、float32 highest、weight decay、gradient clipを用いる。encoder/head/predictor LRは1e-5/1e-4/3e-5。外付け補正器は既存prior_residualのdim128・2層・768幅readout、初期出力0。外付けのLRを結果を見て変更しない。
- 400更新を固定。前の開発集合で延長時の過学習を観測したための選択であり、未観測の事前選択ではない。direct-realは既存400 checkpointの初期重み・データ・batch prefix・設定を検証して再利用する。他9本を新規学習。同更新数であり同FLOPs/同容量ではない。外付けは元decoderを固定、directは更新するという構造上の差が残る。
- 定数対照は同じheadをzero queryから学習して共通zを最適化する。学習済みrealモデルの推論時差替えと別の実験。directのreal/constantは同じdecoder/head初期重み、externalのreal/constantは同じ補正器/head初期重みを使う。
- train512・validation64・observed96・再利用test170で評価。test170は前回に採点済みなので新たな盲検testとは呼ばない。同一タスク群・同一カメラ。評価後の条件追加やstep選択はしない。
- 本学習開始前のユーザー追加指定により、元のfinal10000 encoder＋VAE＋decoderを無更新で同じ特徴・例・precisionにより評価する。通常posterior mean、訓練meanの定数、zero、別例mean、同一画像組posterior meanを比較する。元モデルの同一画像組コードは定数とは限らない。公式配布重みではなくローカルRoboMIND学習重みである。

## 診断と判定

1. 学習済みrealモデルを固定し、通常z、head(0)、訓練z平均、別例z（固定循環置換）、zero zを比較する。MSE/MAE/特徴変化領域MSEに加え、通常出力との差の二乗平均を保存する。誤差が同じことと出力が同じことを区別する。
2. 各splitのz分散・共通成分・有効rank、訓練分布からの平均移動を測る。予測の変化量と実際の未来特徴変化量を比較する。zの非定数性だけで有用性とはしない。
3. trainの現在→未来特徴MSEの三分位で固定した変化量bin、実時間差、元動画shard別に定数化の影響を集計する。testの構成差が小さい利得に関連するか記述する。特徴変化を物理的な動作や接触とは呼ばない。
4. real対constantの学習比較と推論時差替えを合わせる。外付けのみ利得がある場合は予測器側の利用経路の候補、両者とも小さい場合は情報/学習目標/データの不足候補。ただし共同学習でIDMも異なるため、因果的な箇所特定とは呼ばない。
5. seed別差、paired episode bootstrap、動画shard bootstrapを併記する。学習時間・メモリ・trainable parameterを報告する。共有事前学習重み・少数seed/shard・既観測評価の限界を明記する。

GPU smokeで勾配・更新・定数の未来非依存・外付けprior不変・保存/再読込を確認してから本学習を行う。20更新ごとのログと100更新ごとの再開checkpointを設ける。大容量成果物は`/data2/yuki_sadasue/runs/foveated-world-models/z-utility-matched-20261113`へ保存する。
