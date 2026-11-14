# 公式配布モデルとローカル1万更新の勾配比較条件

2026-11-14。公式配布revision `bd993da2a0861afaac5a95ac86d2555b1313ab8c` の同梱YAML・README、現行qfixデータの `meta/modality.json`、公式RLinf Stage-1の損失計算を読み直した独立監査。GPU実行・学習は行っていない。

## 何に対する勾配かを分ける

計算経路は `IDMの32次元出力p → LayerNorm → mu/logvar → sampled z → decoder`。これらの勾配は異なる量で、まとめて「z勾配」と呼ばない。

| 微分対象 | 意味 | 画像損失から | KLから |
|---|---|---|---|
| decoderへ渡すサンプルz | decoderの損失が入力コードの変化にどれだけ敏感か | 接続 | **非接続** |
| μ | 事後平均へ戻る勾配 | 接続 | 接続 |
| log分散l | サンプリングノイズの幅へ戻る勾配 | 接続 | 接続 |
| LayerNorm前のp | 正規化とVAE headを通ってIDM出力へ戻る勾配 | 接続 | 接続 |
| encoderの各パラメータ | 実際にoptimizerが更新する対象の勾配 | 接続可能 | 接続可能 |

KLはμとlog分散から計算する枝で、実現したサンプルzを経由しない。`autograd.grad(KL,z)` がNoneなのは正しい。したがって「画像＋KL」のz勾配は画像単独と同じ。KLと画像の競合を測るならμ・log分散・pで比較する。[公式VAE](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/vq.py#L67)

同一サンプリング `z=mu+exp(l/2)*epsilon` について、画像損失の微分には以下の恒等式が成り立つ。

```text
dL_image/dmu = dL_image/dz
dL_image/dl  = 0.5 * exp(l/2) * epsilon * dL_image/dz
```

clampがある場合のlは、clamp後の実際にサンプリングへ使うlog分散を指す。clamp前のlogitを微分対象にすると、飽和領域に追加の0勾配が入る。KL項まで加えたμ/log分散の勾配は上式と同一ではない。

## 比較する尺度

`L_batch=(1/B) Σ_i L_i` なので、画像特徴の平均方法が同じなら、activation各行のraw勾配には `1/B` が含まれる。B=4の全条件で揃え、次を両方報告する。

- 実際のbatch平均損失に対する全activationのFrobenius norm。
- `B * raw_gradient[i]` の例ごとのnormの分布。これが例単独の平均画像損失に対応する。batchをまたぐ相互作用がないことが前提。

4例の勾配ベクトルを合計してからnormを取ると、別々のz行の勾配を誤って相殺する。パラメータ勾配をbatchで合算する操作とも区別する。複数のepsilon drawではnormの平均だけでなく、勾配ベクトルのdraw平均のnormも記録すると、ノイズによる方向の不一致を切り分けられる。

zの座標スケールが異なればraw勾配normも変わる。加えて `||sigma * dL_i/dz||` を測ると、事後標準偏差単位の微小変化への感度を表せる。これは固定μ・σで `z=mu+sigma*eta` と置いた時の `dL/deta` であり、モデル全体の表現不変な指標ではない。raw z norm・σ・μの例間分散を一緒に提示し、勾配が大きいほど有用なzという解釈をしない。

特に今回のようにLayerNorm前のpが大きい場合、pへの勾配は小さくても、zへの勾配は小さいとは限らない。`p gradient` と `z gradient` の数値をモデル間で取り違えると、減衰箇所の結論が変わってしまう。

### 小さいactivation勾配は、パラメータの学習停止を意味しない

`p=Wq`、`h=LN(p)` とする。epsilonを無視できる範囲では、正の定数cによる `p→cp` はhを変えず、`g_p=∂L/∂p` は概ね `g_p/c` になる。しかしqをc倍した場合、`g_W=g_p qᵀ` は変わらない。Wをc倍した場合には、`g_q=Wᵀg_p` が変わらない。一般にencoderのパラメータ勾配は `J_phi(p)ᵀ g_p` なので、前段の大きいJacobianが小さいactivation勾配を補い得る。

したがって、ローカルの正規化前標準偏差19270と公式122の比から得る約1/158は、**LayerNormの局所微分に現れる尺度の比**である。同じ比でencoderの更新が減った、または学習が止まったという証拠ではない。正規化が実際に大きさ方向の入力変化を抑えるという有限差分の結果と、更新量の主張を分ける必要がある。

Adamも `m/sqrt(v)` により一様な勾配倍率をかなり打ち消す。ただしepsilon、clipping、weight decay、履歴に依存し、パラメータの再尺度化に完全不変ではない。例えばWをc倍するとAdamの絶対更新がほぼ同じでも、Wの大きさに対する相対更新は小さくなる。よって「小さい勾配だから停止」も「Adamなので尺度は無関係」も過剰な説明になる。公式配布ファイルにはoptimizer履歴がないため、過去の実更新量を再構成できたとは扱わない。

より適切な追加比較は以下。

- **同じ現在・未来変更に対する `g_zᵀ Δz` と実際の出力・損失差。** 可逆線形座標変換 `z'=Az` では、勾配は `A^(-T)g_z`、変化は `AΔz` となり内積は不変。有限の大きい介入では一次近似の限界があるので、実際の出力差も併記する。
- **`sqrt(gᵀ C_mu g)` と `sqrt(gᵀ Sigma_noise g)` を分ける。** C_muはμの全共分散行列、Sigma_noiseは事後ノイズの共分散。前者は観測した平均コードの変動への局所感度、後者はノイズへの局所感度である。共分散を正しく変換すれば可逆線形な再表現に不変だが、情報量や行動意味を測る値ではない。
- **共通DINO座標で `∂L_image/∂u_T_encoder_input` を測る。** 予測教師は `u_T.detach()` として固定し、未来特徴のencoder入力側だけを微分する。これならlatent座標の再尺度化によらず、前段Jacobianも含む全経路を比較できる。教師をdetachしないと、単なるtarget側の誤差微分が混ざる。

## 揃える条件と結果の呼び方

同じ32組、同じ順序、B=4、同じ画像特徴・現在/未来時刻、同じ損失reduction、同じ精度、同じepsilonを両モデルへ使う。公開設定同士では画像SmoothL1の切替幅0.1で一致。KL係数はVAE既定値5e-5で、同梱YAMLも上書きしない。[配布YAML](https://huggingface.co/jialei02/lawam_lam/blob/bd993da2a0861afaac5a95ac86d2555b1313ab8c/dino_large_vae.yaml)

モデルを凍結し、activationだけを微分する実験は、**現在の重みにおける勾配の測定**。optimizer更新を伴う追加学習や、公式が行ったfine-tuningの実測ではない。学習用のzサンプリング経路を使っても、画像augmentationや学習streamを省く場合はその条件を明記する。Stage 2では公式posterior IDMはteacherとして凍結されるので、Stage-1の勾配測定とStage-2のteacher更新有無も区別する。

## gの比較は現在の情報では成立しない

配布YAMLの `max_state_dim` は14。ローカルqfixは16有効次元を32次元へpaddingする。qfixの状態順序は以下。

| 成分 | 0始まりの範囲 |
|---|---|
| 左XYZ | 0:3 |
| 左quaternion | 3:7 |
| 左gripper | 7 |
| 右XYZ | 8:11 |
| 右quaternion | 11:15 |
| 右gripper | 15 |

14次元へ単純に切ると右quaternionの1成分と右gripperが欠ける。quaternionをEuler角へ変換すれば次元数は合わせられるが、公式配布gがその角度順・座標系・正規化で訓練された証拠がないため、その変換も正当化できない。現在のupstreamには14次元joint-spaceと16次元EEF quaternionの別設定があり、14という数だけでは意味を特定できない。

配布README/YAMLには検証できる状態成分の契約や正規化統計が揃っていない。よって、**gを含む総勾配の公式対ローカル比較は実施対象から外し、画像＋KLの比較と明記する**。`enc_add_state=False` なので、画像・VAE側は状態契約に依存せず比較できる。gを省くこと自体を、全損失でのfine-tuningの再現と呼ばない。
