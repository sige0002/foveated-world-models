# g を維持した損失・教師の監査

対象は公開ソース `RLinf/LaWAM@c12168d9077edea67fe7c1b70d08b80b8e65a9c8`、保存済み公開重み付属設定、現在の `kl_mse` / continuation。モデル更新・GPU実験は行っていない。論文本文の主張と公開設定、公開重みが実際に訓練された条件は区別する。

## 実際の損失と分母

現行の全robot・1query・B4では、次の合計になる。

```text
L = L_image + beta * mean_batch[0.5 * sum_32(mu² + exp(logvar) - 1 - logvar)]
            + lambda_aux * sum(mask * (g(z, state_now) - delta_state)²) / sum(mask)
```

| 項 | 実装 | 現行の意味 |
|---|---|---|
| 画像特徴 | `F.mse_loss(recon, target)`。元baselineは `F.smooth_l1_loss(..., beta=.1)` | `[B,1,256,768]` 全要素の平均。画素画像ではなく未来DINO特徴 |
| KL | 潜在次元を `sum(-1)`、その後 `mean()` | 32次元を合計、バッチ平均。32次元平均ではない |
| g | masked squared error の総和 / `mask.sum()` | B4×有効16次元=64で割る。32padの無効16次元は分母にも入らない |
| gの時刻mask | `[B,2,32]` を `any(dim=1)` | 始点・終点を二重に数えない |
| gの適用 | `embodiment_id != 0` のrobotだけ | 現行は全robot。混合データではrobot有効成分だけの平均になる |

根拠: [VAE 49–82行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/vq.py#L49)、[画像損失・合計 640–705行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/lam_lightinng.py#L640)、[g損失 102–114行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/utils.py#L102)。

g損失は画像損失の数値そのものへ埋め込まれず、最終目的関数へ加算される。画像decoderとgは同じ接続された `quantized` を受け、画像・g・KLからIDM/VAEへ勾配が戻る。[共有経路 412–455行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/lam_model.py#L412)。`last_kl_loss.detach()` はログ用で、訓練用 `vq_loss` はdetachされていない。

公開版とlocal版でVAE、g loss、collate、状態変換はファイルSHAが一致し、`shared_step` のASTも一致した。[照合結果](loss-reduction-source-equality.json)。CPUで公開関数を実行し、無効padに誤差1000を入れても損失1・pad勾配0、有効16次元のみ平均されることを確認した。[CPU検証](loss-reduction-cpu-check.json)。

## 「5倍」の範囲

公開VAEの既定 `beta=5e-5` に対し、現行combinedは `1e-5`。同じμ、logvar、同じ計算グラフなら、**KL項とKL単独の勾配**が厳密に1/5になる。総勾配、画像/gとの勾配角度、gradient clipping後のベクトル、AdamWによる実パラメータ更新が1/5になるという意味ではない。

KLの `sum_32` を仮に `mean_32` へ変えれば同じbetaでも32倍変わるが、今回その実装差は見つからなかった。B4とB64も双方バッチ平均なので期待目的関数に単純16倍の係数差があるわけではない。勾配の分散・更新回数・データ曝露は別の差である。

`lambda_aux` のPythonコンストラクタ既定は0.2だが、公開YAMLとlocalの実効設定は両方1。これを「localのgだけ5倍」と扱うのも誤り。

SmoothL1の引数 `beta=.1` はKL係数とは別。MSEへの変更も一律の倍率ではない。残差rについてSmoothL1の微分は小残差でr/.1、大残差でsign(r)、MSEは2rであり、残差分布により相対比が変わる。

公開重み付属YAMLは `smooth_l1`、`vq_kwargs` は `layer_norm` のみでKL beta未記載。現在の公開ソース既定と組み合わせれば5e-5になるが、これは当時の実行設定を証明しない。論文の1e-5指定との相違は「論文指定対公開コード既定/初期baseline」と記述する。[重み付属設定 8–35行](https://huggingface.co/jialei02/lawam_lam/blob/bd993da2a0861afaac5a95ac86d2555b1313ab8c/dino_large_vae.yaml#L8)。

## 状態教師とgの確認済み差

現在の16次元は左 `XYZ3 + quaternion4 + gripper1`、右も同じ8次元。XYZとgripperはデータ統計のmin/maxで `[-1,1]` に変換し、gripperは符号を反転する。quaternionはこの設定では正規化変換対象に入らない。その後 `normalized_future - normalized_current` を教師にする。[キーと変換 1050–1110行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/starVLA/dataloader/gr00t_lerobot/data_config.py#L1050)、[min/max式 154–169行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/starVLA/dataloader/gr00t_lerobot/transform/state_action.py#L154)、[差分・mask 49–59行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/data_loader/collate.py#L49)。

これは相対回転quaternionや回転角の誤差ではなく、quaternion各成分の差分MSE。localデータは既存qfixで各episode内の符号不連続を修正済みだが、公開重みが同じ前処理で訓練されたかは未確認。

公開gは `tanh`、localの承認済みpatchは `2*tanh`。[公開g 203–214行](https://github.com/RLinf/LaWAM/blob/c12168d9077edea67fe7c1b70d08b80b8e65a9c8/latent_action_model/core/utils/lam_decoder.py#L203)、[local patch](../../patches/official-state-output-scale.patch)。正規化状態の差分は±2になり得るため、local512教師ではtanhの出力範囲だけで16D MSEに最低0.030720の下限が生じ、2*tanhではその範囲下限は0になる。範囲修正は理由がある一方、勾配・飽和・初期出力も変えるため、公式と完全に同一のgではない。

同じ最終preactivation hと教師tで、g=a*tanh(h)ならMSE勾配は `2*(a*tanh(h)-t)*a*(1-tanh(h)^2)/valid_count`。aを1→2にすると常に2倍または4倍になるわけではない。

公開重み付属設定は状態14次元。現在公開されているRobotwinのキーは16次元で、collateは超過分を先頭から切り落とす。したがって「公開weightの14次元は自動的に両gripperを除く14次元」とは確認できない。現在のキー順でmax14を適用すると右quaternionの末尾1成分と右gripperが落ちる。これは当時そう訓練された証拠ではなく、**公開重みの訓練時教師schemaとsource versionが未確認**という具体的な不足である。公式gをlocal16教師で採点しない既定は維持する。

## 教師の偏りとμ一軸との関係

同じ既存probeのtrain2048とheldout512から計算した。g教師の二乗量の寄与は次の通り。[全16軸の平均・分散・二次moment](state-target-and-pc1-association.json)。

| 対象 | XYZ寄与 | quaternion寄与 | gripper寄与 | 共分散PC1占有率 | participation ratio |
|---|---:|---:|---:|---:|---:|
| train2048教師 | 7.88% | 8.44% | 83.68% | 45.76% | 2.78 |
| val512教師 | 7.60% | 7.19% | 85.21% | 46.61% | 2.69 |
| state-only外部ridgeのheldout残差 | 10.12% | 11.92% | 77.96% | 43.04% | 3.10 |

従って「XYZが多方向だからg教師も均等」とは言えない。一方、教師のPC1は約46%であり、μの99.9%集中を単に「教師が元々ほぼ一軸」と説明することもできない。ここでの残差は外部ridgeの残差で、実際のgの残差とは区別する。

trainでPCAをfitしheldoutで測ったcombined3000のPC1は、右gripper差分との相関−0.293に対して、state-only残差との相関は−0.078。現在状態だけでPC1分散の35.9%を説明した。trainだけでfitしたPC1一軸の残差補正では、heldout gripper MSE利得0.00327の95%CIは[−0.00050,0.00724]、XYZ利得0.000791のCIは[0.000524,0.001048]。PC1は少量の状態未説明XYZ予測情報を持つが、「gripper専用軸」「現在状態だけの軸」のどちらも確定しない。現在画像からの情報・露出等の交絡は除けていない。

全μを使う外部ridgeのgripper利得は0.01704でCI>0だが、state+current-onlyμとの比較では差を検出できない。微小軸を標準化して使う外部headの利得を、そのままPC1や実visual decoderの利用証拠にはしない。

実gのCPU診断とも照合した。先のnative評価と同じ全512×32のμがbitwise一致し、教師/IDも一致。実g残差の共分散はPC1 37.73%、PR3.71であり、PC1 scoreとgripper残差の相関は左−0.0117、右−0.0315。実gのμ勾配の二乗量をtrain-fit μ-PC1に射影すると、同方向の割合はXYZ19.72%、quaternion5.10%、gripper8.96%だった。gripperからμへの勾配の約91%はμ-PC1と直交する。[実g照合](pc1-actual-g-residual-association.json)。これは「gの勾配自体が一軸だけ」という説明に合わない。ただし決定論的μでの局所勾配であり、訓練時sampled z、IDMパラメータのJacobian、clip/Adam後の更新を直接測った値ではない。

## 今の結果で排除・未排除の範囲

排除できた説明は、z未接続、KLのlocalだけ32次元平均、gの32pad/2時刻による希釈、現在画像を未来教師として扱う単純な時刻取り違え、単一taskしか届かないloaderである。接続・時刻・固定512評価は既存監査、loss reductionは本監査で確認した。

未解決は、公開重みの実教師schema・normalization・データmixture・訓練規模と実効runtime、g2*tanh/16有効次元の影響、限られた更新と条件付き未来の曖昧さ、画像/g/KLとoptimizerの相互作用。betaを1/5へ下げても、4000時点のμはPC1 99.8–99.9%、SNR>1方向数1のまま。従って「KL係数5倍だけで説明し、揃えれば解決」とする説明は既存結果に合わない。

教師と残差のgripper優勢は検証可能な候補だが、今回の結果だけで崩壊原因と断定しない。gを維持したまま各教師成分の実残差・gからの勾配・初期からの更新を分解し、公開条件との差と結びつけて検証する必要がある。
