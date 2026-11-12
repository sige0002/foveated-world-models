# F029: Appendix C.6 の独立確認

対象: De Bortoli et al., *Diffusion Schrödinger Bridge Matching*, [arXiv:2303.16852v3](https://arxiv.org/html/2303.16852v3), Proposition 5、Lemma 6、Proposition 7、Theorem 8 と Appendix C.3–C.6。以下は提示された証明の根拠確認であり、Theorem 8 自体が偽であるとの結論ではない。

1. **Markov class の弱閉性。** C.6 は Markov class と reciprocal class が weakly closed と述べるが、Proposition 5 の参照過程・端点分布の条件、Lemma 6 の各反復における正則性および有限 KL から、その列の極限に対する閉性を示す議論や文献が提示されていない。一般の連続経路上の Markov law は弱閉ではない。例えば符号 S=±1 を等確率で選び、時刻 0、1/2、1 の値をそれぞれ S、S/n、S として線形補間する。各 n は全時刻で S を状態から読めるので Markov だが、弱極限は中間時刻の状態が常に 0 で、その前の S が後の S を決めるため Markov でない。この例は論文の正の拡散係数・有限 KL 等の仮定を満たす反例とは主張しない。ここで必要なのは、当該 IMF 列と固定参照に特化した安定性・弱閉性の定理、または極限まで伝播する一様な drift/transition kernel 等の追加条件の確認である。reciprocal class の閉性も bridge kernel の端点連続性等に依存し、同じくこの証明中では未確認。

2. **weak convergence から KL convergence。** C.6 が唯一の弱極限 P★=PSB を得たとしても、その直後に置かれる KL(M^{n_jk}|P★)→0 は弱収束と KL 下半連続性からは導けない。下半連続性は KL(P★|P★)≤liminf KL(M^{n_jk}|P★) という下限だけを与える。例えば [0,1] の一様分布 P に対し密度 1+a sin(2πnx), 0<a<1 を持つ Pn は P に弱収束し、各 KL(Pn|P) は有限かつ n に依存しない正値。有限で一様有界な密度比さえ一般には不十分である。この例も IMF 反復の反例ではない。KL 値の収束を別途証明できる上界・適切な density convergence と entropy uniform integrability、あるいは IMF 特有の entropy contraction/compactness が追加で必要。Proposition 7 の単調性と隣接 KL→0 だけから零への収束は直接出ない。

3. **他の依存条件。** Proposition 5 の完全な C.3 版は、極限が Markov かつ reciprocal、固定端点であるだけでなく KL(P★|Q)<∞ も要求する。C.6 の上限は KL(Pn|PSB) なので、極限での対 Q の有限性を適用前に別途確認する必要がある。端点の弱保存、隣接 KL→0 と KL の同時下半連続性から二つの弱極限が一致する段階は、本文の通り適切な同時部分列を取れば整合する。固定点の一意性と KL 収束の証明を混同しない。

**評価:** Proposition 5/Lemma 6 の記載条件のみで上記2つの遷移が保証されることは、提示された C.6 および関連箇所からは確認できなかった。IMF の特殊構造で成立する可能性は残り、定理の反例や後続訂正の有無は本チェックでは未検証。実用上は反復が有限モデルの paired MSE を単調に改善する根拠として本定理を使わない。
