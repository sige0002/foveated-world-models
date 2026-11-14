# 計算精度の復元順：CPUでの独立再現

実際の学習用`.venv`、PyTorch `2.11.0+cu130`で確認した。[再現コード](../reproduce_torch_precision_restore_cpu.py)、[環境・入力SHA・全設定値・3回分の結果](robotwin-precision-restoration-20261114.json)。CUDA初期化、モデル学習、既存ファイルの変更は行っていない。

検証は既存`evaluate`の精度設定・復元手順をそのまま取り出し、GPUモデル推論の代わりにCPU行列積を挟むもの。旧順序は1回目の評価を完了した後、次の`get_float32_matmul_precision()`がAPI混在の`RuntimeError`になり、2回目・3回目は行列積の前で失敗。修正順序では3回とも成功し、毎回、検査した設定値とCPU RNGが元に戻った。

修正は復元の順序。

```python
torch.backends.cuda.matmul.allow_tf32 = saved_tf32
torch.backends.cudnn.allow_tf32 = saved_cudnn_tf32
torch.set_float32_matmul_precision(saved_precision)  # 最後に復元
assert torch.get_float32_matmul_precision() == saved_precision
```

旧順序でも公開backend設定は初期medium時と同じだった。CUDA matmulは`tf32`、MKLDNN matmulは`bf16`、cuDNN convは`tf32`。異なったのは全体を問い合わせるgetterがエラーになった点である。この再現で確定した障害は、精度設定の管理状態が不整合になり、次の評価が停止すること。

したがって、このgetterの障害だけで「500更新以降のGPU演算精度が変わった」「重みが破損した」と断定しない。一方、CPU検証から501〜1000更新の実GPUカーネル、演算、勾配、optimizer軌跡の一致も保証できない。復旧の評価には、この限定とcheckpoint保存時点・実ストリームの再現性を併記する。

```sh
.venv/bin/python research/reproduce_torch_precision_restore_cpu.py
```
