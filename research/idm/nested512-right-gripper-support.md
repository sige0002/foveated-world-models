# Fixed nested512 selection: right-gripper training support

This is metadata/NPZ-only counting of the already fixed512 selection, **not model evaluation**. Neither selection nor thresholds changed. Static is |measured endpoint delta|≤.01; the small/large boundary remains the previous train128 median nonstatic magnitude, .7268722. Values are source gripper units, not commands or validated physical aperture.

| Fixed subset | Static | Small change | Large change | Large positive | Large negative |
|---|---:|---:|---:|---:|---:|
| Retained128 |110|9|9|3|6|
| Added384 |328|18|38|15|23|
| Full512 |438|27|47|18|29|

The previously problematic large-positive subgroup grows **3→18 examples (6×)**; its fraction grows2.34%→3.52%. It remains rare, and increased support alone does not establish improved prediction. Overall active examples grow18→74;86% of the expanded set remains operationally static.

All retained128 targets exactly match the earlier measured-label artifact. All512 episode/index/dt/endpoint mappings agree with the selected manifest; endpoints are originally observed with zero fill age. Selection/manifest/bin-protocol and per-NPZ hashes are saved in `nested512-right-gripper-support.json`. No image decoding, fitting, GPU use or selection changes occurred; CPU2 only. Reproduction: `summarize_nested512_gripper.py` (refuses existing output).
