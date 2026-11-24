# Joint precision/lowering shape control

For spatial dimensions divisible by4, the same4x4 output region uses
36ceil(C/128) input vectors under BF16 F(4,3), versus64ceil(C/256) under
an eight-bit F(2,3) interface, before output-column tiling. The shape control
chooses BF16 F(4,3) when9ceil(C/128)<=16ceil(C/256), otherwise eight-bit F(2,3).
A precondition check rejects spatial tails rather than assuming this formula.

`mixed-bf16-f4-fp8-f2` uses the existing BF16/FP8 interface geometries.
`mixed-bf16-f4-int8-f2` substitutes the proposed INT8 mode in the wide branch.
Both execute actual downstream model consumers; the rule is shape-only,
calibration-free and a strong baseline, not a claimed new adaptive algorithm.

Fewer packets do not establish lower energy. BF16 arithmetic, zero-lane gating,
weight payload, transform cost, source retention and output cadence differ.
Exact per-layer workload counts and floating coefficient fingerprints accompany
the numerical outputs. The rule's quality must be measured, not inferred from
its local numerical precision or vector count.
