# Systolic action accounting

`scripts/systolic_actions.py` separates arithmetic opportunities from scheduled
array movement. It counts one declared output-stationary schedule with fixed
M/N/K tile dimensions. It is not a chip simulator, physical power estimate,
or a model of every possible systolic dataflow.

```sh
.venv/bin/python scripts/systolic_actions.py --shape 128 768 3072 \
    --retained-k 1536 --output results/systolic-actions.json
```

The array boundary receives A rows and B columns. Operands are forwarded
through clocked registers and wires; each PE multiplies and accumulates.
Outputs require accumulator storage and eventual unloading. SRAM banks,
format conversion, sparse metadata and clocks add costs around this path.
The relative energy of these components depends on arithmetic, implementation,
dataflow and activity. There is no universal component percentage.

[Peltekis et al., Sections II–IV](https://arxiv.org/html/2304.12691v3)
distinguish operand streaming, computation and sum movement for their BF16
array. Their measured/synthesized configuration is not a component-energy
characterization of another array. In particular, small multipliers do not
make wide accumulation or registers proportionally cheaper.

| Representation / execution | Arithmetic opportunity | Movement and schedule contract |
| --- | --- | --- |
| Lower activation width only | Depends on supported mixed-format multiply | B, FP32 sums and tile count may stay unchanged; narrow A must stay encoded through delivery |
| Zero gating | Can suppress redundant multiply/add switching | Dense time slots and B delivery remain unless separately gated |
| Native per-row2:4 skipping | Can reduce issued products with a sparse datapath | B channels may all still be needed; faster issue can increase required B bandwidth |
| Common-K compaction | Removes whole reduction positions for all tile rows | Can reduce both operand delivery and K calls; gathering and a shared mask cost work |
| Two-stream residual | Improves numerical representation with small codes | Can require two passes, extra accumulation/scales/metadata; shared weights alone do not remove this |

The JSON exposes useful products, padded issued products, boundary slots,
internal word hops and two partial-sum-memory alternatives. A word hop is an
action, not a bit toggle. Padding is counted as scheduled slots, not assumed
physical switching. In this schedule, boundary operand slots are reissued per
output tile; SRAM reads can be lower with additional reuse. Raw A values must
still be inspected to choose a mask, even if the compacted array input is small.

For M128,N768,K3072, tile128 in each dimension, common half-K changes144
matrix-tile calls to72. When partial sums stay in PEs across K, the final
FP32 output write is unchanged. If each K tile instead updates result SRAM,
traffic changes17.625MiB to8.625MiB. Both numbers concern local result SRAM;
there is no assumed external-memory spill. For K96, retaining48 positions
still requires one K128 tile and cannot obtain this call saving.

The two feed/fill/drain counters are conditional schedules, excluding stalls,
unloading and overlapping different output tiles. Continuous-K feeding avoids
refilling between K segments only when arithmetic/scales and PE state support
it. Groups of32 independently scaled values inside a K128 tile need native
block scaling or separately scaled partial products; neither is free.

An eventual energy calculation multiplies each action category by a
characterized per-action cost and adds encoding/control, clock and leakage
costs. Include baseline actions avoided as well as new actions incurred.
[Accelergy](https://accelergy.mit.edu/paper.pdf) provides that accounting
framework; [Sparseloop](https://arxiv.org/abs/2205.05826v3) distinguishes sparse
representation, gating and skipping. The ledger deliberately supplies no
invented coefficients or target-energy percentages.
