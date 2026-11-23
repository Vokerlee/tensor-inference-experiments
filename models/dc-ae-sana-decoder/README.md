# DC-AE decoder in SANA-Sprint0.6B

Neural image-decoder component of the pinned SANA-Sprint pipeline. Its spatial
convolutions and residual upsampling pathways are distinct from latent denoising.
The autoencoder's image-encoding path is excluded from this text-to-image scope.

- Checkpoint: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`, revision
  `aa76e7f4f4928f378716b6716a2130fba3caf5b1`, `vae/config.json`.
- Batch1 input:32×32 spatial latent with32 channels; output:1024×1024 RGB image.
- Decoder parameters:159,223,171; hypothetical BF16 payload318,446,342 bytes.
  Parameter payload is not compulsory per-request memory traffic.
- Matrix/convolution MACs:4,184,305,827,840 per decode.
- Spatial-convolution MACs:3,682,134,392,832, about88% of decoder MACs.
- Evidence: BF16 meta shape execution with a separate Torch dispatch FLOP check,
  not numerical model quality or hardware energy.

[Complete immutable component/request profile](../sana-sprint-06b/pipeline-geometry-2026-11-23.json)
retains every recorded convolution/projection shape and interface comparison.
[Request conditions and component balance](../sana-sprint-06b/README.md) explain
prompt encoding and denoising steps. Preserved preliminary and clean profiles
are in the [full statistical archive](../../reports/statistics/2026-11-23-adaptive/README.md).

[Exact-input decoder feedback results](../../reports/2026-11-23-decoder-feedback.md)
cover eight1024px outputs, BF16/INT8 weights, INT4/INT8 activations and a causal
row-statistic controller ablation. Full JSON statistics are archived; pixel
fidelity and MAC coverage do not establish generation quality or energy.

[Resize folding and source-placement results](../../reports/2026-11-23-resize-folding.md)
validate the known nearest-neighbor transformation, coefficient precision and
conversion before duplication, with complete per-branch statistics. Dense MAC
reductions and pixel fidelity remain distinct from hardware energy/task quality.
