# Source-group spatial feedback experiments

`spatial_feedback_reference.py` quantizes source activations before convolution
lowering. It applies a one-dimensional residual feedback along each spatial
row/channel, resets state at row boundaries and uses a current power-of-two
scale for each source pixel/channel group of32. An optional mode emits exact
zeros and resets carry where the original activation is zero.

For each channel, the kernel-derived coefficient divides horizontal lag-one
kernel autocorrelation by kernel energy, then clips to[-0.5,0.5]. A coarse
variant rounds to{-0.5,0,0.5}. A white, independent, equal-variance residual
approximation motivates this coefficient. Actual residuals can be biased,
correlated, nonstationary and affected by clipping; the approximation supplies
neither an error guarantee nor a task-quality prediction.

```sh
.venv/bin/python scripts/spatial_feedback_pilot.py --model resnet18 --bits 4 \
    --samples-per-class 1 --output results/spatial-feedback-r18.json
```

The pilot first compares every policy on identical original-trajectory inputs
at each eligible convolution, then executes the complete modified model for
each policy. It distinguishes input reconstruction error, receiving-output
error and propagated final-logit/prediction changes. Fixed policies include
no feedback, unit feedback and half feedback. Kernel-derived policies include
continuous, coarse and coarse-with-zero-preservation coefficients. No labels
or calibration activations choose these coefficients; they are deterministic
statistics of the pretrained receiving weights. Scales and residuals use live
activation values.

Only aligned dense convolutions are transformed. Their weights stay BF16 and
products use Torch convolutions on reconstructed activations. This is not a
packed low-bit matrix kernel, integer-weight experiment or native array model.
The reference returns a full residual tensor for diagnostics; a physical stream
needs explicitly scheduled carry state, not this complete tensor. State reads,
feedback arithmetic, coefficient delivery, quantization and changed zero
patterns must be priced in a hardware evaluation. CUDA latency is irrelevant
to estimating their target-hardware cost.

Spatial feedback with hardware is established in
[DitherNN](https://doi.org/10.1587/transinf.2019PAP0009). Filter-dependent noise
shaping also has prior art, including
[Minami et al.](https://doi.org/10.9746/sicetr.56.425) for weight quantization and
[PREFILT](https://openaccess.thecvf.com/content/ICCV2025W/BiVision/html/Makhov_PREFILT_Prefiltering_for_Fully_Quantized_Image_Restoration_Neural_Networks_ICCVW_2025_paper.html)
for image-restoration preprocessing and training. This experiment does not
claim a new feedback principle or establish novelty of the specific variant.
No training is performed in this pilot. Smaller local errors must still be
checked against complete-model quality.

Export complete paired runs with `scripts/summarize_spatial_feedback.py --input
RUN.json ... --output-prefix reports/PREFIX`. The exporter checks clean executed
source hashes and the engine gitlink against recorded Git objects. It retains
paired predictions, local counters, checkpoint identities and image manifests;
incomplete runs are rejected. Raw execution reports stay in ignored `results/`.
