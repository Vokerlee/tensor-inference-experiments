# Complete SANA-Sprint request profile

`profile_sana_pipeline.py` inventories the pinned prompt encoder, one latent
denoiser call and the image decoder, then applies explicit request call counts.
The default is batch1,1024×1024 pixels,300 prompt tokens and two denoising calls.
The checkpoint uses guidance embeddings, so no classifier-free doubled call is
assumed. Text encoding and decoding each occur once for a cold prompt request.

The profiler uses metadata/fake tensors without pretrained numerical weights.
Decoder and denoiser shape execution use BF16 meta tensors; the prompt encoder
uses FP32 CPU-fake eager attention with prepared mask shapes to avoid allocating
its vocabulary/model weights. Dense MAC counts are dtype-independent; this
method does not verify numerical equivalence, mixed-precision kernels or masked
attention work elision. No language-model head is executed. Vocabulary lookup,
vector/nonlinear operations, scheduler/tokenization and memory actions are
outside the matrix/convolution denominator.

Every component is independently checked with Torch's dispatch FLOP counter:
its counted matrix/convolution FLOPs must equal twice the recorded MACs.
Executed convolution geometry and dense projection shapes are retained.
Component parameter counts do not imply compulsory parameter reads per request.
Only the decoder path of the autoencoder is counted for text-to-image output.

```sh
.venv/bin/python scripts/profile_sana_pipeline.py \
    --output results/sana-full-pipeline.json
```

Denoiser-only proportions cannot be applied to a whole request. Change the
explicit step/prompt-reuse assumptions before comparing request scope. Component
MAC shares are not energy shares, software latency or target-hardware timing.
