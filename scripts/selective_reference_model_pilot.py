#!/usr/bin/env python3
"""Complete model/control fidelity and software timing of selective observations."""

import argparse
import json
import time
from pathlib import Path

import torch

from activation_pilot import ENGINE, ROOT, digest, load_data, revision, write_json
from modern_models import vision
from paired_precision_pilot import receiving_layers
from selective_reference_models import compare_controls, instrumented, prepare_layers, write_controls
from selective_reference_operator_pilot import METHODS, time_call


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("convnextv2-tiny", "dinov3-small"), required=True)
    parser.add_argument("--count", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--policy", action="append", choices=("bf16", "dynamic8", "live4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=False)
    model, data, checkpoint = vision(args.model)
    layers = receiving_layers(model)
    loader, classes, manifest = load_data(data, 1, "quality", 1, 250)
    batches = [(images.cuda().bfloat16(), classes[int(labels[0])]) for images, labels in loader]
    start = time.perf_counter()
    weights = prepare_layers(layers)
    torch.cuda.synchronize()
    preparation_ms = (time.perf_counter() - start) * 1000
    policies = args.policy or ["bf16", "dynamic8", "live4"]
    report = {"schema": 1, "status": "running", "model": args.model,
        "experiments": revision(ROOT), "engine": revision(ENGINE), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "checkpoint": checkpoint, "dataset": manifest,
        "preparation_ms": preparation_ms, "selected_layers": list(layers), "policies": policies,
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "selective_reference_model_pilot.py", "selective_reference_models.py",
            "selective_bf16_reference.py", "modern_models.py", "source_feedback_backend.py")},
        "native_sha256": digest(ENGINE / "build/libtensor_inference_engine_torch.so"),
        "runs": [], "invalid_cases": [], "limitations": [
            "Selected linear and aligned1x1 operators use the declared sequential RN primitive128 contract",
            "Other operators remain Torch; this is not complete target-device arithmetic emulation",
            "Exactness is at post-bias BF16 observations, not FP32 primitive output values",
            "Source quantization actually feeds consumers; live4 state resets per consumer/image",
            "Live source controller uses preceding32-row terminal moments; a height1 source has no subsequent block",
            "Per-output validation and complete controller state are recorded separately from timed passes",
            "Warm batch1 eager timing includes source control, filtering, compaction and repair, excludes one-time preparation",
            "Timing is software emulator latency, not target-device energy or latency",
            "Small deterministic input set tests fidelity; it does not establish population accuracy or optimal quantization",
        ]}
    destination = args.output / "report.json"
    write_json(destination, report)
    for image_id, (image, target) in enumerate(batches[:args.count]):
        original = model(image).float().cpu()
        for policy in policies:
            reference_outputs, reference_controls, reference = None, None, None
            reference_tape = None
            for method in METHODS:
                with instrumented(layers, weights, method, policy, record=True,
                                  reference_outputs=reference_outputs) as audit:
                    output = model(image).float().cpu()
                invalid = sum(int(value.item()) for value in audit["invalid"])
                finite = bool(torch.isfinite(output).all())
                if reference is None:
                    reference = output
                    reference_outputs = audit["outputs"]
                    reference_controls = audit["controllers"]
                output_mismatches = int((output != reference).sum())
                controls = compare_controls(audit["controllers"], reference_controls)
                control_mismatches = sum(sum(row["mismatches"].values()) for row in controls)
                tape = args.output / f"{policy}-{image_id:02d}-{method}-controls.npz"
                write_controls(tape, audit["controllers"])
                tape_sha = digest(tape)
                if reference_tape is not None and tape_sha == reference_tape[1]:
                    tape.unlink()
                    tape = reference_tape[0]
                elif reference_tape is None:
                    reference_tape = (tape, tape_sha)
                # Statistics and CPU snapshots are intentionally disabled in timing.
                with instrumented(layers, weights, method, policy):
                    model(image)
                with instrumented(layers, weights, method, policy) as timed_audit:
                    timed, timing = time_call(lambda: model(image))
                timed_invalid = sum(int(value.item()) for value in timed_audit["invalid"])
                timed_equal = bool(torch.equal(timed.float().cpu(), reference))
                row = {"image_id": image_id, "target": target, "policy": policy, "method": method,
                    "output": output.tolist(), "original_torch_output": original.tolist(),
                    "finite": finite, "final_output_mismatches": output_mismatches,
                    "control_mismatches": control_mismatches, "controller_comparison": controls,
                    "invalid_events": invalid, "timed_invalid_events": timed_invalid,
                    "timed_output_identical": timed_equal, "calls": audit["calls"],
                    "trace": str(tape), "trace_sha256": tape_sha, "timing": timing}
                if not finite or output_mismatches or control_mismatches or invalid or timed_invalid or not timed_equal:
                    report["invalid_cases"].append({"image_id": image_id, "policy": policy, "method": method})
                report["runs"].append(row)
                write_json(destination, report)
                print(args.model, image_id, policy, method, output_mismatches, control_mismatches,
                      round(timing["wall_ms"], 3), flush=True)
    report["status"] = "invalid" if report["invalid_cases"] else "complete"
    write_json(destination, report)


if __name__ == "__main__":
    main()
