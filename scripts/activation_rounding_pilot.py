#!/usr/bin/env python3
"""Audit bounded-state activation rounding on hash-verified modern operands."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from activation_rounding import group_grid, prepare_geometry, round_groups


def from_words(words: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(words.copy()).view(torch.bfloat16).cuda().contiguous()


def isolated_errors(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    groups = error.reshape(error.shape[0], -1, 32).permute(1, 0, 2).double()
    blocks = weight.reshape(-1, 32, weight.shape[1]).double()
    return (groups @ blocks).square().sum(-1).transpose(0, 1)


def audit_values(values: torch.Tensor, source: torch.Tensor, weight: torch.Tensor,
                 nearest_group_errors: tuple[torch.Tensor, torch.Tensor] | None, output_path: Path, result=None) -> dict:
    error = values.double() - source.double()
    receiving = error @ weight.double()
    group_errors = isolated_errors(error, weight)
    source_groups = error.reshape(error.shape[0], -1, 32).square().sum(-1)
    tile_errors = torch.stack([receiving[:, start:start + 128].square().sum(-1)
                               for start in range(0, weight.shape[1], 128)], 1)
    arrays = {"source_group_sse": source_groups.cpu().numpy(),
              "isolated_group_receiving_sse": group_errors.cpu().numpy(),
              "output_tile_receiving_sse": tile_errors.cpu().numpy()}
    row = {"source_sse": float(source_groups.sum()), "receiving_sse": float(tile_errors.sum()),
           "isolated_group_sse": float(group_errors.sum()),
           "cross_group_sse_term": float(tile_errors.sum() - group_errors.sum()),
           "exact_source_zeros_changed": int(((source == 0) & (values != 0)).sum()),
           "reconstructed_zeros": int((values == 0).sum())}
    if result is not None:
        arrays.update(selected=result.candidate_selected.cpu().numpy(),
                      exponents=result.exponents.cpu().numpy(),
                      proxy_candidate=result.proxy_candidate.cpu().numpy(),
                      proxy_nearest=result.proxy_nearest.cpu().numpy(),
                      spectral_margin=result.spectral_margin.cpu().numpy(),
                      code_histogram=torch.bincount((result.codes.int() + 128).flatten(), minlength=256).cpu().numpy())
        row.update(selected_groups=int(result.candidate_selected.sum()),
                   groups=result.candidate_selected.numel(),
                   changed_codes=int((values != nearest_group_errors[1]).sum()))
        baseline_errors = nearest_group_errors[0]
        row["worse_isolated_groups"] = int((group_errors > baseline_errors * (1 + 1e-6) + 1e-12).sum())
    np.savez_compressed(output_path, **arrays)
    row.update(arrays=str(output_path), arrays_sha256=digest(output_path))
    return row


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--bits", type=int, nargs="+", choices=(4, 8), default=[4, 8])
    parser.add_argument("--rank", type=int, nargs="+", choices=(1, 2, 4, 32), default=[1, 2, 4, 32])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "status": "running", "experiments": revision(ROOT), "engine": revision(ENGINE),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "activation_rounding_pilot.py", "activation_rounding.py")},
        "inputs": [{"path": str(path), "sha256": digest(path)} for path in args.input],
        "bits": args.bits, "ranks": args.rank, "operators": [], "invalid_cases": [],
        "limitations": [
            "Existing captured operators and deterministic rows, not new independent images or task evaluation",
            "All output columns share one activation copy; no output-tile-specific activation reencoding",
            "Weight-only group geometry is prepared without activation calibration",
            "Runtime rule uses r running scalar states; no receiving GEMM selects codes",
            "Rank32 is an expensive diagnostic, not a proposed lightweight controller",
            "Spectral screen is real-algebra diagnostics evaluated in FP64, not a hardware numerical certificate",
            "Cross-group error terms remain in actual product errors; isolated group sums are not the full error",
            "FP64 receiving products audit quantization error, not a specific finite accelerator accumulation graph",
            "No software timing, energy, native packed execution or hardware throughput claim"]}
    destination = args.output / "report.json"
    for parent_path in args.input:
        parent = json.loads(parent_path.read_text())
        if parent["status"] != "complete" or parent["invalid_cases"]:
            raise ValueError("complete valid captured operator report required")
        for operator in parent["operators"]:
            operands = Path(operator["operands"])
            if digest(operands) != operator["operands_sha256"]:
                raise ValueError("captured operand changed")
            with np.load(operands, allow_pickle=False) as arrays:
                source = from_words(arrays["source_bf16"])
                weight = from_words(arrays["weight_bf16"])
            identifier = len(report["operators"])
            case = {"model": parent["model"], "layer": operator["layer"], "shape_mnk": operator["shape_mnk"],
                    "operands_sha256": operator["operands_sha256"], "records": [], "geometries": []}
            geometries = {rank: prepare_geometry(weight, rank).to("cuda") for rank in args.rank}
            for rank, geometry in geometries.items():
                path = args.output / f"operator-{identifier:02d}-rank{rank}-geometry.npz"
                np.savez_compressed(path, factor=geometry.factor.cpu().numpy(), diagonal=geometry.diagonal.cpu().numpy(),
                                    residual_norm=geometry.residual_norm.cpu().numpy(),
                                    normalizer=geometry.normalizer.cpu().numpy(), eigenvalues=geometry.eigenvalues.cpu().numpy())
                case["geometries"].append({"rank": rank, "arrays": str(path), "arrays_sha256": digest(path)})
            for bits in args.bits:
                values, exponent, scale, limit = group_grid(source, 32, bits)
                nearest = (values.round().clamp(-limit, limit) * scale[:, :, None]).reshape_as(source).bfloat16()
                baseline_errors = isolated_errors(nearest.double() - source.double(), weight)
                baseline = audit_values(nearest, source, weight, None,
                                        args.output / f"operator-{identifier:02d}-a{bits}-nearest.npz")
                case["records"].append({"bits": bits, "method": "nearest", **baseline})
                for rank, geometry in geometries.items():
                    selections = ("proxy",) if rank == 32 else ("candidate", "proxy", "spectral")
                    for selection in selections:
                        result = round_groups(source, geometry, bits, selection)
                        path = args.output / f"operator-{identifier:02d}-a{bits}-r{rank}-{selection}.npz"
                        stats = audit_values(result.reconstructed, source, weight, (baseline_errors, nearest), path, result)
                        stats["receiving_sse_ratio"] = stats["receiving_sse"] / baseline["receiving_sse"] if baseline["receiving_sse"] else None
                        if selection == "spectral" and stats["worse_isolated_groups"]:
                            report["invalid_cases"].append({"operator": identifier, "bits": bits, "rank": rank,
                                                            "reason": "spectral group screen failed finite audit"})
                        case["records"].append({"bits": bits, "rank": rank, "method": selection, **stats})
                print(parent["model"], operator["layer"], bits,
                      [(row.get("rank"), row["method"], round(row.get("receiving_sse_ratio") or 1, 4))
                       for row in case["records"] if row["bits"] == bits], flush=True)
            report["operators"].append(case)
            write_json(destination, report)
    report["status"] = "invalid" if report["invalid_cases"] else "complete"
    write_json(destination, report)


if __name__ == "__main__":
    main()
