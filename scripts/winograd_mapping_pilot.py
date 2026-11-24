#!/usr/bin/env python3
"""Derive a reproducible interface-work ledger from a captured model geometry."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from activation_pilot import ENGINE, ROOT, digest, revision, write_json
from winograd_mapping import BF16, FP8, from_profile, retention, work


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path,
                        default=ROOT / "models/sana-sprint-06b/pipeline-geometry-2026-11-23.json")
    parser.add_argument("--component", default="image_decoder")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.profile.read_text())
    component = source["components"][args.component]
    layers = []
    for layer in component["geometry"]["layers"]:
        shape = from_profile(layer)
        if shape is None:
            continue
        records = [work(shape, interface, tile)
                   for interface in (BF16, FP8) for tile in (None, 2, 4)]
        assert records[0]["logical_macs"] == layer["macs"]
        layers.append({"name": layer["name"], "shape": asdict(shape), "work": records})
    totals = []
    for interface in (BF16, FP8):
        for tile in (None, 2, 4):
            rows = [work(from_profile(layer), interface, tile)
                    for layer in component["geometry"]["layers"] if from_profile(layer)]
            totals.append({"interface": interface.name, "method": rows[0]["method"],
                           **{key: sum(row[key] for row in rows) for key in (
                               "logical_macs", "issued_vectors", "padded_product_slots",
                               "lhs_interface_capacity_bytes",
                               "fp32_output_capacity_bytes_if_every_segment_emits",
                               "weight_payload_bytes")}})
    report = {
        "schema": 1, "status": "complete", "experiments": revision(ROOT),
        "engine": revision(ENGINE), "profile": str(args.profile.resolve()),
        "profile_sha256": digest(args.profile), "component": args.component,
        "source_sha256": {name: digest(ROOT / "scripts" / name) for name in (
            "winograd_mapping.py", "winograd_mapping_pilot.py", "activation_pilot.py")},
        "original_component_macs": component["geometry"]["matrix_and_conv_macs"],
        "eligible_direct_macs": totals[0]["logical_macs"],
        "layers": layers, "totals": totals,
        "retention_variants": [
            retention(interface, tile, rows, planes, double_buffer_inputs=double)
            for interface in (BF16, FP8) for tile in (2, 4)
            for rows in (32, 64, 128, 256)
            for planes in (1, 4, (tile + 2) ** 2) for double in (False, True)],
        "limitations": [
            "Eligible dense 3x3 stride1 dilation1 convolutions only; all other component work remains unchanged",
            "Vector counts are interface work, not cycles, joules or measured SRAM transactions",
            "FP8 geometry does not establish a numerically valid transformed quantization policy",
            "Fixed K128/BF16 or K256/FP8 and N128; streamed M is not a fixed array dimension",
            "2MiB argument memory is shared by lhs and rhs; 2MiB result memory is separate",
            "Payload beyond 64MiB motivates tiling, not an assertion that a layer cannot execute",
            "Weight transform cost, metadata, input/output transforms and scheduling require separate accounting",
            "Streaming inverse transform storage assumes linear per-K fusion, whose arithmetic and rounding must be tested",
            "FP32 output capacity is reported conditionally: inter-K retention and actual output cadence are unspecified",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"{len(layers)} eligible layers; {report['eligible_direct_macs']} original MACs")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
