"""Same-input local checks; extra receiving GEMMs are diagnostic oracles only."""

import torch

from integer_segment_backend import load_native
from paired_precision_reference import encode_fp8, prepare_weight, select_cohorts
from precision_moments import predicted_energy


class MomentDiagnostics:
    def __init__(self, rows: int, calls: int = 1):
        self.rows = rows
        self.calls = calls
        self.records = {}

    def observe(self, name, module, values):
        records = self.records.setdefault(name, [])
        if len(records) >= self.calls:
            return
        if values.dtype != torch.bfloat16 or module.weight.dtype != torch.bfloat16:
            records.append({"skipped": "non-BF16 operands", "input_shape": list(values.shape)})
            return
        convolution = isinstance(module, torch.nn.Conv2d)
        matrix = (values.permute(0, 2, 3, 1).reshape(-1, module.in_channels) if convolution
                  else values.reshape(-1, module.in_features))
        weight = module.weight.reshape(module.weight.shape[0], -1)
        indices = torch.linspace(0, len(matrix) - 1, min(len(matrix), self.rows),
                                 device=matrix.device).long().unique()
        sample = matrix[indices].contiguous()
        prepared = prepare_weight(weight.detach(), with_moments=True)
        load_native()
        result = {"input_shape": list(values.shape), "matrix_shape": list(matrix.shape),
                  "weight_shape": list(weight.shape), "sampled_row_indices": indices.cpu().tolist(),
                  "slices": []}
        for pair, begin in enumerate(range(0, sample.shape[1], 256)):
            a = sample[:, begin:begin + 256]
            if a.shape[1] <= 128:
                continue
            z, exponent = encode_fp8(a)
            w, q = weight[:, begin:begin + 256], prepared.reconstructed[:, begin:begin + 256]
            high = torch.ops.tensor_inference_engine.gemm(a, w.T, True)
            low = torch.ops.tensor_inference_engine.gemm(z, q.T, True)
            error = (high - low).square()
            reference = high.square()
            actual_error = torch.stack([error[:, start:start + 128].mean(-1)
                                        for start in range(0, weight.shape[0], 128)], -1)
            actual_reference = torch.stack([reference[:, start:start + 128].mean(-1)
                                            for start in range(0, weight.shape[0], 128)], -1)
            moment_error, moment_reference = predicted_energy(a, z, prepared.moments[pair])
            diagonal_error, diagonal_reference = predicted_energy(a, z, prepared.moments[pair], centered=False)
            _, bound = select_cohorts(a, z, prepared.l1[pair], prepared.error_l1[pair],
                                      tolerance=0, cohort=1)
            result["slices"].append({
                "k_begin": begin, "width": a.shape[1], "activation_exponents": exponent.cpu().tolist(),
                "max_l1_normalized_bound": bound.cpu().tolist(),
                "actual_error_energy": actual_error.cpu().tolist(),
                "actual_reference_energy": actual_reference.cpu().tolist(),
                "moment_error_energy": moment_error.cpu().tolist(),
                "moment_reference_energy": moment_reference.cpu().tolist(),
                "diagonal_error_energy": diagonal_error.cpu().tolist(),
                "diagonal_reference_energy": diagonal_reference.cpu().tolist(),
            })
        records.append(result)
