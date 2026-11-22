"""Numerical representation policies shared by model-dependent pilots."""

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "deps/tensor-inference-engine"
SPEC = importlib.util.spec_from_file_location("selection", ENGINE / "src/selection_reference.py")
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def packed_gram(gram):
    upper = torch.triu_indices(4, 4, device=gram.device)
    return gram[:, upper[0], upper[1]].contiguous()


def sparse_selection(x, q, gram, packed, policy, backend):
    if backend in {"cuda", "cuda-tile"} and policy in {"diagonal", "gram"}:
        shape = x.shape
        output, pairs = torch.ops.tensor_inference_engine.select_2of4(
            x.float().reshape(-1, shape[-1]).contiguous(),
            q.float().reshape(-1, shape[-1]).contiguous(), packed, policy == "diagonal")
        return output.reshape(shape), pairs.reshape(*shape[:-1], -1)
    return selection.select_2of4(x, q, gram, policy)


def common_representation(x, gram, policy, tolerance):
    """A common BF16 mask per128 rows; scalar quartet covariance accumulation."""
    if policy not in {"diagonal", "gram"}:
        raise ValueError("common masks require diagonal/gram scores")
    shape = x.shape
    values = x.float().reshape(-1, gram.shape[0], 4)
    rows, groups, _ = values.shape
    padded = torch.nn.functional.pad(values, (0, 0, 0, 0, 0, (-rows) % 128))
    windows = padded.reshape(-1, 128, groups, 4)
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    squares = windows.square().sum(1)
    losses = []
    for kept in selection.PAIRS:
        i, j = [index for index in range(4) if index not in kept]
        loss = squares[..., i] * diagonal[:, i] + squares[..., j] * diagonal[:, j]
        if policy == "gram":
            loss = loss + 2 * (windows[..., i] * windows[..., j]).sum(1) * gram[:, i, j]
        losses.append(loss)
    losses = torch.stack(losses, -1).clamp_min(0)
    indices = losses.argmin(-1)
    masks = torch.zeros((6, 4), device=x.device, dtype=torch.bool)
    for pair, kept in enumerate(selection.PAIRS):
        masks[pair, list(kept)] = True
    chosen = masks[indices]
    enabled = torch.ones(windows.shape[0], device=x.device, dtype=torch.bool)
    if tolerance is not None:
        if policy == "gram":
            denominator = selection.group_loss(padded.reshape(-1, shape[-1]), gram)
            denominator = denominator.reshape(-1, 128, groups).sum((1, 2))
        else:
            denominator = (squares * diagonal).sum((1, 2))
        enabled = losses.amin(-1).sum(-1) <= tolerance * denominator
        # A fixed-size128 reduction cannot save a call if both K sizes round to one tile.
        if (shape[-1] // 2 + 127) // 128 >= (shape[-1] + 127) // 128:
            enabled.zero_()
    chosen = torch.where(enabled[:, None, None], chosen, True)
    output = (windows * chosen[:, None]).reshape(-1, groups, 4)[:rows].reshape(shape)
    counts = torch.zeros(6, device=x.device, dtype=torch.int64)
    elements = torch.zeros_like(counts)
    counts[5] = enabled.sum() * ((shape[-1] + 127) // 128)
    counts[0] = (~enabled).sum() * ((shape[-1] + 127) // 128)
    row_enabled = enabled.repeat_interleave(128)[:rows]
    elements[5] = row_enabled.sum() * shape[-1]
    elements[0] = (~row_enabled).sum() * shape[-1]
    return output, counts, elements


def sum_mode_tiles(values: torch.Tensor, tile_rows: int = 128) -> torch.Tensor:
    """Aggregate [M, K/4] scores into physical input blocks of 128 x 128."""
    rows, groups = values.shape
    padded = torch.nn.functional.pad(values, (0, (-groups) % 32, 0, (-rows) % tile_rows))
    return padded.reshape(-1, tile_rows, padded.shape[1] // 32, 32).sum(dim=(1, 3))


def adaptive_representation(x, gram, packed, family, policy, tolerance, args):
    """Finite numerical mode menu; mode ordering is not a hardware energy model."""
    if policy not in {"diagonal", "gram"} or tolerance <= 0:
        raise ValueError("adaptive cases need diagonal/gram and positive relative tolerance")
    shape = x.shape
    if family == "precision" and args.selector_backend == "cuda-tile":
        flat = x.reshape(-1, shape[-1])
        low_format = getattr(args, "low_format", "int4")
        if args.group_size != 32 or getattr(args, "mode_rows", 128) != 128 or low_format == "affine4":
            raise ValueError("native tile precision supports128 rows, group32 and INT4/E2M1")
        output, modes = torch.ops.tensor_inference_engine.adaptive_precision(
            flat.contiguous(), packed, tolerance,
            low_format == "e2m1", policy == "diagonal")
        expanded = modes.repeat_interleave(128, 0).repeat_interleave(128, 1)
        expanded = expanded[:flat.shape[0], :flat.shape[1]]
        counts = torch.zeros(6, device=x.device, dtype=torch.int64)
        elements = torch.zeros_like(counts)
        for mode, index in [(0, 3), (1, 1), (2, 0)]:
            counts[index] = (modes == mode).sum()
            elements[index] = (expanded == mode).sum()
        return output.reshape(shape), counts, elements
    original = x.float().reshape(-1, shape[-1])
    candidates = [(0, original, 1.0)]
    if family in {"adaptive", "precision"}:
        for bits, dense_id, sparse_id, rank in [(8, 1, 2, .5), (4, 3, 4, .25)]:
            q = selection.quantize_symmetric(original, bits, args.group_size).bfloat16().float()
            low_format = getattr(args, "low_format", "int4")
            if bits == 4 and low_format == "affine4":
                q = selection.quantize_affine(original, 4, args.group_size).bfloat16().float()
            elif bits == 4 and low_format == "e2m1":
                q = selection.quantize_e2m1(original, args.group_size).bfloat16().float()
            candidates.append((dense_id, q, rank))
            if family == "adaptive":
                sparse, _ = sparse_selection(original, q, gram, packed, policy, args.selector_backend)
                candidates.append((sparse_id, sparse, rank / 2))
    elif family == "sparsity":
        sparse, _ = sparse_selection(original, original, gram, packed, policy, args.selector_backend)
        candidates.append((5, sparse, .5))
    else:
        raise ValueError(f"unsupported family {family}")

    def loss(error):
        if policy == "gram":
            return selection.group_loss(error, gram)
        values = error.reshape(-1, gram.shape[0], 4)
        return (values.square() * gram.diagonal(dim1=-2, dim2=-1)).sum(-1)

    tile_rows = getattr(args, "mode_rows", 128)
    denominator = sum_mode_tiles(loss(original), tile_rows).clamp_min(1e-30)
    scores = torch.stack([sum_mode_tiles(loss(values - original), tile_rows) / denominator
                          for _, values, _ in candidates])
    ranks = torch.tensor([rank for _, _, rank in candidates], device=x.device)[:, None, None]
    valid_ranks = torch.where(scores <= tolerance, ranks, torch.inf)
    best_rank = valid_ranks.amin(0)
    # Tie at equal nominal rank: use lower proxy error, then candidate order.
    choices = torch.where(valid_ranks == best_rank, scores, torch.inf).argmin(0)
    expanded = choices.repeat_interleave(tile_rows, 0).repeat_interleave(128, 1)
    expanded = expanded[:original.shape[0], :original.shape[1]]
    output = original.clone()
    counts = torch.zeros(6, device=x.device, dtype=torch.int64)
    elements = torch.zeros_like(counts)
    for index, (mode, values, _) in enumerate(candidates):
        output = torch.where(expanded == index, values, output)
        counts[mode] = (choices == index).sum()
        elements[mode] = (expanded == index).sum()
    return output.reshape(shape), counts, elements
