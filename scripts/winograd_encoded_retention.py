"""Capacity conditions for reusing one encoded source across receiver tiles.

This ledger proposes storage placement only. SRAM ports, producer delivery,
DMA overlap and execution cycles require a separate implementation.
"""

from math import ceil

from winograd_mapping import MIB


def retention(channels, outputs, *, tile=2, rows=64, plane_batch=4, rhs_buffers=2):
    if (min(channels, outputs, rows, plane_batch, rhs_buffers) < 1
            or tile not in (2, 4) or plane_batch > (tile + 2) ** 2):
        raise ValueError("positive dimensions and valid Winograd tile/plane batch required")
    planes, reduction, columns = (tile + 2) ** 2, 256, 128
    groups = ceil(channels / reduction)
    padded = groups * reduction
    encoded = planes * rows * padded
    source_scales = planes * rows * groups * 2
    rhs = plane_batch * reduction * columns * rhs_buffers
    rhs_scales = plane_batch * columns * 2 * rhs_buffers
    state = rows * reduction * 4
    coefficients = planes * reduction * 2
    arguments = encoded + source_scales + rhs + rhs_scales
    colocated = arguments + state + coefficients
    products = planes * rows * columns * 4
    spatial = tile * tile * rows * columns * 4
    weight_payload = planes * padded * outputs
    source_materialized = planes * rows * channels * 2
    return {
        "channels": channels, "outputs": outputs, "tile": tile, "rows": rows,
        "plane_batch": plane_batch, "rhs_buffers": rhs_buffers, "K": reduction, "N": columns,
        "encoded_source_cache_bytes": encoded, "source_exponent_bytes_int16": source_scales,
        "rhs_bytes": rhs, "rhs_exponent_bytes_int16": rhs_scales,
        "controller_state_bytes_current_k": state,
        "controller_metadata_bytes_current_k": coefficients,
        "shared_argument_bytes_with_separate_controller_storage": arguments,
        "shared_argument_bytes_if_controller_storage_colocated": colocated,
        "arguments_with_colocated_controller_fit_2mib": colocated <= 2 * MIB,
        "retained_product_planes_bytes_current_n": products,
        "spatial_result_bytes_current_n": spatial,
        "products_plus_spatial_fit_2mib": products + spatial <= 2 * MIB,
        "full_prepared_weight_payload_int8_padded_bytes": weight_payload,
        "source_materialized_tile_bf16_bytes": source_materialized,
        "weights_plus_materialized_source_fit_general_64mib": weight_payload + source_materialized <= 64 * MIB,
        "receiver_column_tiles_reusing_source": ceil(outputs / columns),
    }
