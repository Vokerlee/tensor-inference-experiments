"""Operation/retention ledger for a causal POT plane encoder, without energy constants.

Counts distinguish logical scalar operations, locally retained state and interface
work. They do not assert cycle counts, mandatory SRAM transactions or joules.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ControllerTile:
    rows: int = 64
    reduction: int = 256
    planes: int = 16
    outputs: int = 128
    state_bytes: int = 4
    metadata_bytes: int = 2

    def __post_init__(self):
        if min(asdict(self).values()) < 1:
            raise ValueError("positive dimensions and payload widths required")


def ledger(tile: ControllerTile) -> dict:
    # Count above ordinary nearest rounding: two residual subtractions and
    # three state/criterion additions. POT adjustments include the extra
    # adjacent reconstruction, projection, h term, doubling and state update.
    elements = tile.rows * tile.reduction * tile.planes
    macs = elements * tile.outputs
    state = tile.rows * tile.reduction * tile.state_bytes
    metadata = tile.planes * tile.reduction * tile.metadata_bytes
    return {
        **asdict(tile), "encoded_elements": elements, "useful_macs": macs,
        "extra_scalar_fp32_add_sub": 5 * elements,
        "extra_logical_power2_adjustments": 5 * elements,
        "state_payload_bytes": state, "immutable_metadata_bytes": metadata,
        "state_access_capacity_bytes_if_each_update_reads_and_writes": 2 * elements * tile.state_bytes,
        "metadata_preload_bytes_per_encoded_element_if_reused_across_rows": metadata / elements,
        "extra_fp32_add_sub_per_useful_mac": 5 / tile.outputs,
        "state_access_bytes_per_useful_mac": 2 * tile.state_bytes / tile.outputs,
        "metadata_preload_bytes_per_useful_mac": metadata / macs,
        "state_port_capacity_bytes_per_vector_if_full_reduction_parallel": 2 * tile.reduction * tile.state_bytes,
        "bf16_source_bytes_per_vector": 2 * tile.reduction,
        "int8_encoded_bytes_per_vector": tile.reduction,
        "int4_codes_in_int8_words_bytes_per_vector": tile.reduction,
        "int4_packed_bytes_per_vector_if_supported": (tile.reduction + 1) // 2,
    }
