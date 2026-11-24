"""Conditional work/retention ledger for fixed-K, fixed-N array interfaces.

M is a streamed row count, not a physical array dimension. Vector counts are
interface work counts, not cycles. Footprints do not assert SRAM traffic or
energy: stationarity, output cadence and inter-K retention remain explicit
unknowns. FP8 denotes a proposed geometry, not a validated numerical format.
"""

from dataclasses import asdict, dataclass
from math import ceil


MIB = 1 << 20


@dataclass(frozen=True)
class Interface:
    name: str
    reduction: int
    input_bytes: int
    columns: int = 128
    output_bytes: int = 4


BF16 = Interface("bf16", 128, 2)
FP8 = Interface("fp8_geometry_only", 256, 1)


@dataclass(frozen=True)
class Convolution:
    batch: int
    height: int
    width: int
    channels: int
    outputs: int

    def __post_init__(self):
        if min(asdict(self).values()) < 1:
            raise ValueError("convolution dimensions must be positive")


def work(shape: Convolution, interface: Interface, tile: int | None = None) -> dict:
    if tile not in (None, 2, 4):
        raise ValueError("choose direct, F(2,3) or F(4,3)")
    rows = shape.batch * (shape.height if tile is None else ceil(shape.height / tile)) * (
        shape.width if tile is None else ceil(shape.width / tile))
    planes = 1 if tile is None else (tile + 2) ** 2
    reduction = shape.channels * (9 if tile is None else 1)
    segments = ceil(reduction / interface.reduction)
    column_tiles = ceil(shape.outputs / interface.columns)
    vectors = rows * planes * segments * column_tiles
    logical = rows * planes * reduction * shape.outputs
    return {
        "method": "direct" if tile is None else f"F({tile},3)",
        "interface": interface.name,
        "rows_per_plane": rows,
        "transform_planes": planes,
        "reduction": reduction,
        "k_segments": segments,
        "n_tiles": column_tiles,
        "logical_macs": logical,
        "issued_vectors": vectors,
        "padded_product_slots": vectors * interface.reduction * interface.columns,
        "lhs_interface_capacity_bytes": vectors * interface.reduction * interface.input_bytes,
        "fp32_output_capacity_bytes_if_every_segment_emits": vectors * interface.columns * interface.output_bytes,
        "weight_payload_bytes": planes * reduction * shape.outputs * interface.input_bytes,
        "weight_payload_exceeds_64mib": planes * reduction * shape.outputs * interface.input_bytes > 64 * MIB,
    }


def retention(interface: Interface, tile: int, rows: int, plane_batch: int,
              *, double_buffer_inputs: bool = False) -> dict:
    if tile not in (2, 4) or rows < 1 or not 1 <= plane_batch <= (tile + 2) ** 2:
        raise ValueError("valid transform, positive streamed rows and plane batch required")
    planes = (tile + 2) ** 2
    lhs = plane_batch * rows * interface.reduction * interface.input_bytes
    rhs = plane_batch * interface.reduction * interface.columns * interface.input_bytes
    arguments = (lhs + rhs) * (2 if double_buffer_inputs else 1)
    products = planes * rows * interface.columns * interface.output_bytes
    spatial = tile ** 2 * rows * interface.columns * interface.output_bytes
    return {
        "interface": interface.name, "tile": tile, "streamed_rows": rows,
        "plane_batch": plane_batch, "double_buffer_inputs": double_buffer_inputs,
        "lhs_bytes": lhs, "rhs_bytes": rhs, "shared_argument_bytes": arguments,
        "arguments_fit_2mib": arguments <= 2 * MIB,
        # Conservative separate lifetimes. Reuse can lower this peak but must be scheduled.
        "retained_planes_bytes": products,
        "spatial_accumulator_bytes": spatial,
        "retained_planes_plus_spatial_bytes": products + spatial,
        "retained_planes_fit_2mib": products + spatial <= 2 * MIB,
        # Streaming inverse transforms can avoid retaining every transformed plane.
        # This counts storage only, not their extra per-K arithmetic or rounding.
        "streaming_inverse_bytes": spatial + interface.columns * interface.output_bytes,
        "streaming_inverse_fit_2mib": spatial + interface.columns * interface.output_bytes <= 2 * MIB,
    }


def from_profile(layer: dict) -> Convolution | None:
    eligible = (layer["groups"] == 1 and layer["kernel_height"] == layer["kernel_width"] == 3
                and layer["stride_height"] == layer["stride_width"] == 1
                and layer["dilation_height"] == layer["dilation_width"] == 1)
    if not eligible:
        return None
    return Convolution(layer["batch"], layer["output_height"], layer["output_width"],
                       layer["input_channels"], layer["output_channels"])
