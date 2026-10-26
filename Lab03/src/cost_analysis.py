from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import torch
from thop import profile
from torchinfo import summary

from models import CNN, MLP, PatchTransformer


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LATENCY_RUNS = {1: 100, 128: 20}
WARMUP_RUNS = 10


def hand_calculated_flops() -> dict[str, dict[str, int]]:
    sequence_length = 16
    d_model = 64
    feedforward_size = 128

    transformer_layer_flops = {
        "qkv_projections_per_encoder_layer": (
            2 * sequence_length * d_model * 3 * d_model
        ),
        "attention_scores_per_encoder_layer": 2 * sequence_length**2 * d_model,
        "attention_weighted_values_per_encoder_layer": (
            2 * sequence_length**2 * d_model
        ),
        "attention_output_projection_per_encoder_layer": (
            2 * sequence_length * d_model * d_model
        ),
        "feedforward_per_encoder_layer": (
            2 * sequence_length * d_model * feedforward_size
            + 2 * sequence_length * feedforward_size * d_model
        ),
    }
    transformer_layers = {
        name: flops * 2 for name, flops in transformer_layer_flops.items()
    }

    models = {
        "MLP": {
            "linear_784_to_256": 2 * 784 * 256,
            "linear_256_to_128": 2 * 256 * 128,
            "linear_128_to_10": 2 * 128 * 10,
        },
        "CNN": {
            "conv_1_to_16": 2 * 3 * 3 * 1 * 16 * 28 * 28,
            "conv_16_to_32": 2 * 3 * 3 * 16 * 32 * 14 * 14,
            "linear_1568_to_10": 2 * 32 * 7 * 7 * 10,
        },
        "PatchTransformer": {
            "patch_projection": 2 * sequence_length * 49 * d_model,
            **transformer_layers,
            "linear_64_to_10": 2 * d_model * 10,
        },
    }
    for layer_flops in models.values():
        layer_flops["total"] = sum(layer_flops.values())
    return models


def median_inference_ms(model: torch.nn.Module, batch_size: int, runs: int) -> float:
    inputs = torch.randn(batch_size, 1, 28, 28)
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(inputs)

        durations_ms = []
        for _ in range(runs):
            start = time.perf_counter()
            model(inputs)
            durations_ms.append((time.perf_counter() - start) * 1000)
    return statistics.median(durations_ms)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hand_flops = hand_calculated_flops()
    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "PatchTransformer": PatchTransformer(),
    }
    rows = []
    summary_lines = []

    for name, model in models.items():
        model.eval()
        model_summary = summary(
            model,
            input_size=(1, 1, 28, 28),
            device="cpu",
            verbose=0,
        )
        torchinfo_parameters = int(model_summary.total_params)
        summary_lines.extend([name, str(model_summary), ""])

        profiler_input = torch.randn(1, 1, 28, 28)
        thop_macs, thop_parameters = profile(
            model,
            inputs=(profiler_input,),
            verbose=False,
        )
        hand_total = hand_flops[name]["total"]
        thop_flops = 2 * thop_macs
        gap_percent = (hand_total - thop_flops) / hand_total * 100

        row = {
            "model": name,
            "torchinfo_parameters": torchinfo_parameters,
            "thop_parameters": int(thop_parameters),
            "hand_calculated_flops_per_image": hand_total,
            "thop_macs_per_image": int(thop_macs),
            "thop_flops_estimate_per_image": int(thop_flops),
            "hand_minus_thop_percent": round(gap_percent, 2),
            "batch_1_median_ms_100_runs": round(
                median_inference_ms(model, 1, LATENCY_RUNS[1]), 6
            ),
            "batch_128_median_ms_20_runs": round(
                median_inference_ms(model, 128, LATENCY_RUNS[128]), 6
            ),
        }
        rows.append(row)
        print(
            f"{name}: {torchinfo_parameters:,} parameters; "
            f"hand FLOPs/image={hand_total:,}; "
            f"THOP={int(thop_flops):,} FLOPs/image; "
            f"batch 1={row['batch_1_median_ms_100_runs']:.6f} ms; "
            f"batch 128={row['batch_128_median_ms_20_runs']:.6f} ms"
        )
        print("Hand-calculated operation FLOPs:")
        for operation, flops in hand_flops[name].items():
            print(f"  {operation}: {flops:,}")

    csv_path = RESULTS_DIR / "cost_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = RESULTS_DIR / "torchinfo_summaries.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved measurements to {csv_path}")
    print(f"Saved model summaries to {summary_path}")


if __name__ == "__main__":
    main()