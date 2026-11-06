"""
Parse and display all experiment results from exms directory
"""

import argparse
import re
from pathlib import Path


def _parse_metric(text, name):
    number = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
    match = re.search(
        r"\|\s*" + re.escape(name) + r"\s*\|[^|]*\|\s*" + number
        + r"\s*\|\s*(?:±|\+/-)\s*\|\s*" + number + r"\s*\|", text
    )
    return tuple(map(float, match.groups())) if match else None


def parse_result_file(filepath):
    """Parse a single result file and extract experiment data"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract configuration from filename
    filename = Path(filepath).name
    # Canonical pattern:
    #   {model}_{routing}_{activation}_fixed_{silu}_batch_{batch}_acc.txt
    # Hyphens in CLI mode names are rendered as underscores in filenames.
    canonical_profiles = {
        "original_silu": ("original", "silu"),
        "float_all_experts_silu": ("float-all-experts", "silu"),
        "capricorn_piecewise_fixed_silu_16_-8_8": (
            "capricorn-piecewise-fixed",
            "silu_16_-8_8",
        ),
        "capricorn_polynomial_fixed_silu_poly_approx": (
            "capricorn-polynomial-fixed",
            "silu_poly_approx",
        ),
        "cryptomoe_piecewise_fixed_silu_16_-8_8": (
            "cryptomoe-piecewise-fixed",
            "silu_16_-8_8",
        ),
        "cryptomoe_polynomial_fixed_silu_poly_approx": (
            "cryptomoe-polynomial-fixed",
            "silu_poly_approx",
        ),
    }
    match = re.match(
        r"(.+?)_("
        + "|".join(
            sorted(canonical_profiles, key=len, reverse=True)
        )
        + r")_batch_(\d+)_acc\.txt$",
        filename,
    )

    if not match:
        return None

    model_name = match.group(1)
    mode, activation = canonical_profiles[match.group(2)]
    batch_size = match.group(3)

    # Find all experiment runs (experts=60 or experts=4, etc.)
    experiments = []

    # Split by expert configuration
    expert_sections = re.split(r"Running tests with num_experts_per_tok=(\d+)", content)

    for i in range(1, len(expert_sections), 2):
        if i >= len(expert_sections):
            break

        num_experts = expert_sections[i]
        section = expert_sections[i + 1] if i + 1 < len(expert_sections) else ""

        # Find all datasets in this section
        datasets = []

        # Find all dataset test sections
        dataset_sections = re.findall(
            r">>> Testing dataset: (\w+) .*?experts=(\d+)(.*?)(?=>>> Testing dataset:|Running tests with num_experts_per_tok|={10,}|\Z)",
            section,
            re.DOTALL,
        )

        for dataset_name, expert_num, ds_section in dataset_sections:
            acc = _parse_metric(ds_section, "acc")
            norm = _parse_metric(ds_section, "acc_norm")
            if acc is None and norm is None:
                continue
            if acc is None:
                acc = norm
            datasets.append({
                "name": dataset_name,
                "acc": acc[0],
                "stderr": acc[1],
                "acc_norm": norm[0] if norm is not None else None,
                "acc_norm_stderr": norm[1] if norm is not None else None,
            })

        if datasets:
            experiments.append({"num_experts": int(num_experts), "datasets": datasets})

    return {
        "model": model_name,
        "mode": mode,
        "activation": activation,
        "batch_size": batch_size,
        "experiments": experiments,
    }


def display_results(results):
    """Display results in formatted tables"""

    for result in results:
        model = result["model"]
        activation = result["activation"]
        mode = result["mode"] or "legacy"
        batch_size = result["batch_size"]

        for exp in result["experiments"]:
            num_experts = exp["num_experts"]
            datasets = exp["datasets"]

            print(f"\n{'='*100}")
            print(
                f"Model: {model} | Mode: {mode} | Activation: {activation} | "
                f"Batch Size: {batch_size} | Experts: {num_experts}"
            )
            print(f"{'='*100}")
            print(f"{'Dataset':<15} | {'Metric':<10} | {'Accuracy':<10} | {'Stderr':<10}")
            print(f"{'-'*100}")

            scores = []
            errors = []
            for ds in datasets:
                normalized = ds["acc_norm"] is not None
                score = ds["acc_norm"] if normalized else ds["acc"]
                error = ds["acc_norm_stderr"] if normalized else ds["stderr"]
                scores.append(score)
                errors.append(error)
                metric = "acc_norm" if normalized else "acc"
                print(f"{ds['name']:<15} | {metric:<10} | {score:<10.4f} | {error:<10.4f}")
            avg_acc = sum(scores) / len(scores)
            avg_stderr = sum(errors) / len(errors)
            print(f"{'-'*100}")
            print(f"{'Average':<15} | {'':<10} | {avg_acc:<10.4f} | {avg_stderr:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Summarize plaintext paper accuracy logs")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "exms")
    exms_dir = parser.parse_args().log_dir

    if not exms_dir.exists():
        parser.error(f"Directory {exms_dir} not found")

    # Get all result files
    result_files = sorted(exms_dir.glob("*_acc.txt"))

    print(f"Found {len(result_files)} result files")

    # Parse all files
    all_results = []
    for filepath in result_files:
        result = parse_result_file(filepath)
        if result:
            all_results.append(result)

    # Display results grouped by model
    models = set(r["model"] for r in all_results)

    for model in sorted(models):
        print(f"\n\n{'#'*100}")
        print(f"# {model:^96} #")
        print(f"{'#'*100}")

        model_results = [r for r in all_results if r["model"] == model]
        display_results(model_results)


if __name__ == "__main__":
    main()
