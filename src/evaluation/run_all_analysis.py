import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_SCRIPTS = [
    "compare_models.py",
    "plot_model_comparison.py",
    "prediction_analysis.py",
    "plot_error_analysis.py",
    "plot_confusion_matrix.py",
    "plot_training_curves.py",
    "plot_runtime_comparison.py",
]


def run_script(script_name: str) -> None:
    """Run one evaluation script from the project root."""
    script_path = (
        PROJECT_ROOT
        / "src"
        / "evaluation"
        / script_name
    )

    if not script_path.exists():
        raise FileNotFoundError(
            f"Analysis script not found: {script_path}"
        )

    print("\n" + "=" * 70)
    print(f"Running: {script_name}")
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with "
            f"exit code {result.returncode}."
        )

    print(f"Completed: {script_name}")


def main() -> None:
    print("Starting complete evaluation and visualisation pipeline.")

    for script_name in ANALYSIS_SCRIPTS:
        run_script(script_name)

    print("\n" + "=" * 70)
    print("All analysis scripts completed successfully.")
    print("=" * 70)

    print(
        "\nGenerated outputs are available in:\n"
        f"- {PROJECT_ROOT / 'results' / 'tables'}\n"
        f"- {PROJECT_ROOT / 'results' / 'figures'}"
    )


if __name__ == "__main__":
    main()