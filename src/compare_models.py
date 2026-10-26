from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "results" / "metrics"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

baseline_file = METRICS_DIR / "training_metrics.csv"
finetuned_file = METRICS_DIR / "finetuned_training_metrics.csv"

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

baseline = pd.read_csv(baseline_file)
finetuned = pd.read_csv(finetuned_file)

# Find the best validation rows
baseline_best = baseline.loc[baseline["val_accuracy"].idxmax()]
finetuned_best = finetuned.loc[finetuned["val_accuracy"].idxmax()]

baseline_val = float(baseline_best["val_accuracy"])
finetuned_val = float(finetuned_best["val_accuracy"])

# Test accuracy from final evaluation
baseline_test = 0.5556
finetuned_test = 0.6667

print()
print("BASELINE MODEL")
print("-" * 70)
print(f"Best validation accuracy: {baseline_val:.4f}")
print(f"Best epoch:               {int(baseline_best['epoch'])}")
print(f"Test accuracy:            {baseline_test:.4f}")

print()
print("FINE-TUNED MODEL")
print("-" * 70)
print(f"Best validation accuracy: {finetuned_val:.4f}")
print(f"Best epoch:               {int(finetuned_best['epoch'])}")
print(f"Test accuracy:            {finetuned_test:.4f}")

print()
print("IMPROVEMENT")
print("-" * 70)

val_improvement = finetuned_val - baseline_val
test_improvement = finetuned_test - baseline_test

print(f"Validation improvement: {val_improvement:+.4f}")
print(f"Test improvement:       {test_improvement:+.4f}")

print()
print("SUMMARY")
print("-" * 70)

if test_improvement > 0:
    print("Fine-tuning improved held-out test performance.")
else:
    print("Fine-tuning did not improve held-out test performance.")

print()
print("Results:")
print(f"Baseline test accuracy:   {baseline_test * 100:.2f}%")
print(f"Fine-tuned test accuracy: {finetuned_test * 100:.2f}%")
print(f"Absolute improvement:     {test_improvement * 100:.2f} percentage points")

print()
print("=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)