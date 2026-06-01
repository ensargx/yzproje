import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_consolidated(results_dir="results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory {results_dir} does not exist.")
        return

    # Problem bazlı verileri topla
    # Structure: { 'A': { 'train_25pct': df, 'train_50pct': df, ... }, 'B': ... }
    problem_data = {}

    for history_file in results_path.glob("**/loss_history.csv"):
        try:
            df = pd.read_csv(history_file)
            if df.empty:
                continue
            
            # Path parts: results / [Problem] / [Fraction] / loss_history.csv
            parts = history_file.relative_to(results_path).parts
            if len(parts) < 2:
                continue
            
            problem = parts[0]
            fraction_label = parts[1]

            if problem not in problem_data:
                problem_data[problem] = {}
            
            problem_data[problem][fraction_label] = df
        except Exception as e:
            print(f"Error reading {history_file}: {e}")

    for problem, fractions in problem_data.items():
        print(f"Generating consolidated plots for Problem {problem}...")
        problem_dir = results_path / problem
        
        # 1. Consolidated Loss Plot
        plt.figure(figsize=(12, 7))
        for label, df in sorted(fractions.items()):
            if 'train_loss' in df.columns:
                plt.plot(df['epoch'], df['train_loss'], label=f"Loss ({label})")
        plt.title(f"Problem {problem} - Training Loss Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log') # Kayıp genelde log scale'de daha iyi görünür
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.savefig(problem_dir / "consolidated_loss.png", dpi=300)
        plt.close()

        # 2. Consolidated Accuracy Plot (Test Acc)
        plt.figure(figsize=(12, 7))
        for label, df in sorted(fractions.items()):
            if 'test_accuracy' in df.columns:
                plt.plot(df['epoch'], df['test_accuracy'], label=f"Test Acc ({label})")
        plt.title(f"Problem {problem} - Test Accuracy Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(problem_dir / "consolidated_accuracy.png", dpi=300)
        plt.close()

        # 3. Consolidated MAE Plot (Test MAE - only if regression)
        has_mae = any('test_mae' in df.columns for df in fractions.values())
        if has_mae:
            plt.figure(figsize=(12, 7))
            for label, df in sorted(fractions.items()):
                if 'test_mae' in df.columns:
                    plt.plot(df['epoch'], df['test_mae'], label=f"Test MAE ({label})")
            plt.title(f"Problem {problem} - Test MAE Comparison")
            plt.xlabel("Epoch")
            plt.ylabel("Mean Absolute Error")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(problem_dir / "consolidated_mae.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    visualize_consolidated()
