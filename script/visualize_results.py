import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metric(df, x_col, y_cols, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], label=col)
    
    plt.title(title)
    plt.xlabel(x_col.capitalize())
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_results(results_dir="results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Directory {results_dir} does not exist.")
        return

    problem_summaries = {}

    # Find all loss_history.csv and summary.csv files
    for history_file in results_path.glob("**/loss_history.csv"):
        print(f"Processing individual history: {history_file}...")
        try:
            df = pd.read_csv(history_file)
            output_dir = history_file.parent

            # Plot Loss
            if 'train_loss' in df.columns:
                plot_metric(
                    df, 'epoch', ['train_loss'], 
                    f"Loss History - {output_dir.relative_to(results_path)}", 
                    "Loss", output_dir / "loss.png"
                )

            # Plot Accuracy
            acc_cols = [col for col in ['train_accuracy', 'test_accuracy'] if col in df.columns]
            if acc_cols:
                plot_metric(
                    df, 'epoch', acc_cols, 
                    f"Accuracy History - {output_dir.relative_to(results_path)}", 
                    "Accuracy", output_dir / "accuracy.png"
                )

            # Plot MAE
            mae_cols = [col for col in ['train_mae', 'test_mae'] if col in df.columns]
            if mae_cols:
                plot_metric(
                    df, 'epoch', mae_cols, 
                    f"MAE History - {output_dir.relative_to(results_path)}", 
                    "Mean Absolute Error", output_dir / "mae.png"
                )

            # Regression Scatter Plot
            preds_file = output_dir / "predictions.csv"
            if preds_file.exists():
                preds_df = pd.read_csv(preds_file)
                if 'true' in preds_df.columns and 'pred' in preds_df.columns:
                    plt.figure(figsize=(8, 8))
                    plt.scatter(preds_df['true'], preds_df['pred'], alpha=0.5)
                    
                    # Ideal line
                    max_val = max(preds_df['true'].max(), preds_df['pred'].max())
                    min_val = min(preds_df['true'].min(), preds_df['pred'].min())
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
                    
                    plt.title(f"True vs Predicted - {output_dir.relative_to(results_path)}")
                    plt.xlabel("True Values")
                    plt.ylabel("Predicted Values")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(output_dir / "scatter_preds.png")
                    plt.close()

            # Collect data for problem-level comparison
            summary_file = output_dir / "summary.csv"
            if summary_file.exists():
                summary_df = pd.read_csv(summary_file)
                problem = summary_df['problem'].iloc[0]
                if problem not in problem_summaries:
                    problem_summaries[problem] = []
                problem_summaries[problem].append(summary_df)

        except Exception as e:
            print(f"Error processing {history_file}: {e}")

    # Problem-level comparisons
    for problem, summaries in problem_summaries.items():
        print(f"Generating comparison plots for Problem {problem}...")
        combined_df = pd.concat(summaries).sort_values('train_fraction')
        problem_dir = results_path / problem
        
        # Plot Accuracy vs Fraction
        if 'test_accuracy' in combined_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(combined_df['train_fraction'], combined_df['test_accuracy'], marker='o', label='Test Accuracy')
            if 'train_accuracy' in combined_df.columns:
                plt.plot(combined_df['train_fraction'], combined_df['train_accuracy'], marker='x', linestyle='--', label='Train Accuracy')
            plt.title(f"Accuracy vs Train Fraction - Problem {problem}")
            plt.xlabel("Train Fraction")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig(problem_dir / "comparison_accuracy.png")
            plt.close()

        # Plot MAE vs Fraction
        if 'test_mae' in combined_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(combined_df['train_fraction'], combined_df['test_mae'], marker='o', label='Test MAE')
            if 'train_mae' in combined_df.columns:
                plt.plot(combined_df['train_fraction'], combined_df['train_mae'], marker='x', linestyle='--', label='Train MAE')
            plt.title(f"MAE vs Train Fraction - Problem {problem}")
            plt.xlabel("Train Fraction")
            plt.ylabel("MAE")
            plt.legend()
            plt.grid(True)
            plt.savefig(problem_dir / "comparison_mae.png")
            plt.close()

if __name__ == "__main__":
    visualize_results()
