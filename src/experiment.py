from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models import get_model
from util import create_matrix_figure, load_dataset


@dataclass
class ProblemConfig:
    problem_name: str
    task_type: str
    batch_size: int
    epochs: int
    log_interval: int
    tolerance: float = 1.0


PROBLEM_CONFIGS = {
    'A': ProblemConfig('A', 'regression', 1024, 10000, 1000),
    'B': ProblemConfig('B', 'regression', 1024, 10000, 1000),
    'C': ProblemConfig('C', 'regression', 1024, 10000, 1000),
    'D': ProblemConfig('D', 'classification', 512, 1000, 100),
    'E': ProblemConfig('E', 'distance', 512, 10000, 100),
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_config(problem_name):
    problem_name = str(problem_name).upper()
    if problem_name not in PROBLEM_CONFIGS:
        raise ValueError("Geçersiz problem adı! Lütfen 'A', 'B', 'C', 'D' veya 'E' kullanın.")
    return PROBLEM_CONFIGS[problem_name]


def create_loaders(x_train, y_train, x_test, y_test, batch_size):
    x_train_tensor = torch.tensor(x_train).float()
    y_train_tensor = torch.tensor(y_train).float()
    x_test_tensor = torch.tensor(x_test).float()
    y_test_tensor = torch.tensor(y_test).float()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_criterion(task_type):
    if task_type == "classification":
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


def prepare_targets(batch_y, task_type):
    if task_type == "classification":
        return batch_y.float().squeeze()
    if task_type == "distance":
        return batch_y.float().squeeze()
    return batch_y.float().view(-1, 1)


def prepare_predictions(predictions, task_type):
    if task_type in ["classification", "distance"]:
        return predictions.squeeze()
    return predictions


def predict(model, x_data, task_type=None):
    model.eval()

    if not isinstance(x_data, torch.Tensor):
        x_data = torch.tensor(x_data, dtype=torch.float32)

    is_single = (x_data.dim() == 2)

    if is_single:
        x_data = x_data.unsqueeze(0)

    device = next(model.parameters()).device
    x_data = x_data.to(device)

    with torch.no_grad():
        sonuc = model(x_data)

    if task_type == "classification":
        sonuc_tensor = sonuc.float().view(-1).int()
    else:
        sonuc_tensor = torch.round(sonuc).int().view(-1)

    if is_single:
        return sonuc_tensor.item()
    else:
        return sonuc_tensor.cpu()


def train_one_epoch(model, train_loader, criterion, optimizer, device, config):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_close = 0
    train_error_sum = 0.0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = prepare_targets(batch_y.to(device), config.task_type)

        optimizer.zero_grad()
        predictions = prepare_predictions(model(batch_x), config.task_type)

        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        with torch.no_grad():
            if config.task_type == "classification":
                predicted_labels = (predictions > 0.0).float()
                train_correct += (predicted_labels == batch_y).sum().item()
            else:
                predicted_labels = torch.round(predictions)
                abs_errors = torch.abs(predicted_labels - batch_y)
                train_correct += (predicted_labels == batch_y).sum().item()
                train_close += (abs_errors <= config.tolerance).sum().item()
                train_error_sum += abs_errors.sum().item()

            total_samples += batch_y.numel()

    metrics = {
        "train_loss": train_loss / len(train_loader),
        "train_accuracy": train_correct / total_samples,
    }

    if config.task_type != "classification":
        metrics["train_close_accuracy"] = train_close / total_samples
        metrics["train_mae"] = train_error_sum / total_samples

    return metrics


def evaluate(model, test_loader, device, config):
    model.eval()
    test_correct = 0
    test_close = 0
    test_error_sum = 0.0
    test_total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = prepare_targets(batch_y.to(device), config.task_type)

            predictions = prepare_predictions(model(batch_x), config.task_type)

            if config.task_type == "classification":
                predicted_labels = (predictions > 0.0).float()
                test_correct += (predicted_labels == batch_y).sum().item()
            else:
                predicted_labels = torch.round(predictions)
                abs_errors = torch.abs(predicted_labels - batch_y)
                test_correct += (predicted_labels == batch_y).sum().item()
                test_close += (abs_errors <= config.tolerance).sum().item()
                test_error_sum += abs_errors.sum().item()

            test_total += batch_y.numel()

    metrics = {
        "test_accuracy": test_correct / test_total,
    }

    if config.task_type != "classification":
        metrics["test_close_accuracy"] = test_close / test_total
        metrics["test_mae"] = test_error_sum / test_total

    return metrics


def print_epoch(epoch, config, metrics):
    if (epoch + 1) % config.log_interval != 0:
        return

    if config.task_type == "classification":
        print(
            f"Epoch [{epoch+1}/{config.epochs}] | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Train Acc: %{metrics['train_accuracy']*100:.2f} | "
            f"Test Acc: %{metrics['test_accuracy']*100:.2f}"
        )
    else:
        print(
            f"Epoch [{epoch+1}/{config.epochs}] | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Test Error: {metrics['test_mae']:.4f} | "
            f"Test Acc(±{int(config.tolerance)}): %{metrics['test_close_accuracy']*100:.1f}"
        )


def plot_error_distribution(error_diffs, save_path):
    diff_map = dict(Counter(error_diffs))
    print("\nHata Farkı Haritası (Fark: Adet):", diff_map)

    if error_diffs:
        sorted_diffs = sorted(diff_map.keys())
        counts = [diff_map[d] for d in sorted_diffs]
        x_labels = [str(d) for d in sorted_diffs]
    else:
        sorted_diffs = [0]
        counts = [0]
        x_labels = ["0"]

    fig_width = max(8, len(sorted_diffs) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars = ax.bar(x_labels, counts, color='#FF6F61', edgecolor='#CC3333', width=0.4, zorder=3)

    if len(sorted_diffs) < 6:
        ax.set_xlim(-0.5, 5.5)

    max_count = max(counts) if counts else 1
    y_ekseni_maksimum = 100
    ax.set_ylim(0, max(y_ekseni_maksimum, max_count + (max_count * 0.1)))

    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + (ax.get_ylim()[1] * 0.015),
                int(yval),
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

    ax.set_title('Yanlış Tahminlerdeki Hata Büyüklüğü Dağılımı', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Mutlak Hata Farkı (|Tahmin - Gerçek|)', fontsize=12, labelpad=10)
    ax.set_ylabel('Hata Adedi', fontsize=12, labelpad=10)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_sample_images(x_data, samples, save_dir, prefix):
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for count, (idx, true_y, pred_y) in enumerate(samples, start=1):
        matris = x_data[idx]
        fig = create_matrix_figure(matris)
        sample_path = save_dir / f"{prefix}_{count:02d}_idx_{idx}_true_{int(true_y)}_pred_{int(pred_y)}.png"
        fig.savefig(sample_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        rows.append({
            "index": int(idx),
            "true": int(true_y),
            "pred": int(pred_y),
            "path": str(sample_path),
        })

    columns = ["index", "true", "pred", "path"]
    pd.DataFrame(rows, columns=columns).to_csv(save_dir / f"{prefix}.csv", index=False)


def analyze_predictions(model, x_test, y_test, config, result_dir, random_sample_count=5, wrong_sample_count=5):
    preds = predict(model, x_test, task_type=config.task_type).numpy()

    prediction_rows = []
    wrong_indices = []
    error_diffs = []

    for i, (pred, y) in enumerate(zip(preds, y_test, strict=True)):
        pred_int = int(pred)
        true_int = int(y)
        diff = abs(pred_int - true_int)

        prediction_rows.append({
            "index": i,
            "true": true_int,
            "pred": pred_int,
            "abs_error": diff,
            "is_correct": pred_int == true_int,
        })

        if pred_int != true_int:
            wrong_indices.append((i, true_int, pred_int))
            error_diffs.append(diff)

    pd.DataFrame(prediction_rows).to_csv(result_dir / "predictions.csv", index=False)
    plot_error_distribution(error_diffs, result_dir / "error_distribution.png")

    print(f"\n--- İlk {wrong_sample_count} Hatalı Tahmin Kaydediliyor ---")
    save_sample_images(
        x_test,
        wrong_indices[:wrong_sample_count],
        result_dir / "wrong_predictions",
        "wrong"
    )

    rng = np.random.default_rng(11337)
    random_count = min(random_sample_count, len(x_test))
    random_indices = rng.choice(len(x_test), size=random_count, replace=False)
    random_samples = [(idx, y_test[idx], preds[idx]) for idx in random_indices]
    save_sample_images(
        x_test,
        random_samples,
        result_dir / "random_samples",
        "random"
    )

    return {
        "wrong_count": len(wrong_indices),
        "exact_accuracy": 1.0 - (len(wrong_indices) / len(y_test)),
    }


def run_problem(problem_name, data_dir="data", results_dir="results", epochs=None):
    config = get_config(problem_name)
    if epochs is not None:
        config = ProblemConfig(
            config.problem_name,
            config.task_type,
            config.batch_size,
            int(epochs),
            min(config.log_interval, max(1, int(epochs))),
            config.tolerance
        )

    result_dir = Path(results_dir) / config.problem_name
    result_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_dataset(config.problem_name, folder_name=data_dir)
    train_loader, test_loader = create_loaders(x_train, y_train, x_test, y_test, config.batch_size)

    device = get_device()
    model = get_model(config.problem_name).to(device)
    criterion = get_criterion(config.task_type)
    optimizer = optim.AdamW(model.parameters())

    print(f"\n===== Problem {config.problem_name} eğitiliyor ({device}) =====")

    history = []
    log_path = result_dir / "loss_history.csv"

    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config)
        test_metrics = evaluate(model, test_loader, device, config)
        metrics = {**train_metrics, **test_metrics}

        history.append({
            "epoch": epoch + 1,
            **metrics,
        })

        pd.DataFrame(history).to_csv(log_path, index=False)
        print_epoch(epoch, config, metrics)

    torch.save(model.state_dict(), result_dir / "model.pt")

    prediction_metrics = analyze_predictions(model, x_test, y_test, config, result_dir)

    final_metrics = {
        "problem": config.problem_name,
        "task_type": config.task_type,
        "epochs": config.epochs,
        **history[-1],
        **prediction_metrics,
    }

    pd.DataFrame([final_metrics]).to_csv(result_dir / "summary.csv", index=False)

    return final_metrics


def run_all(problems=None, data_dir="data", results_dir="results", epochs=None):
    if problems is None:
        problems = ['A', 'B', 'C', 'D', 'E']

    all_results = []
    for problem_name in problems:
        result = run_problem(problem_name, data_dir=data_dir, results_dir=results_dir, epochs=epochs)
        all_results.append(result)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(results_path / "summary.csv", index=False)

    return all_results
