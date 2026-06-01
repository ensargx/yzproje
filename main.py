import argparse

from src.experiment import run_all

def parse_args():
    parser = argparse.ArgumentParser(description="Problem A-E modellerini eğit ve sonuçları kaydet.")
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        help="Çalıştırılacak problemler. Örn: --problems A C E"
    )
    parser.add_argument("--data-dir", default="data", help="Veri seti klasörü")
    parser.add_argument("--results-dir", default="results", help="Sonuçların kaydedileceği klasör")
    parser.add_argument("--epochs", type=int, default=None, help="Test için tüm problemlerde epoch sayısını override eder")
    parser.add_argument(
        "--train-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 1.00],
        help="Eğitim verisi oranları. Varsayılan: 0.25 0.50 1.00"
    )
    parser.add_argument("--seed", type=int, default=24, help="Tekrarlanabilirlik (reproducibility) için rastgelelik tohumu (seed)")

    return parser.parse_args()


def main():
    args = parse_args()

    results = run_all(
        problems=args.problems,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        seed=args.seed,
        train_fractions=args.train_fractions,
    )

    print("\n===== Tüm Sonuçlar =====")
    for result in results:
        if result["task_type"] == "classification":
            print(
                f"Problem {result['problem']} | "
                f"{result['train_fraction_label']} | "
                f"Train: {result['train_size']}/{result['train_total']} | "
                f"Test Acc: %{result['test_accuracy']*100:.2f} | "
                f"Hatalı: {result['wrong_count']}"
            )
        else:
            print(
                f"Problem {result['problem']} | "
                f"{result['train_fraction_label']} | "
                f"Train: {result['train_size']}/{result['train_total']} | "
                f"Test MAE: {result['test_mae']:.4f} | "
                f"Exact Acc: %{result['exact_accuracy']*100:.2f} | "
                f"Hatalı: {result['wrong_count']}"
            )

    print(f"\nSonuçlar '{args.results_dir}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
