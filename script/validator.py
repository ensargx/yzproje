#!/usr/bin/env python3

import os
import numpy as np
import argparse

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def gtest_header(msg):
    print(f"{Colors.GREEN}[==========]{Colors.RESET} {msg}")

def gtest_section(msg):
    print(f"{Colors.GREEN}[----------]{Colors.RESET} {msg}")

def gtest_run(test_name):
    print(f"{Colors.GREEN}[ RUN      ]{Colors.RESET} {test_name}")

def gtest_ok(test_name, details=""):
    msg = f"{Colors.GREEN}[       OK ]{Colors.RESET} {test_name}"
    if details:
        msg += f" ({details})"
    print(msg)

def gtest_failed(test_name):
    print(f"{Colors.RED}[  FAILED  ]{Colors.RESET} {test_name}")

def gtest_warn(msg):
    print(f"{Colors.YELLOW}[ WARNING  ]{Colors.RESET} {msg}")

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def extract_points_from_grid(grid):
    points_array = np.argwhere(grid == 1)
    return [(p[0], p[1]) for p in points_array]

class DatasetValidator:
    def __init__(self, folder_name="data"):
        self.folder_name = folder_name
        self.grid_size = 25
        self.min_train = 800
        self.min_test = 200

    def load_split(self, problem_name, split="train"):
        x_path = os.path.join(self.folder_name, f"x_{split}_{problem_name}.npy")
        y_path = os.path.join(self.folder_name, f"y_{split}_{problem_name}.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            return None, None
        return np.load(x_path), np.load(y_path)

    def check_distribution(self, y_data, split_name, min_expected):
        if len(y_data) < min_expected:
            print(f"  {Colors.RED}[ HATA ]{Colors.RESET} {split_name.capitalize()} veri sayısı yetersiz. Beklenen: {min_expected}, bulunan: {len(y_data)}")
            return False

        unique, counts = np.unique(y_data, return_counts=True)
        if len(set(counts)) > 1:
            print(f"  {Colors.RED}[ HATA ]{Colors.RESET} {split_name.capitalize()} etiket (label) dağılımı dengesiz.")
            print(f"           Dağılım: {dict(zip(unique, counts))}")
            return False

        return True

    def validate_problem(self, problem_name, validation_fn):
        test_name = f"DatasetValidation.Problem{problem_name}"
        gtest_run(test_name)

        problem_passed = True
        total_samples = 0

        for split in ["train", "test"]:
            x_data, y_data = self.load_split(problem_name, split)
            if x_data is None:
                gtest_warn(f"Problem {problem_name} için '{split}' dosyaları bulunamadı, atlanıyor.")
                problem_passed = False
                continue

            total_samples += len(x_data)
            min_exp = self.min_train if split == "train" else self.min_test

            # 1. Label Dağılım Kontrolü
            if not self.check_distribution(y_data, split, min_exp):
                problem_passed = False

            errors = 0
            point_counts = [] # D ve E testleri için nokta sayılarını biriktiriyoruz

            # 2. Mantıksal Kontroller ve Nokta Sayımı
            for idx in range(len(x_data)):
                grid = x_data[idx]
                label = y_data[idx]
                pts = extract_points_from_grid(grid)
                point_counts.append(len(pts))

                is_correct, error_msg = validation_fn(pts, label)

                if not is_correct:
                    errors += 1
                    if errors <= 3:
                        print(f"  {Colors.RED}[ MANTIK HATASI ]{Colors.RESET} ({split} indeks {idx}): {error_msg}")
                        print(f"                    Beklenen etiket: {label}, Bulunan noktalar: {pts}")

            if errors > 0:
                print(f"  {split.capitalize()} setinde toplam {errors} adet mantıksal hata tespit edildi.")
                problem_passed = False

            # 3. Nokta Sayısı (1-10) Dağılım Kontrolü (Sadece D ve E için)
            if problem_name in ['D', 'E']:
                unique_pts, counts_pts = np.unique(point_counts, return_counts=True)

                # 1'den 10'a kadar tüm sınıfların var olup olmadığına ve sayılarının eşitliğine bakıyoruz
                if len(unique_pts) != 10 or len(set(counts_pts)) > 1:
                    print(f"  {Colors.RED}[ HATA ]{Colors.RESET} {split.capitalize()} setinde matris içi nokta sayısı (1-10) dengesiz!")
                    print(f"           Nokta dağılımı: {dict(zip(unique_pts, counts_pts))}")
                    problem_passed = False
                else:
                    print(f"  {Colors.CYAN}[ BİLGİ ]{Colors.RESET} {split.capitalize()} setinde nokta sayısı (1-10) tamamen dengeli (Her sayıdan {counts_pts[0]} adet).")

        if problem_passed:
            gtest_ok(test_name, f"Veri dengeli, nokta dağılımları eşit ve {total_samples} örneğin mantığı doğrulandı")
        else:
            gtest_failed(test_name)

    # --- PROBLEM DOĞRULAMA KURALLARI ---

    def check_logicA(self, pts, label):
        if len(pts) != 5:
            return False, f"Nokta sayısı 5 olmalı, {len(pts)} bulundu."
        distances = [manhattan(pts[i], pts[j]) for i in range(5) for j in range(i+1, 5)]
        min_dist = min(distances)
        if min_dist != label:
            return False, f"Hesaplanan min mesafe: {min_dist}"
        return True, ""

    def check_logicB(self, pts, label):
        if len(pts) != 5:
            return False, f"Nokta sayısı 5 olmalı, {len(pts)} bulundu."
        distances = [manhattan(pts[i], pts[j]) for i in range(5) for j in range(i+1, 5)]
        max_dist = max(distances)
        if max_dist != label:
            return False, f"Hesaplanan max mesafe: {max_dist}"
        return True, ""

    def check_logicC(self, pts, label):
        n = len(pts)
        if not (1 <= n <= 10):
            return False, f"KURAL İHLALİ: Nokta sayısı {n}. Sınırlar (1-10) dışında!"

        if n != label:
            return False, f"Gerçek nokta sayısı: {n}, Etiket: {label}"
        return True, ""

    def check_logicD(self, pts, label):
        n = len(pts)
        if not (1 <= n <= 10):
            return False, f"KURAL İHLALİ: Nokta sayısı {n}. Sınırlar (1-10) dışında!"

        parity = n % 2
        if parity != label:
            return False, f"Nokta sayısı: {n} (Hesaplanan parite: {parity}), Etiket: {label}"
        return True, ""

    def check_logicE(self, pts, label):
        n = len(pts)
        if not (1 <= n <= 10):
            return False, f"KURAL İHLALİ: Nokta sayısı {n}. Sınırlar (1-10) dışında!"

        target_corner = (0, 0) if (n % 2 == 1) else (self.grid_size - 1, self.grid_size - 1)
        min_corner_dist = min(manhattan(p, target_corner) for p in pts)

        if min_corner_dist != label:
            corner_name = "Sol üst (0,0)" if (n % 2 == 1) else "Sağ alt (24,24)"
            return False, f"Nokta sayısı: {n}. {corner_name} köşesine en kısa mesafe: {min_corner_dist}"
        return True, ""

    def run_pipeline(self):
        gtest_header("Veri seti doğrulama süreci başlatılıyor")
        gtest_section(f"Hedef dizin: {self.folder_name}/")

        self.validate_problem("A", self.check_logicA)
        self.validate_problem("B", self.check_logicB)
        self.validate_problem("C", self.check_logicC)
        self.validate_problem("D", self.check_logicD)
        self.validate_problem("E", self.check_logicE)

        gtest_header("Doğrulama testleri tamamlandı")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset validation script (GTest format)")
    parser.add_argument("--data_dir", type=str, default="data", help="Veri dizini yolu")
    args = parser.parse_args()

    validator = DatasetValidator(folder_name=args.data_dir)
    validator.run_pipeline()
