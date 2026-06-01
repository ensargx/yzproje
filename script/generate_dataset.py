#!/usr/bin/env python3

import os
import numpy as np
import argparse

SEED = 42

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_points_at_exact_distance(center_point, target_dist):
    points = []
    for dx in range(-target_dist, target_dist + 1):
        dy = target_dist - abs(dx)
        for yOffset in (dy, -dy):
            nx, ny = center_point[0] + dx, center_point[1] + yOffset
            if 0 <= nx < 25 and 0 <= ny < 25:
                points.append((nx, ny))
    return sorted(list(set(points)))

class TargetDrivenDatasetGenerator:
    def __init__(self, seed=SEED):
        self.grid_size = 25
        self.min_total_train = 1024
        self.min_total_test = 256
        self.rng = np.random.default_rng(seed)

    def generate_sampleA(self, target_dist):
        max_retries = 200
        for _ in range(max_retries):
            p1X, p1Y = self.rng.integers(0, self.grid_size), self.rng.integers(0, self.grid_size)
            p1 = (p1X, p1Y)
            ring_points = get_points_at_exact_distance(p1, target_dist)
            if not ring_points:
                continue

            p2 = ring_points[self.rng.integers(len(ring_points))]
            selected_points = [p1, p2]
            is_valid = True

            for _ in range(3):
                found_point = False
                for _ in range(100):
                    candX, candY = self.rng.integers(0, self.grid_size), self.rng.integers(0, self.grid_size)
                    candidate = (candX, candY)
                    if candidate in selected_points:
                        continue

                    distances = [manhattan(candidate, existing) for existing in selected_points]
                    if min(distances) >= target_dist:
                        selected_points.append(candidate)
                        found_point = True
                        break

                if not found_point:
                    is_valid = False
                    break

            if is_valid:
                all_distances = [manhattan(selected_points[i], selected_points[j]) 
                                for i in range(5) for j in range(i+1, 5)]
                if min(all_distances) == target_dist:
                    grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
                    for px, py in selected_points: grid[px, py] = 1
                    return grid
        return None

    def generate_sampleB(self, target_dist):
        max_retries = 200
        for _ in range(max_retries):
            p1X, p1Y = self.rng.integers(0, self.grid_size), self.rng.integers(0, self.grid_size)
            p1 = (p1X, p1Y)
            ring_points = get_points_at_exact_distance(p1, target_dist)
            if not ring_points:
                continue

            p2 = ring_points[self.rng.integers(len(ring_points))]
            selected_points = [p1, p2]
            is_valid = True

            for _ in range(3):
                possible_candidates = []
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        candidate = (x, y)
                        if candidate not in selected_points:
                            if all(manhattan(candidate, existing) <= target_dist for existing in selected_points):
                                possible_candidates.append(candidate)

                if not possible_candidates:
                    is_valid = False
                    break
                selected_points.append(possible_candidates[self.rng.integers(len(possible_candidates))])

            if is_valid:
                all_distances = [manhattan(selected_points[i], selected_points[j]) 
                                for i in range(5) for j in range(i+1, 5)]
                if max(all_distances) == target_dist:
                    grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
                    for px, py in selected_points: grid[px, py] = 1
                    return grid
        return None

    def generate_sampleE(self, target_dist, num_points):
        is_odd = (num_points % 2 != 0)
        target_corner = (0, 0) if is_odd else (self.grid_size - 1, self.grid_size - 1)

        max_retries = 100
        for _ in range(max_retries):
            ring_points = get_points_at_exact_distance(target_corner, target_dist)
            if not ring_points:
                return None

            p1 = ring_points[self.rng.integers(len(ring_points))]
            selected_points = [p1]
            is_valid = True

            for _ in range(num_points - 1):
                found_point = False
                for _ in range(100):
                    candX, candY = self.rng.integers(0, self.grid_size), self.rng.integers(0, self.grid_size)
                    candidate = (candX, candY)
                    if candidate not in selected_points and manhattan(candidate, target_corner) >= target_dist:
                        selected_points.append(candidate)
                        found_point = True
                        break
                if not found_point:
                    is_valid = False
                    break

            if is_valid:
                min_corner_dist = min(manhattan(p, target_corner) for p in selected_points)
                if min_corner_dist == target_dist:
                    grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
                    for px, py in selected_points: grid[px, py] = 1
                    return grid
        return None

    def find_practical_limits(self, generator_fn, search_range):
        valid_distances = []
        for d in search_range:
            if generator_fn(d) is not None:
                valid_distances.append(d)
        return min(valid_distances), max(valid_distances)

    def build_balanced_dataset(self, problem_name, generator_fn, min_val, max_val, is_conditional=False):
        total_classes = max_val - min_val + 1

        if is_conditional:
            train_per_class = (((self.min_total_train // total_classes) + 9) // 10) * 10
            test_per_class = (((self.min_total_test // total_classes) + 9) // 10) * 10

            train_per_class = max(10, train_per_class)
            test_per_class = max(10, test_per_class)

            samples_per_class = train_per_class + test_per_class
        else:
            samples_per_class = max(1, (self.min_total_train + self.min_total_test) // total_classes + 1)
            samples_per_class = ((samples_per_class + 4) // 5) * 5 
            train_per_class = int(samples_per_class * 0.8)
            test_per_class = samples_per_class - train_per_class

        x_train, y_train, x_test, y_test = [], [], [], []

        print(f"[{problem_name}] Sınırlar: {min_val}-{max_val} | Sınıf: {total_classes} | Train/Test Sınıf Başına: {train_per_class}/{test_per_class}")

        for dist in range(min_val, max_val + 1):
            class_samples = []
            fail_cnt = 0  

            print(f"  Mesafe {dist:2d} üretiliyor...", end="\r")

            while len(class_samples) < samples_per_class:
                if is_conditional:
                    num_pts = (len(class_samples) % 10) + 1
                    sample = generator_fn(dist, num_pts)
                else:
                    sample = generator_fn(dist)

                if sample is not None:
                    class_samples.append(sample)
                    fail_cnt = 0 
                else:
                    fail_cnt += 1

                if fail_cnt > 1000:
                    print(f"\n[{problem_name}] Uyarı: Mesafe {dist} için yeterli alan yok. Üretim bu mesafe için kesiliyor.")
                    break

            if len(class_samples) == samples_per_class:
                for i, s in enumerate(class_samples):
                    if i < train_per_class:
                        x_train.append(s); y_train.append(dist)
                    else:
                        x_test.append(s); y_test.append(dist)

        print(f"\n[{problem_name}] Veri toplama bitti. Karıştırılıyor...")

        train_indices = self.rng.permutation(len(x_train))
        x_train = np.array(x_train)[train_indices]
        y_train = np.array(y_train)[train_indices]

        test_indices = self.rng.permutation(len(x_test))
        x_test = np.array(x_test)[test_indices]
        y_test = np.array(y_test)[test_indices]

        print(f"[{problem_name}] Bitti -> Train: {len(x_train)}, Test: {len(x_test)}\n")
        return x_train, y_train, x_test, y_test

    def build_dataset_CandD(self):
        print("[Problem C & D] Üretiliyor...")
        total_classes = 10 
        samples_per_class = max(1, (self.min_total_train + self.min_total_test) // total_classes + 1)
        samples_per_class = ((samples_per_class + 4) // 5) * 5 

        train_per_class = int(samples_per_class * 0.8)

        x_trainC, y_trainC, x_testC, y_testC = [], [], [], []
        x_trainD, y_trainD, x_testD, y_testD = [], [], [], []

        for num_points in range(1, total_classes + 1):
            class_samples = []
            while len(class_samples) < samples_per_class:
                grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
                indices = self.rng.choice(self.grid_size * self.grid_size, num_points, replace=False)
                for idx in indices:
                    r, c = divmod(idx, self.grid_size)
                    grid[r, c] = 1
                class_samples.append(grid)

            parity_label = num_points % 2

            for i, s in enumerate(class_samples):
                if i < train_per_class:
                    x_trainC.append(s); y_trainC.append(num_points)
                    x_trainD.append(s); y_trainD.append(parity_label)
                else:
                    x_testC.append(s); y_testC.append(num_points)
                    x_testD.append(s); y_testD.append(parity_label)

        train_indicesC = self.rng.permutation(len(x_trainC))
        x_trainC, y_trainC = np.array(x_trainC)[train_indicesC], np.array(y_trainC)[train_indicesC]
        test_indicesC = self.rng.permutation(len(x_testC))
        x_testC, y_testC = np.array(x_testC)[test_indicesC], np.array(y_testC)[test_indicesC]

        train_indicesD = self.rng.permutation(len(x_trainD))
        x_trainD, y_trainD = np.array(x_trainD)[train_indicesD], np.array(y_trainD)[train_indicesD]
        test_indicesD = self.rng.permutation(len(x_testD))
        x_testD, y_testD = np.array(x_testD)[test_indicesD], np.array(y_testD)[test_indicesD]

        print(f"[{'Problem C'}] Bitti -> Train: {len(x_trainC)}, Test: {len(x_testC)}")
        print(f"[{'Problem D'}] Bitti -> Train: {len(x_trainD)}, Test: {len(x_testD)}\n")

        return (x_trainC, y_trainC, x_testC, y_testC), \
               (x_trainD, y_trainD, x_testD, y_testD)

    def save_datasets_to_disk(self, datasets, folder_name="data"):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Kayıt dizini oluşturuldu: {folder_name}/")

        problem_names = ['A', 'B', 'C', 'D', 'E']

        for name, data in zip(problem_names, datasets):
            xTr, yTr, xTe, yTe = data
            np.save(os.path.join(folder_name, f"x_train_{name}.npy"), xTr)
            np.save(os.path.join(folder_name, f"y_train_{name}.npy"), yTr)
            np.save(os.path.join(folder_name, f"x_test_{name}.npy"), xTe)
            np.save(os.path.join(folder_name, f"y_test_{name}.npy"), yTe)
            print(f"Problem {name} verileri '{folder_name}/' dizinine kaydedildi.")

    def run_pipeline(self):
        print("Sınırlar hesaplanıyor, üretim başlıyor...\n")

        print("Problem A sınırları hesaplanıyor...")
        minA, maxA = self.find_practical_limits(self.generate_sampleA, range(1, 26))
        datasetA = self.build_balanced_dataset("Problem A", self.generate_sampleA, minA, maxA)

        print("Problem B sınırları hesaplanıyor...")
        minB, maxB = self.find_practical_limits(self.generate_sampleB, range(1, 49))
        datasetB = self.build_balanced_dataset("Problem B", self.generate_sampleB, minB, maxB)

        datasetC, datasetD = self.build_dataset_CandD()

        print("Problem E sınırları hesaplanıyor...")
        minE, maxE = self.find_practical_limits(lambda d: self.generate_sampleE(d, 10), range(0, 40))
        datasetE = self.build_balanced_dataset("Problem E", self.generate_sampleE, minE, maxE, is_conditional=True)

        allDatasets = (datasetA, datasetB, datasetC, datasetD, datasetE)
        self.save_datasets_to_disk(allDatasets)

        return allDatasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets with a fixed seed.")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for random number generators.")
    args = parser.parse_args()

    print(f"Using SEED: {args.seed}")

    generator = TargetDrivenDatasetGenerator(seed=args.seed)
    dataA, dataB, dataC, dataD, dataE = generator.run_pipeline()
