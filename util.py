import os
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(problem_name, folder_name="data"):
    """
    Belirtilen probleme ait veri setini (.npy dosyalarından) belleğe yükler.
    
    Argümanlar:
        problem_name (str): Yüklenecek problem etiketi ('A', 'B', 'C', 'D', 'E').
        folder_name (str): .npy dosyalarının bulunduğu dizin.
        
    Dönüş Değerleri:
        x_train, y_train, x_test, y_test (numpy.ndarray)
    """
    # Girdi standardizasyonu (küçük harf girilirse büyüt)
    problem_name = str(problem_name).upper()
    
    if problem_name not in ['A', 'B', 'C', 'D', 'E']:
        raise ValueError("Geçersiz problem adı! Lütfen 'A', 'B', 'C', 'D' veya 'E' kullanın.")

    x_train_path = os.path.join(folder_name, f"x_train_{problem_name}.npy")
    y_train_path = os.path.join(folder_name, f"y_train_{problem_name}.npy")
    x_test_path  = os.path.join(folder_name, f"x_test_{problem_name}.npy")
    y_test_path  = os.path.join(folder_name, f"y_test_{problem_name}.npy")

    try:
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_test  = np.load(x_test_path)
        y_test  = np.load(y_test_path)
        
        print(f"[+] Problem {problem_name} veri seti başarıyla yüklendi.")
        print(f"    Train -> X: {x_train.shape}, y: {y_train.shape}")
        print(f"    Test  -> X: {x_test.shape}, y: {y_test.shape}")
        
        return x_train, y_train, x_test, y_test
        
    except FileNotFoundError as e:
        print(f"\n[!] HATA: '{folder_name}' dizininde Problem {problem_name} dosyaları bulunamadı.")
        print("[!] Lütfen önce 'TargetDrivenDatasetGenerator' ile verileri ürettiğinizden emin olun.")
        raise e

def plot_mat(matrix):
    h, w = matrix.shape
    
    fig, ax = plt.subplots(figsize=(6, 6))

    # 0,0 sol üst olacak şekilde göster
    ax.imshow(matrix, cmap="Greys", interpolation="none")

    # GRID
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)

    # KOORDİNATLAR
    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    ax.set_xticklabels(np.arange(w))
    ax.set_yticklabels(np.arange(h))

    # okunabilirlik
    ax.tick_params(axis='both', labelsize=8)

    plt.show()