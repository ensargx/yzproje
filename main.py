from pathlib import Path
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from collections import Counter
from src.models import get_model
from util import create_matrix_figure, load_dataset, plot_mat

GRID_SIZE = 25
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

torch.manual_seed(11337)

problems = ['A', 'B', 'C', 'D', 'E']

######

problem_name = problems[0]
x_train, y_train, x_test, y_test = load_dataset(problem_name)

x_train_tensor = torch.tensor(x_train).float()
y_train_tensor = torch.tensor(y_train).float()
x_test_tensor = torch.tensor(x_test).float()
y_test_tensor = torch.tensor(y_test).float()

# 2. Artık Tensör oldukları için TensorDataset sorunsuz çalışacaktır
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

# 2. Model, Loss ve Optimizatör
model = get_model(problem_name).to(device)
model.train()

criterion = nn.MSELoss() 
optimizer = optim.AdamW(model.parameters())

log_path = Path(f"results/{problem_name}/loss_history.csv")
base_dir = os.path.dirname(log_path)
if base_dir:
    os.makedirs(base_dir, exist_ok=True)

history = []

epochs = 10000


#######


for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        predictions = model(batch_x)

        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    test_mae = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float().view(-1, 1)
            predictions = model(batch_x)
            test_mae += torch.mean(torch.abs(predictions - batch_y)).item()
    avg_test_mae = test_mae / len(test_loader)

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "test_mae": avg_test_mae,
        "test_mae": avg_test_mae,
    })

    # Her epoch sonunda dosyaya kaydet
    pd.DataFrame(history).to_csv(log_path, index=False)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Error: {avg_test_mae:.4f}")


#######

def predict(model, x_data):
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

    sonuc_tensor = torch.round(sonuc).int()

    if is_single:
        return sonuc_tensor.item()
    else:
        return sonuc_tensor.flatten()


#######

wrong_indices = []
error_diffs = []

with torch.no_grad():
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test, dtype=torch.float32)
    x_test = x_test.to(device)
    preds = predict(model, x_test)

for i, (pred, y) in enumerate(zip(preds, y_test, strict=True)):
    if pred != y:
        # Hatanın büyüklüğünü (mutlak farkını) hesapla
        diff = abs(int(pred) - int(y)) 

        wrong_indices.append((i, y, pred))
        error_diffs.append(diff)

Y_EKSENI_MAKSIMUM = 100 

diff_map = dict(Counter(error_diffs))
print("\nHata Farkı Haritası (Fark: Adet):", diff_map)

if error_diffs:
    sorted_diffs = sorted(diff_map.keys())
    counts = [diff_map[d] for d in sorted_diffs]
    x_labels = [str(d) for d in sorted_diffs]

    fig_width = max(8, len(sorted_diffs) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    bars = ax.bar(x_labels, counts, color='#FF6F61', edgecolor='#CC3333', width=0.4, zorder=3)

    if len(sorted_diffs) < 6:
        ax.set_xlim(-0.5, 5.5)

    max_count = max(counts) if counts else 1
    
    # Her zaman 0'dan başlar. Tavan değeri ise belirlediğiniz limit (100) ile grafikteki en yüksek bar arasında seçilir.
    ax.set_ylim(0, max(Y_EKSENI_MAKSIMUM, max_count + (max_count * 0.1)))

    for bar in bars:
        yval = bar.get_height()
        if yval > 0: 
            # Yazıları yazdırırken üst kısımdaki boşluk (tavan yüksekliğine göre ayarlandı)
            ax.text(bar.get_x() + bar.get_width()/2, yval + (ax.get_ylim()[1] * 0.015), 
                    int(yval), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Yanlış Tahminlerdeki Hata Büyüklüğü Dağılımı', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Mutlak Hata Farkı (|Tahmin - Gerçek|)', fontsize=12, labelpad=10)
    ax.set_ylabel('Hata Adedi', fontsize=12, labelpad=10)
    
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    save_path = f"results/{problem_name}/error_distribution.png"

    base_dir = os.path.dirname(save_path)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


######


# Gösterilecek maksimum hatalı örnek sayısı
gosterilecek_adet = 5

save_dir = f"results/{problem_name}/wrong_predictions"

os.makedirs(save_dir, exist_ok=True)

print(f"\n--- İlk {gosterilecek_adet} Hatalı Tahmin Görselleştiriliyor ---")

for count, (idx, true_y, pred_y) in enumerate(wrong_indices):
    if count >= gosterilecek_adet:
        break # Sınıra ulaşınca döngüyü bitir
        
    fark = abs(int(pred_y) - int(true_y))
    print(f"\nİndeks: {idx} | Gerçek Değer: {true_y} | Tahmin: {pred_y} | Hata Payı: {fark}")
    
    # x_test GPU'da (cuda) olduğu için matplotlib sorun çıkarabilir.
    # plot_mat fonksiyonunuzun beklentisine göre x_test'i CPU'ya alıp numpy'a çeviriyoruz.
    # Eğer plot_mat doğrudan PyTorch tensörlerini destekliyorsa sadece x_test[idx] de yazabilirsiniz.
    matris = x_test[idx].cpu().numpy() if isinstance(x_test[idx], torch.Tensor) else x_test[idx]
    
    fig = create_matrix_figure(matris)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

