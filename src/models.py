import torch
import torch.nn as nn

class ProblemACNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Nesneler (x, y) koordinatlarından oluşuyor.
        # Çiftler ise (x_i, y_i, x_j, y_j) şeklinde 4 boyutlu olacak.
        self.g_theta = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Toplam vektör 32 boyutlu çıkacak, f_phi onu alıp tek sayıya (mesafe) düşürecek
        self.f_phi = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        """
        x: (Batch, 25, 25)
        """
        b = x.size(0)
        
        # noktaların koordinatlarını bulma (5, 2)
        objects = torch.nonzero(x == 1).float()[:, 1:].view(b, 5, 2).to(x.device)
        objects = objects / 24.0
        
        # ikili kombinasyonları oluşturma
        o_i = objects.unsqueeze(2).expand(b, 5, 5, 2) # (Batch, 5, 5, 2)
        o_j = objects.unsqueeze(1).expand(b, 5, 5, 2) # (Batch, 5, 5, 2)
        
        # çiftleri birleştir (Batch, 5, 5, 4) -> [x_i, y_i, x_j, y_j]
        pairs = torch.cat([o_i, o_j], dim=-1)
        
        # g_theta ile ilişkileri öğren (Artık sadece 25 çift üzerinden geçiyor!)
        g_out = self.g_theta(pairs) # (Batch, 5, 5, 32)
        
        # min aggregation
        f_in = torch.amin(g_out, dim=(1, 2))
        
        # f_phi ile sonucu üret
        out = self.f_phi(f_in) # (Batch, 1)
        
        return out


def get_model(problem_name):
    if problem_name == 'A':
        return ProblemACNN()
    raise ValueError('Cannot found problem')

