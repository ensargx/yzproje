import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class SelfAttention(nn.Module):
    def __init__(self, embedDim):
        super().__init__()
        self.embedDim = embedDim
        self.query = nn.Linear(embedDim, embedDim)
        self.key = nn.Linear(embedDim, embedDim)
        self.value = nn.Linear(embedDim, embedDim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embedDim)
        attnWeights = F.softmax(scores, dim=-1)
        out = torch.matmul(attnWeights, V)

        return out


class ProblemRelationCNN(nn.Module):
    def __init__(self, aggregation="min", include_self_pairs=True):
        super().__init__()
        self.aggregation = aggregation
        self.include_self_pairs = include_self_pairs

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

        # Toplam vektör 32 boyutlu çıkacak, f_phi onu alıp tek sayıya düşürecek
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
        o_i = objects.unsqueeze(2).expand(b, 5, 5, 2)
        o_j = objects.unsqueeze(1).expand(b, 5, 5, 2)

        # çiftleri birleştir (Batch, 5, 5, 4) -> [x_i, y_i, x_j, y_j]
        pairs = torch.cat([o_i, o_j], dim=-1)

        # g_theta ile ilişkileri öğren
        g_out = self.g_theta(pairs)

        if not self.include_self_pairs:
            mask = torch.eye(5, dtype=torch.bool, device=x.device).view(1, 5, 5, 1)
            if self.aggregation == "max":
                g_out = g_out.masked_fill(mask, float('-inf'))
            else:
                g_out = g_out.masked_fill(mask, float('inf'))

        if self.aggregation == "max":
            f_in = torch.amax(g_out, dim=(1, 2))
        else:
            f_in = torch.amin(g_out, dim=(1, 2))

        out = self.f_phi(f_in)

        return out


class ProblemACNN(ProblemRelationCNN):
    def __init__(self):
        super().__init__(aggregation="min", include_self_pairs=False)


class ProblemBCNN(ProblemRelationCNN):
    def __init__(self):
        super().__init__(aggregation="max")


class ProblemCCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x: (Batch, 25, 25)
        """
        x = x.unsqueeze(1)

        out = self.conv_layers(x)
        out = self.fc_layers(out)

        return out


class ProblemDCNN(nn.Module):
    def __init__(self, embedDim=16, hiddenDim=64):
        super().__init__()
        self.pixelEmbedding = nn.Linear(1, embedDim)

        self.attention = SelfAttention(embedDim)

        self.mlp = nn.Sequential(
            nn.Linear(embedDim, hiddenDim),
            Sine(),
            nn.Linear(hiddenDim, hiddenDim),
            Sine(),
            nn.Linear(hiddenDim, 1)
        )

    def forward(self, x):
        batchSize = x.size(0)

        x = x.view(batchSize, 625, 1)

        emb = self.pixelEmbedding(x)
        emb = emb * x

        attnOut = self.attention(emb)

        pooled = torch.sum(attnOut * x, dim=1)

        out = self.mlp(pooled)
        if self.training:
            return out
        else:
            return (out > 0.0).float()


class ProblemECNN(nn.Module):
    def __init__(self, embedDim=32, hiddenDim=64):
        super().__init__()
        
        self.pixelEmb = nn.Linear(3, embedDim)
        
        self.parityEncoder = nn.Sequential(
            nn.Linear(1, hiddenDim),
            Sine(),
            nn.Linear(hiddenDim, hiddenDim),
            Sine(),
            nn.Linear(hiddenDim, embedDim)
        )
        
        self.attention = SelfAttention(embedDim)
        
        self.distanceHead = nn.Sequential(
            nn.Linear(embedDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, 1)
        )
        
        yCoords, xCoords = torch.meshgrid(torch.arange(25), torch.arange(25), indexing='ij')
        coords = torch.stack([xCoords.flatten(), yCoords.flatten()], dim=-1).float()
        self.register_buffer('coordsGrid', coords)

    def forward(self, x):
        batchSize = x.size(0)

        xFlat = x.view(batchSize, 625, 1)
        batchCoords = self.coordsGrid.unsqueeze(0).expand(batchSize, -1, -1)
        xWithCoords = torch.cat([xFlat, batchCoords], dim=-1)
        
        countFeature = torch.sum(xFlat, dim=1) 
        parityVector = self.parityEncoder(countFeature).unsqueeze(1)
        
        pixelFeatures = self.pixelEmb(xWithCoords)
        
        combinedFeatures = pixelFeatures + parityVector
        
        combinedFeatures = combinedFeatures * xFlat
        
        attnOut = self.attention(combinedFeatures)
        
        attnOutMasked = attnOut.masked_fill(xFlat == 0, float('-inf'))
        pooledMax = torch.max(attnOutMasked, dim=1)[0]
        
        final_out = self.distanceHead(pooledMax).squeeze(-1)
        
        if self.training:
            return final_out
        else:
            return torch.round(final_out)


def get_model(problem_name):
    problem_name = str(problem_name).upper()

    if problem_name == 'A':
        return ProblemACNN()
    if problem_name == 'B':
        return ProblemBCNN()
    if problem_name == 'C':
        return ProblemCCNN()
    if problem_name == 'D':
        return ProblemDCNN()
    if problem_name == 'E':
        return ProblemECNN()

    raise ValueError('Cannot found problem')
