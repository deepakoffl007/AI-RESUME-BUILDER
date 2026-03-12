import torch
import torch.nn as nn

class JobModel(nn.Module):

    def __init__(self, vocab_size=30522, embed_dim=128):

        super(JobModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.fc = nn.Linear(embed_dim, 64)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.embedding(x)

        x = x.mean(dim=1)

        x = self.fc(x)

        return self.relu(x)