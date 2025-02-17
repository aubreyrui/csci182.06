import torch
import torch.nn as nn
import torch.nn.functional as F

class PoemClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(PoemClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len) tensor of token indices
        """
        embeds = self.embedding(x)  # Shape: (batch_size, seq_len, embed_dim)
        pooled = embeds.mean(dim=1) # Mean pooling over sequence length
        result = self.fc(pooled)    # Shape: (batch_size, num_classes)
        return result
