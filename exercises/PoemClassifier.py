import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

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

# Printing for checking
f_name = "output_tokens.csv" # unpacking the csv created from Activity 2 in DataFrame, using the TextProcessor created from Activity 1.
load_data = pd.read_csv(f_name) 
print(load_data)
vocab_size = int(load_data.to_numpy().max() + 1) # converting to numpy to get the vocab size, which is the max value + 1.
print(vocab_size)
load_data = load_data.values.tolist() # converting from DataFrame to list of tokens
print(load_data)

# Getting the sample Data
sample1 = load_data[3]
sample2 = load_data[16]

# Answer to no. 3

"""
To enter values for PoemClassifier, the number of embed_dim should be equal to the dimensions of the sample, with the num_classes as 2
since we only have two classifications: happy or sad. They have to be equal because you have to perform matrix multiplcation wherein the inner
dimensions of two matrices must be equal.
"""
model1 = PoemClassifier(vocab_size,len(sample1),2)
model2 = PoemClassifier(vocab_size,len(sample2),2)

"""
Our samples, if we will get it from the dataset, are lists, which means we have to convert them to torch tensors.
Once they are converted, they have to be converted as well to long because the initial datatype is float. Also, converting them
provides a very wide range of indices, which is essential in dealing large tensors.
"""
tensor1 = torch.tensor(sample1).long()
tensor2 = torch.tensor(sample2).long()
print(model1.forward(tensor1))
print(model2.forward(tensor2))


