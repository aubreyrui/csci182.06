import torch as nn

class SimpleEmbeddingModelForNextWord(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_layer_1 = nn.Linear(embedding_dim, 5)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

class SimpleEmbeddingModelForTranslation(nn.Module):
     def __init__(self, vocab_size, embedding_dim):
        super(SimpleEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_layer_1 = nn.Linear(embedding_dim, 5)
        self.output_layer = nn.Linear(5,5)

class SimpleEmbeddingModelForSentimentAnalysis(nn.Module):
     def __init__(self, vocab_size, embedding_dim):
        super(SimpleEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear_layer_1 = nn.Linear(embedding_dim, 5)
        self.output_layer = nn.Linear(5,5)

# if your want your model to predict the model, it should be the same as the size of the vocabulary
# Activity 1: x is a tensor representing the array. Output ng encode for activity 1
# Instead of encodes, this gets fed to the        
# expects the 
# f2f in monday. Train the model to see if it gives something
