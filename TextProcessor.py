import numpy as np
import torch
import os

class TextProcessor:
    def __init__(self, dir_text, num_dim):
        self.dir_text = dir_text
        self.num_dim = num_dim
        self.vocabulary = []
        self.vocab_size = 0
        self.appendtext = ''
        self.word_to_index = []

    def create_vocabulary(self, text):
        preprocessed = text.lower()
        preprocessed = preprocessed.replace('.', ' .')
        preprocessed = preprocessed.replace('!', ' !')
        split_words = preprocessed.split()
        self.vocabulary = sorted(list(set(split_words)))
        self.vocab_size = len(self.vocabulary)
        return self.vocabulary
    
    def encode(self, tokens):
        features = np.zeros(len(tokens))
        for i, token in enumerate(tokens):
                if token in self.vocabulary:
                     features[i] += self.vocabulary.index(token)

        return features

    def decode(self, indices):
        tokens = []
        for i in range(self.vocab_size):
            tokens.append(self.vocabulary[i])
        return tokens


    def get_embedding_features(self, tokens):
        embedding = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.num_dim)
        return embedding
    
    def readText(self):
        for file in os.listdir(): 
        # Check whether file is in text format or not 
            if file.endswith(".txt"): 
                file_path = f"{self.dir_text}/{file}"
                with open(file_path, 'r') as f: 
                    self.appendtext += f.read()
                    self.appendtext += ' '

        return self.appendtext
        
e_size = int(input("Embedding size: "))
dir = input("Input directory: ")
os.chdir(dir)
tp = TextProcessor(dir, e_size)
text = tp.readText()
vocab = tp.create_vocabulary(text)
indices = tp.encode(vocab)
print(vocab)
print("Encode")
print(indices)
print("Decode")
print(tp.decode(indices))
print(tp.get_embedding_features(vocab))

    