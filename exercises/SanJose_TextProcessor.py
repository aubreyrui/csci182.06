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
        self.vocabulary = sorted(list(set(text)))
        self.vocab_size = len(self.vocabulary)
        return self.vocabulary
    
    def encode(self, tokens):
        features = np.zeros(len(tokens))
        for i, token in enumerate(tokens):
                if token in self.vocabulary:
                     features[i] += self.vocabulary.index(token)

        return features

    def decode(self, in00dices):
        tokens = []
        for i in indices:
                tokens.append(self.vocabulary[int(i)])
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
    
    def preprocessed(self,text):
        preprocessed = text.lower()
        preprocessed = preprocessed.replace('.', ' <period>')
        preprocessed = preprocessed.replace('!', ' <exclamation>')
        split_words = preprocessed.split()

        return split_words
        
e_size = int(input("Embedding size: "))
dir = input("Input directory: ")
os.chdir(dir)
tp = TextProcessor(dir, e_size)
text = tp.readText()
processed_vocab = tp.preprocessed(text)
vocab = tp.create_vocabulary(processed_vocab)
print("Vocabulary:")
print(vocab)

sample_text = "Hello World!"
print("To encode: " + sample_text)
prep = tp.preprocessed(sample_text)
indices = tp.encode(prep)
print("Encode")
print(indices)
print("Decode")
print(tp.decode(indices))
print(tp.get_embedding_features(prep))

    