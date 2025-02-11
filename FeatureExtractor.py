import numpy as np
import torch
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 

class TextProcessor:
    def __init__(self):
        self.vocabulary = []
        self.vocab_size = 0

    def create_vocabulary(self, text):
        preprocessed = text.lower()
        preprocessed = preprocessed.replace('.', ' .')
        preprocessed = preprocessed.replace('!', ' !')
        split_words = preprocessed.split()
        self.vocabulary = sorted(list(set(split_words)))
        self.vocab_size = len(self.vocabulary)
        return self.vocabulary
    
    def encode(self, tokens):
        self.word_to_index = {word: i for i, word in enumerate(tokens)}
        # sentence feature
        features = np.zeros(self.vocab_size)
        for token in tokens:
                if token in self.word_to_index:
                     features[self.word_to_index[token]] += 1

        return features
    
    def flattening(self, data_poem):
        column = data_poem["Poem"].tolist()
        all_tokens = []
        tokenized_poems = []
        for item in column:
            tokens = self.create_vocabulary(item)

    
    def tfidf(data):
        # initializing the columns into arrays
        poems = data["Poem"].tolist()
        sentiment = data["Label"].tolist()

        #TF-IDF Vectorization
        vector = TfidfVectorizer()
        tfidf_matrix = vector.fit_transform(poems)
        tfidf_array = tfidf_matrix.toarray() # matrix to array

        converted_senti = []
        for senti in sentiment:
            if senti == 0:
                converted_senti.append([0,1])
            elif senti == 1:
                converted_senti.append([1,0])
            else:
                print(f"Unknown Sentiment '{senti}'. Handling as [0,0]")
                converted_senti.append([0,0])
        converted_senti = np.array(converted_senti)

        # Create DataFrame for output
        num_tfidf = tfidf_array.shape[1]
        print(num_tfidf)
        num_tfidf_name = [f"tfidf_{i}" for i in range(num_tfidf)]
        print(num_tfidf_name)

        output_df = pd.DataFrame(tfidf_array, columns=num_tfidf_name)
        output_df[['happy', 'sad']] = converted_senti # Add sentiment columns
        print(output_df)


try:
    fname = "unique_poem_nlp_dataset.csv"
    load_data = pd.read_csv(fname)
    print(load_data)
    TextProcessor.tfidf(load_data)
    TextProcessor.tokens(load_data)

except FileNotFoundError:
    print("File not found!")


""" e_size = int(input("Embedding size: "))
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
 """
    
