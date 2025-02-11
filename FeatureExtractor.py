import numpy as np
import torch
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

class TextProcessor:
    def __init__(self):
        self.vocabulary = []
        self.vocab_size = 0

    def preprocessed(self, text):
        preprocessed = text.lower()
        preprocessed = preprocessed.replace('.', ' .')
        preprocessed = preprocessed.replace('!', ' !')
        split_words = preprocessed.split()

        return split_words
    
    # this should be applied
    def encode(tokens):
        word_to_index = {word: i for i, word in enumerate(tokens)}
        size = len(token)
        # sentence feature
        features = np.zeros(size)
        for token in tokens:
                if token in word_to_index:
                     features[word_to_index[token]] += 1

        return features
    
    def tokens(self, data_poem):
        column_poem = data_poem["Poem"].tolist()
        column_senti = data_poem["Label"].tolist()
        all_tokens = []
        tokenized_poems = []
        for poem in column_poem:
            tokens = self.preprocessed(poem)
            all_tokens.extend(tokens)
            tokenized_poems.append(tokens)

        print(tokenized_poems)
        
        unique_tokens = sorted(list(set(all_tokens)))
        print("Unique tokens:")
        print(unique_tokens)

        output_data = []

        for j, tokens in enumerate(tokenized_poems):
            # counts occurence of a token in a poem
            token_counter = Counter(tokens) # this is more on advanced version honestly
            row = [token_counter[tokens] for tokens in unique_tokens]

            if column_senti[j] == 0:
                row.extend([0, 1])
            elif column_senti[j] == 1:
                row.extend([1, 0])
            else:
                print(f"Unknown sentiment: '{column_senti[j]}. Outputing as [0,0]")
                row.extend([0,0])

            output_data.append(row)
        output_csv = "output_tokens.csv"
        column_names = unique_tokens + ['happy', 'sad']
        output_df = pd.DataFrame(output_data, columns=column_names)

        try:
            output_df.to_csv(output_csv, index=False)
            print(f"Token counts and encoded sentiments saved to '{output_csv}'")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    
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

        output_path = 'output_tfidf.csv'

        output_df.to_csv(output_path, index=False)
        print(f"TF-IDF features and encoded sentiments saved to '{output_path}'")

        print(output_df)


try:
    fname = "unique_poem_nlp_dataset.csv"
    load_data = pd.read_csv(fname)
    print(load_data)
    p = TextProcessor()
    TextProcessor.tfidf(load_data)
    p.tokens(load_data)

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
    
