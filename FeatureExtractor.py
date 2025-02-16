import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TextProcessor:
    def __init__(self, num_dim):
        self.vocabulary = []
        self.vocab_size = len(self.vocabulary)
        self.num_dim = ''
        self.column_poem = num_dim["Poem"].tolist()
        self.column_senti = num_dim["Label"].tolist()

    def preprocessed(self, text):
        preprocessed = text.lower()
        preprocessed = preprocessed.replace('.', ' .')
        preprocessed = preprocessed.replace('!', ' !')
        split_words = preprocessed.split()

        return split_words
    
    def tokens(self):
        all_tokens = []
        tokenized_poems = []
        for poem in self.column_poem:
            tokens = self.preprocessed(poem)
            all_tokens.extend(tokens)
            tokenized_poems.append(tokens)

        print(tokenized_poems)
        
        unique_tokens = sorted(list(set(all_tokens)))
        print("Unique tokens:")
        print(unique_tokens)
        print("Number of unique tokens")
        print(len(unique_tokens))
        self.vocabulary = unique_tokens

        output_data = []

        for j, tokens in enumerate(tokenized_poems):
            # using the encode Function to create Sentence Feature for each poem
            word_to_index = {word: i for i, word in enumerate(tokens)}
            features = np.zeros(len(unique_tokens))
            for token in tokens:
                if token in word_to_index:
                     features[word_to_index[token]] += 1

            output_data.append(features)
        output_csv = "output_tokens.csv"
        column_names = unique_tokens
        output_df = pd.DataFrame(output_data, columns=column_names)

        try:
            output_df.to_csv(output_csv, index=False)
            print(f"Token counts and encoded sentiments saved to '{output_csv}'")
        except Exception as e:
            print(f"Error saving CSV: {e}")

    
    def tfidf(self):
        #TF-IDF Vectorization
        vector = TfidfVectorizer(vocabulary=self.vocabulary)  # use the vocabulary from the tokens generated
        tfidf_matrix = vector.fit_transform(self.column_poem)
        tfidf_names = vector.get_feature_names_out()
        print(tfidf_names) # to check
        tfidf_array = tfidf_matrix.toarray() # matrix to array

        converted_senti = []
        for senti in self.column_senti:
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
        num_tfidf_name = [f"{tfidf_names[i]}" for i in range(len(tfidf_names))]
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
    p = TextProcessor(load_data)
    p.tokens()
    p.tfidf()

except FileNotFoundError:
    print("File not found!")
