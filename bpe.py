# 1 - Initial tokenization
from collections import Counter

def get_pairs(tokens):
    pairs = Counter()
    for word in tokens:
        for i in range(len(word) - 1):
            pairs[word[i], word[i + 1]] += 1
    return pairs

tokens = [['h', 'e', 'l', 'l', 'o'], ['h', 'e', 'l', 'l']]
print(get_pairs(tokens))

# 2 - Merging most frequent pair
def merge_tokens(tokens, pair):
    new_tokens = []
    for word in tokens:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_tokens.append(new_word)
    return new_tokens

pair_to_merge = ('l', 'l')
#tokens = merge_tokens(tokens, pair_to_merge)
#print(tokens)

# Full implementation
def byte_pair_encoding(corpus, vocab_size):
    tokens = [[char for char in word] for word in corpus]
    print(tokens)
    while len(set(sum(tokens, []))) < vocab_size:
        pairs = get_pairs(tokens)
        print("Pairs:")
        print(pairs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        tokens = merge_tokens(tokens, best_pair)
    return tokens

corpus = ["hello", "hell"]
corpus = ["apple", "apples", "ape", "apex"]

corpus = [
    "natural", "nature", "nation", "national", "navigate", "narrative",
    "artificial", "artist", "artistry", "artifact", "arbitrary"
]

vocab_size = 20
tokenized_words = byte_pair_encoding(corpus, vocab_size)
vocabulary = sorted(set(token for word in tokenized_words for token in word))
print("Vocabulary:")
print(vocabulary)
