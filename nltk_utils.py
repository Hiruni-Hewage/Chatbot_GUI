import torch
import numpy as np
import nltk
# Uncomment the line below if you haven't already downloaded the 'punkt' tokenizer models
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    A token can be a word or punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stemming = find the root form of the word.
    Examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Example usage
a = "How are you?"
print("Original sentence:", a)
a_tokenized = tokenize(a)
print("Tokenized sentence:", a_tokenized)
a_stemmed = [stem(word) for word in a_tokenized]
print("Stemmed words:", a_stemmed)

words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag = bag_of_words(a_tokenized, words)
print("Bag of words:", bag)
