# Імпортування бібліотек
import numpy as np
import pandas as pd
import requests
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import re
import string
import nltk

nltk.download("twitter_samples")
nltk.download("stopwords")

# --- Завантаження векторів слів ---
print("Завантаження векторів слів...")
lang1_embeddings = KeyedVectors.load_word2vec_format("cc.en.300.vec", binary=False)
lang2_embeddings = KeyedVectors.load_word2vec_format("cc.en.300.vec", binary=False)

# --- Завантаження словника перекладів ---
def load_dict_from_url(url, max_words=100):
    print(f"Завантаження перших {max_words} слів зі словника з {url}...")
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    pairs = [line.split()[:2] for line in lines if len(line.split()) >= 2]
    return dict(pairs[:max_words])


# Завантаження словника з посилання
dictionary_url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-en.txt"
translation_dict = load_dict_from_url(dictionary_url)
print(f"Розмір словника перекладів: {len(translation_dict)}")

# --- Побудова матриць X та Y ---
def get_matrices(word_dict, src_embeddings, tgt_embeddings):
    X, Y = [], []
    for src_word, tgt_word in word_dict.items():
        if src_word in src_embeddings and tgt_word in tgt_embeddings:
            X.append(src_embeddings[src_word])
            Y.append(tgt_embeddings[tgt_word])
    return np.array(X), np.array(Y)

X_train, Y_train = get_matrices(translation_dict, lang1_embeddings, lang2_embeddings)

# --- Обчислення матриці перетворення R ---
def compute_loss(X, Y, R):
    diff = np.dot(X, R) - Y
    return np.sum(diff**2) / len(X)

def compute_gradient(X, Y, R):
    diff = np.dot(X, R) - Y
    return 2 * np.dot(X.T, diff) / len(X)

def align_embeddings(X, Y, steps=100, lr=0.001):
    R = np.random.rand(X.shape[1], Y.shape[1])
    for i in range(steps):
        R -= lr * compute_gradient(X, Y, R)
        if i % 10 == 0:
            print(f"Ітерація {i}, втрата: {compute_loss(X, Y, R):.4f}")
    return R

R = align_embeddings(X_train, Y_train, steps=100, lr=0.05)

# --- Функція перекладу ---
def translate(word, R, src_embeddings, tgt_embeddings):
    if word not in src_embeddings:
        return "Слово відсутнє у словнику"
    transformed_vector = np.dot(src_embeddings[word], R)
    similarities = cosine_similarity(
        transformed_vector.reshape(1, -1), tgt_embeddings.vectors
    )
    closest_idx = similarities.argmax()
    return tgt_embeddings.index_to_key[closest_idx]

# --- Оцінка точності ---
def evaluate(word_dict, R, src_embeddings, tgt_embeddings):
    correct = 0
    for src_word, tgt_word in word_dict.items():
        if src_word in src_embeddings and tgt_word in tgt_embeddings:
            predicted_word = translate(src_word, R, src_embeddings, tgt_embeddings)
            if predicted_word == tgt_word:
                correct += 1
    return correct / len(word_dict)

accuracy = evaluate(translation_dict, R, lang1_embeddings, lang2_embeddings)
print(f"Точність перекладу: {accuracy:.2%}")

# --- Завантаження твітів ---
all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")
all_tweets = all_positive_tweets + all_negative_tweets

# --- Перетворення твітів на вектори ---
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_eng = stopwords.words("english")
    tweet = re.sub(r"https?://\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tokens = TweetTokenizer().tokenize(tweet.lower())
    return [stemmer.stem(w) for w in tokens if w not in stopwords_eng and w not in string.punctuation]

def get_tweet_embedding(tweet, embeddings):
    words = process_tweet(tweet)
    vectors = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

tweet_embeddings = {
    i: get_tweet_embedding(tweet, lang1_embeddings) for i, tweet in enumerate(all_tweets)
}

# --- Локально-сенситивне хешування (LSH) ---
def hash_vector(vector, planes):
    return tuple((np.dot(vector, planes.T) > 0).astype(int))

def create_hash_table(embeddings, planes):
    table = {}
    for idx, vector in embeddings.items():
        hash_code = hash_vector(vector, planes)
        table.setdefault(hash_code, []).append(idx)
    return table

n_planes = 10
random_planes = np.random.randn(n_planes, 300)
hash_table = create_hash_table(tweet_embeddings, random_planes)

# --- Пошук найближчих сусідів ---
def search_similar_tweets(tweet_id, embedding, hash_table, random_planes, k=5):
    hash_code = hash_vector(embedding, random_planes)
    candidates = hash_table.get(hash_code, [])
    distances = cosine_similarity(
        embedding.reshape(1, -1), [tweet_embeddings[i] for i in candidates]
    ).flatten()
    top_k_idx = np.argsort(distances)[-k:][::-1]
    return [candidates[i] for i in top_k_idx]

# --- Тестування пошуку ---
tweet_id = 0  # Приклад ID
query_tweet = all_tweets[tweet_id]
print(f"Запит: {query_tweet}")

query_embedding = tweet_embeddings[tweet_id]
similar_tweets = search_similar_tweets(tweet_id, query_embedding, hash_table, random_planes, k=3)

print("Найбільш схожі твіти:")
for idx in similar_tweets:
    print(all_tweets[idx])
