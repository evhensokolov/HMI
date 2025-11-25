import numpy as np
import re
import nltk
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pickle

nltk.download('punkt')

#Завантаження корпусу
def load_corpus(file_path, max_lines=1000):
    lines = []
    with open(file_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            lines.append(line.strip())
            if i >= max_lines:
                break
    return " ".join(lines)

#Токенізація 
def tokenize(corpus):
    corpus = re.sub(r'[,!?;-]+', '.', corpus)
    tokens = nltk.word_tokenize(corpus)
    return [w.lower() for w in tokens if w.isalpha() or w == '.']

def get_dict(data):
    words = sorted(set(data))
    word2Ind = {w: i for i, w in enumerate(words)}
    Ind2word = {i: w for w, i in word2Ind.items()}
    return word2Ind, Ind2word

#Навчальні приклади
def get_windows(words, C):
    for i in range(C, len(words) - C):
        center = words[i]
        context = words[i-C:i] + words[i+1:i+C+1]
        yield context, center

def one_hot(word, word2Ind, V):
    vec = np.zeros(V)
    vec[word2Ind[word]] = 1
    return vec

def context_vector(context_words, word2Ind, V):
    vectors = [one_hot(w, word2Ind, V) for w in context_words]
    return np.mean(vectors, axis=0)

def get_training_data(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        x = context_vector(context_words, word2Ind, V)
        y = one_hot(center_word, word2Ind, V)
        yield x, y

#Ініціалізація CBOW-моделі
def initialize_model(N, V, seed=1):
    np.random.seed(seed)
    W1 = np.random.rand(N, V)
    W2 = np.random.rand(V, N)
    b1 = np.random.rand(N, 1)
    b2 = np.random.rand(V, 1)
    return W1, W2, b1, b2

def relu(h): return np.maximum(0, h)
def softmax(z): return np.exp(z) / np.sum(np.exp(z), axis=0)

def forward_prop(x, W1, W2, b1, b2):
    h = relu(np.dot(W1, x) + b1)
    z = np.dot(W2, h) + b2
    return z, h

def compute_cost(y, yhat):
    return -np.sum(y * np.log(yhat + 1e-9))

def back_prop(x, yhat, y, h, W1, W2, b1, b2):
    z1 = np.dot(W1, x) + b1
    dz2 = yhat - y
    dW2 = np.dot(dz2, h.T)
    db2 = dz2
    dh = np.dot(W2.T, dz2)
    dh[z1 <= 0] = 0
    dW1 = np.dot(dh, x.T)
    db1 = dh
    return dW1, dW2, db1, db2

def train_CBOW(data, word2Ind, V, N, C, alpha, iterations):
    W1, W2, b1, b2 = initialize_model(N, V)
    for i, (x, y) in enumerate(get_training_data(data, C, word2Ind, V)):
        x = x.reshape(V, 1)
        y = y.reshape(V, 1)
        z, h = forward_prop(x, W1, W2, b1, b2)
        yhat = softmax(z)
        cost = compute_cost(y, yhat)
        if i % 100 == 0:
            print(f"[{i}] Loss: {cost:.4f}")
        dW1, dW2, db1, db2 = back_prop(x, yhat, y, h, W1, W2, b1, b2)
        W1 -= alpha * dW1
        W2 -= alpha * dW2
        b1 -= alpha * db1
        b2 -= alpha * db2
        if i >= iterations:
            break
    return W1, W2

def compute_pca(X, n=2):
    X_mean = X - np.mean(X, axis=0)
    cov = np.cov(X_mean, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]
    return np.dot(X_mean, eig_vecs[:, :n])

def visualize(word_embeddings, word2Ind, words):
    valid_words = [w for w in words if w in word2Ind]
    if not valid_words:
        print("Жодного з обраних слів немає у словнику.")
        return
    idx = [word2Ind[w] for w in valid_words]
    X = word_embeddings[idx]
    X_pca = compute_pca(X)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    for i, word in enumerate(valid_words):
        plt.annotate(word, (X_pca[i, 0], X_pca[i, 1]))
    plt.title("PCA Візуалізація")
    plt.grid()
    plt.show()

def save_embeddings(W1, W2, word2Ind, filename="word_embeddings.pkl"):
    embeddings = (W1.T + W2) / 2.0
    with open(filename, "wb") as f:
        pickle.dump((embeddings, word2Ind), f)
    print(f"✅ Векторні подання збережено до {filename}")

# === MAIN ===
if __name__ == "__main__":
    print("Завантаження корпусу...")
    raw = load_corpus("wiki_dump.tokenized.txt", max_lines=1000)
    words = tokenize(raw)
    word2Ind, Ind2word = get_dict(words)
    V = len(word2Ind)
    print(f"Кількість слів: {len(words)}, розмір словника: {V}")

    N = 50
    C = 2
    alpha = 0.05
    iterations = 2000

    print("Навчання CBOW...")
    W1, W2 = train_CBOW(words, word2Ind, V, N, C, alpha, iterations)
    embeddings = (W1.T + W2) / 2.0

    visualize_words = ['the', 'and', 'man', 'woman', 'world', 'life', 'god', 'king', 'city']
    visualize(embeddings, word2Ind, visualize_words)

    save_embeddings(W1, W2, word2Ind)