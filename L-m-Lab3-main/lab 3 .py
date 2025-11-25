import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

url_families = "https://raw.githubusercontent.com/lang-uk/vecs/refs/heads/master/test/test_vocabulary.txt"
data = pd.read_csv(url_families, delimiter='\t', header=None, skiprows=1, 
                   names=['word1', 'relation1', 'word2', 'relation2'])


family_keywords = [
    "хлопець", "дівчина", "брат", "сестра", "тато", "мама", "батько", "мати",
    "дідусь", "бабуся", "дід", "баба", "внук", "внучка", "наречений", "наречена",
    "чоловік", "дружина", "племінник", "племінниця", "принц", "принцеса",
    "син", "дочка", "вітчим", "мачуха", "пасинок", "падчерка", "дядько", "тітка",
    "хлопчик", "дівчинка"
]

family_data = data[data['word1'].isin(family_keywords)]
print("Перші 20 рядків набору даних (Сімейні відношення):")
print(family_data.head(20))

model = KeyedVectors.load_word2vec_format("ubercorpus.cased.tokenized.word2vec.300d", binary=False)
print("\nМодель успішно завантажена!")

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def find_relation(word1, relation1, word2, word_embeddings, cosine_similarity=cosine_similarity):
    group = {word1, relation1, word2}

    for word in group:
        if word not in word_embeddings:
            print(f"Слово '{word}' відсутнє у моделі.")
            return "Невідомо", -1

    vec = word_embeddings[word2] - word_embeddings[word1] + word_embeddings[relation1]
    similarity = -1
    best_match = ''

    for word in word_embeddings.index_to_key:
        if word not in group:
            cur_similarity = cosine_similarity(vec, word_embeddings[word])
            if cur_similarity > similarity:
                similarity = cur_similarity
                best_match = word

    return best_match, similarity

def get_accuracy(word_embeddings, data, find_relation_func=find_relation):
    num_correct = 0
    total = 0

    for i, row in data.iterrows():
        word1 = row['word1']
        relation1 = row['relation1']
        word2 = row['word2']
        relation2 = row['relation2']

        if any(word not in word_embeddings for word in [word1, relation1, word2, relation2]):
            print(f"Пропускаємо рядок: {row.values}")
            continue

        predicted_word, _ = find_relation_func(word1, relation1, word2, word_embeddings)
        if predicted_word == relation2:
            num_correct += 1
        total += 1

    accuracy = num_correct / total if total > 0 else 0
    return accuracy

accuracy = get_accuracy(model, family_data)
print(f"\nТочність на сімейних відношеннях: {accuracy:.2%}")

def visualize(words, word_embeddings):
    valid_words = [w for w in words if w in word_embeddings.key_to_index]
    word_vectors = np.array([word_embeddings[w] for w in valid_words])

    pca = PCA(n_components=2)
    components = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(components[:, 0], components[:, 1], color='blue')

    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(components[i, 0], components[i, 1]), fontsize=12, color='red')

    plt.title("Візуалізація сімейних відношень (PCA)")
    plt.show()
    
words = ["хлопець", "дівчина", "брат", "сестра", "тато", "мама", "дідусь", "бабуся", "син", "дочка"]
visualize(words, model)
