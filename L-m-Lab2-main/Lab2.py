import nltk
from nltk.corpus import stopwords, product_reviews_1
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import numpy as np

# Завантаження ресурсів
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('product_reviews_1')

# ====================== ПРЕДОБРОБКА ======================
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]

# ====================== ЗАВАНТАЖЕННЯ ДАНИХ ======================
def load_reviews():
    pos = []
    neg = []

    for fileid in product_reviews_1.fileids():
        text = " ".join(product_reviews_1.sents(fileid))
        cleaned = clean_text(text)

        if "pos" in fileid:
            pos.append(cleaned)
        elif "neg" in fileid:
            neg.append(cleaned)

    return pos, neg

pos_data, neg_data = load_reviews()

print("Кількість позитивних відгуків:", len(pos_data))
print("Кількість негативних відгуків:", len(neg_data))

# ====================== ЧАСТОТНА КАРТА ======================
def build_freq_dict(pos_data, neg_data):
    freq_map = defaultdict(lambda: [0, 0])
    for review in pos_data:
        for token in review:
            freq_map[token][0] += 1
    for review in neg_data:
        for token in review:
            freq_map[token][1] += 1
    return freq_map

freq_map = build_freq_dict(pos_data, neg_data)

# ====================== PRIOR LOG ======================
prior_log = np.log(len(pos_data) / len(neg_data))

# ====================== ЛОГ-ПРАВДОПОДОБІЯ ======================
def calc_log_like(freq_map, pos_data, neg_data):
    vocab_size = len(freq_map)
    pos_total = sum([freq[0] for freq in freq_map.values()])
    neg_total = sum([freq[1] for freq in freq_map.values()])

    log_likes = {}
    for token, (pos_count, neg_count) in freq_map.items():
        p_pos = (pos_count + 1) / (pos_total + vocab_size)
        p_neg = (neg_count + 1) / (neg_total + vocab_size)
        log_likes[token] = np.log(p_pos / p_neg)
    return log_likes

log_likes = calc_log_like(freq_map, pos_data, neg_data)

# ====================== ПЕРЕДБАЧЕННЯ ======================
def bayes_predict(text, prior_log, log_likes):
    tokens = clean_text(text)
    score = prior_log
    for token in tokens:
        if token in log_likes:
            score += log_likes[token]
    return "Позитивний" if score > 0 else "Негативний"

# ====================== ТЕСТ НА НЕСКОЛЬКИХ ПРИМЕРАХ ======================
test_data = [
    ("I love this product, it works great!", "Позитивний"),
    ("Terrible quality. I want a refund.", "Негативний"),
    ("Very happy with this purchase.", "Позитивний"),
    ("This is disappointing and useless.", "Негативний")
]

correct = 0
for text, true_label in test_data:
    pred = bayes_predict(text, prior_log, log_likes)
    print(f"Текст: \"{text}\" | Передбачення: {pred} | Правильно: {true_label}")
    if pred == true_label:
        correct += 1

accuracy = correct / len(test_data)
print("\nТочність моделі:", round(accuracy, 2))

# ====================== ТОП ТОКЕНІВ ======================
sorted_tokens = sorted(log_likes.items(), key=lambda x: x[1], reverse=True)

print("\nТОП-5 позитивних токенів:")
for tok, score in sorted_tokens[:5]:
    print(f"{tok}: {score:.2f}")

print("\nТОП-5 негативних токенів:")
for tok, score in sorted_tokens[-5:]:
    print(f"{tok}: {score:.2f}")

# ====================== КАСТОМНИЙ ПРИКЛАД ======================
my_review = "I am excited about the new phone, but the battery life worries me."
print("\nМій текст:", my_review)
print("Передбачення:", bayes_predict(my_review, prior_log, log_likes))
