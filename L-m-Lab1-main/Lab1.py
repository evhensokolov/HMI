import nltk
import string
from nltk.corpus import stopwords, product_reviews_1
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Завантаження ресурсів
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('product_reviews_1')

# ===================== Предобработка текста =====================
def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    preprocessed_texts = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        preprocessed_texts.append(" ".join(stemmed_tokens))
    return preprocessed_texts

# ===================== Загрузка отзывов =====================
def load_product_reviews():
    pos_texts = [
        " ".join(product_reviews_1.sents(fileid))
        for fileid in product_reviews_1.fileids()
        if 'pos' in fileid
    ]

    neg_texts = [
        " ".join(product_reviews_1.sents(fileid))
        for fileid in product_reviews_1.fileids()
        if 'neg' in fileid
    ]

    pos_texts = preprocess_text(pos_texts)
    neg_texts = preprocess_text(neg_texts)
    return pos_texts, neg_texts

# ===================== Основной код =====================
pos_texts, neg_texts = load_product_reviews()

# Частота слов
positive_freq = Counter(word_tokenize(" ".join(pos_texts)))
negative_freq = Counter(word_tokenize(" ".join(neg_texts)))
print("ТОП-5 позитивних слів:", positive_freq.most_common(5))
print("ТОП-5 негативних слів:", negative_freq.most_common(5))

# Метки
texts = pos_texts + neg_texts
labels = [1]*len(pos_texts) + [0]*len(neg_texts)

# Векторизация
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение базовой модели
model = LogisticRegression(max_iter=300, solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n=== Базовая модель ===")
print("Точність моделі:", accuracy_score(y_test, y_pred))
print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

# ===================== Исследование гиперпараметров =====================
def evaluate_hyperparameters(X_train, y_train, X_test, y_test):
    print("\n=== Дослідження гіперпараметрів ===\n")

    # Параметр регуляризации C
    for C in np.logspace(-3, 3, 7):
        model = LogisticRegression(C=C, max_iter=300, solver='liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\nПараметр C={C}")
        print("Точність:", accuracy_score(y_test, y_pred))
        print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

    # Количество итераций
    for max_iter in [100, 200, 300, 500]:
        model = LogisticRegression(C=1.0, max_iter=max_iter, solver='liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\nКількість ітерацій: {max_iter}")
        print("Точність:", accuracy_score(y_test, y_pred))
        print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

    # Методы solver
    for solver in ['liblinear', 'saga', 'lbfgs']:
        try:
            model = LogisticRegression(C=1.0, max_iter=300, solver=solver)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"\nМетод solver: {solver}")
            print("Точність:", accuracy_score(y_test, y_pred))
            print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))
        except Exception as e:
            print(f"Solver {solver} не поддерживается: {e}")

    # Балансировка классов
    model = LogisticRegression(C=1.0, max_iter=300, solver='liblinear', class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nТочність з балансуванням класів:", accuracy_score(y_test, y_pred))
    print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

evaluate_hyperparameters(X_train, y_train, X_test, y_test)
