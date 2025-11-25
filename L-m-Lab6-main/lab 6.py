import nltk
from nltk.corpus import sinica_treebank
from collections import defaultdict, Counter
import random

# Завантаження корпусу
nltk.download('sinica_treebank')

# ================================================
# 1. Зчитування корпусу Sinica Treebank
# ================================================
# Корпус уже розмічений: кожне речення містить пари (слово, тег)
tagged_sents = list(sinica_treebank.tagged_sents())

# Для стабільності результатів
random.seed(42)
random.shuffle(tagged_sents)

# Розділення на train/test (80/20)
split_point = int(0.8 * len(tagged_sents))
train_sents = tagged_sents[:split_point]
test_sents = tagged_sents[split_point:]

# ================================================
# 2. Побудова частот переходів і емісій
# ================================================
transition_counts = defaultdict(Counter)
emission_counts = defaultdict(Counter)
tag_counts = Counter()

for sentence in train_sents:
    prev_tag = '<s>'
    for word, tag in sentence:
        # У китайській не зводимо до lower()
        transition_counts[prev_tag][tag] += 1
        emission_counts[tag][word] += 1
        tag_counts[tag] += 1
        prev_tag = tag
    transition_counts[prev_tag]['</s>'] += 1

# ================================================
# 3. Обчислення ймовірностей переходів (A) і емісій (B)
# ================================================
A = defaultdict(dict)
for prev_tag in transition_counts:
    total = sum(transition_counts[prev_tag].values())
    for tag in transition_counts[prev_tag]:
        A[prev_tag][tag] = transition_counts[prev_tag][tag] / total

B = defaultdict(dict)
for tag in emission_counts:
    total = sum(emission_counts[tag].values())
    for word in emission_counts[tag]:
        B[tag][word] = emission_counts[tag][word] / total

# ================================================
# 4. Алгоритм Вітербі
# ================================================
def viterbi(sentence, A, B, all_tags):
    V = [{}]
    path = {}

    for tag in all_tags:
        trans_p = A['<s>'].get(tag, 1e-6)
        emis_p = B[tag].get(sentence[0], 1e-6)
        V[0][tag] = trans_p * emis_p
        path[tag] = [tag]

    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}

        for curr_tag in all_tags:
            best = []
            emis_p = B[curr_tag].get(sentence[t], 1e-6)
            for prev_tag in all_tags:
                if V[t - 1].get(prev_tag, 0) > 0:
                    trans_p = A[prev_tag].get(curr_tag, 1e-6)
                    prob = V[t - 1][prev_tag] * trans_p * emis_p
                    best.append((prob, prev_tag))
            if best:
                prob, prev_tag = max(best)
                V[t][curr_tag] = prob
                new_path[curr_tag] = path[prev_tag] + [curr_tag]

        if not new_path:
            most_common_tag = max(tag_counts, key=tag_counts.get)
            for tag in all_tags:
                V[t][tag] = 1e-6
                new_path[tag] = path.get(tag, []) + [most_common_tag]

        path = new_path

    n = len(sentence) - 1
    (prob, final_tag) = max((V[n].get(tag, 0), tag) for tag in all_tags)
    return path[final_tag]

# ================================================
# 5. Оцінка точності моделі
# ================================================
all_tags = list(tag_counts.keys())

def evaluate(test_sents, A, B, all_tags):
    total = 0
    correct = 0
    for sentence in test_sents:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]
        predicted_tags = viterbi(words, A, B, all_tags)
        for p, t in zip(predicted_tags, true_tags):
            if p == t:
                correct += 1
            total += 1
    return correct / total

accuracy = evaluate(test_sents, A, B, all_tags)
print(f"Accuracy of HMM model (Sinica Treebank): {accuracy:.4f}")

# ================================================
# 6. Порівняння з базовим POS-теггером NLTK
# ================================================
def nltk_pos_accuracy(test_sents):
    total = 0
    correct = 0
    for sentence in test_sents:
        words = [word for word, tag in sentence]
        true_tags = [tag for word, tag in sentence]
        # POS-теггер англомовний, тож очікувана точність буде низька
        predicted = nltk.pos_tag(words)
        predicted_tags = [tag for word, tag in predicted]
        for p, t in zip(predicted_tags, true_tags):
            if p == t:
                correct += 1
            total += 1
    return correct / total

nltk_acc = nltk_pos_accuracy(test_sents)
print(f"Accuracy of NLTK English POS-tagger on Sinica corpus: {nltk_acc:.4f}")
