import re
from collections import Counter
import numpy as np

#Зчитування тексту та створення словника частотності
def read_text(file_path):
    with open(file_path, encoding="utf-8") as f:
        return re.findall(r'\w+', f.read().lower())

def get_word_count(words_list):
    return Counter(words_list)

def get_probabilities(word_counts):
    total = sum(word_counts.values())
    return {word: count / total for word, count in word_counts.items()}

#Операції редагування
def delete_letter(word):
    return [word[:i] + word[i+1:] for i in range(len(word))]

def insert_letter(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [word[:i] + c + word[i:] for i in range(len(word) + 1) for c in letters]

def replace_letter(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return [word[:i] + c + word[i+1:] for i in range(len(word)) for c in letters if word[i] != c]

def switch_letter(word):
    return [word[:i] + word[i+1] + word[i] + word[i+2:] for i in range(len(word)-1)]

def edit_one_letter(word, allow_switches=True):
    edits = set()
    edits.update(delete_letter(word))
    edits.update(insert_letter(word))
    edits.update(replace_letter(word))
    if allow_switches:
        edits.update(switch_letter(word))
    return edits

def edit_two_letters(word, allow_switches=True):
    edits = set()
    for e1 in edit_one_letter(word, allow_switches):
        edits.update(edit_one_letter(e1, allow_switches))
    return edits

#Пошук кандидатів
def get_candidates(word, vocab, probs, n=1):
    candidates = []
    
    if word in vocab:
        candidates = [word]
    else:
        edits1 = edit_one_letter(word) & vocab
        edits2 = edit_two_letters(word) & vocab
        candidates = edits1 or edits2 or [word]

    return sorted([(w, probs.get(w, 0)) for w in candidates], key=lambda x: x[1], reverse=True)[:n]

#Мінімальна відстань редагування 
def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
    m, n = len(source), len(target)
    D = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        D[i][0] = i * del_cost
    for j in range(n+1):
        D[0][j] = j * ins_cost

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if source[i-1] == target[j-1] else rep_cost
            D[i][j] = min(
                D[i-1][j] + del_cost,
                D[i][j-1] + ins_cost,
                D[i-1][j-1] + cost
            )
    return D[m][n]

def autocorrect(word, vocab, probs):
    suggestions = get_candidates(word, vocab, probs, n=1)
    return suggestions[0][0] if suggestions else word

if __name__ == "__main__":
    word_list = read_text("shakespeare.txt")
    word_counts = get_word_count(word_list)
    vocab = set(word_counts)
    probs = get_probabilities(word_counts)

    test_words = ['recieve', 'speek', 'frend', 'beautee', 'monarck']

    print("=== Результати автокорекції ===")
    for word in test_words:
        corrected = autocorrect(word, vocab, probs)
        print(f"{word} -> {corrected}")

    print("\n=== Мінімальна відстань редагування ===")
    pairs = [('intention', 'execution'), ('sunday', 'saturday'), ('play', 'stay')]
    for w1, w2 in pairs:
        dist = min_edit_distance(w1, w2)
        print(f"{w1} -> {w2} : {dist}")
