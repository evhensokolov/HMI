import re
from collections import defaultdict, Counter
from nltk.util import ngrams
import math

# === 1. Завантаження та попередня обробка корпусу ===
def load_corpus(file_path, max_tokens=200_000):
    tokens = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            tokens.extend(line.strip().lower().split())
            if len(tokens) >= max_tokens:
                break
    return tokens[:max_tokens]


#Побудова N-грам моделі
def build_ngram_model(tokens, n):
    model = defaultdict(Counter)
    for gram in ngrams(tokens, n):
        prefix, word = tuple(gram[:-1]), gram[-1]
        model[prefix][word] += 1
    return model

def predict_next(model, context, top_k=5):
    context = tuple(context[-(len(next(iter(model))) if model else 0):])
    candidates = model.get(context, {})
    total = sum(candidates.values())
    probs = {word: count / total for word, count in candidates.items()} if total > 0 else {}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]

#Автозавершення тексту 
def autocomplete(text, models, top_k=5):
    tokens = text.lower().split()
    for n in reversed(range(2, 6)):
        if len(tokens) >= n - 1:
            context = tokens[-(n - 1):]
            if tuple(context) in models[n]:
                return predict_next(models[n], context, top_k)
    total = sum(models[1].values())
    probs = {word: count / total for word, count in models[1].items()}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]

#Перплексія
def perplexity(model, tokens, n):
    N = 0
    log_prob = 0
    for gram in ngrams(tokens, n):
        prefix, word = tuple(gram[:-1]), gram[-1]
        prefix_count = sum(model[prefix].values())
        word_count = model[prefix][word]
        prob = word_count / prefix_count if prefix_count > 0 else 1e-6
        log_prob += math.log(prob)
        N += 1
    return math.exp(-log_prob / N)

#
def create_ui(models):
    import ipywidgets as widgets
    from IPython.display import display

    input_box = widgets.Text(description="Введи текст:")
    output = widgets.Output()

    def on_submit(change):
        output.clear_output()
        with output:
            predictions = autocomplete(change.new, models)
            print("Можливі продовження:")
            for word, prob in predictions:
                print(f"{word} ({prob:.4f})")

    input_box.observe(on_submit, names='value')
    display(input_box, output)

#Головний запуск
if __name__ == "__main__":
    file_path = "wiki_dump.tokenized.txt"
    print("Завантаження корпусу...")
    tokens = load_corpus(file_path, max_tokens=200_000)

    print("Побудова моделей N-грам...")
    models = {}
    for n in range(1, 6):
        if n == 1:
            models[n] = Counter(tokens)
        else:
            models[n] = build_ngram_model(tokens, n)

    print("Тест автозавершення:")
    test_input = "the united"
    predictions = autocomplete(test_input, models)
    print(f"Введення: '{test_input}'")
    for i, (word, prob) in enumerate(predictions):
        print(f"{i+1}. {word} (ймовірність: {prob:.4f})")

    print("\nОцінка перплексії (на 3-грамі):")
    pp = perplexity(models[3], tokens[:10000], 3)
    print(f"Перплексія: {pp:.2f}")
