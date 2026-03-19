import re
from better_profanity import profanity

word = []
prof = []
profanity.load_censor_words()
for round in range(1, 6):
    with open(f"singleline/stage_{round}.txt", "r", encoding="utf-8") as f:
        text = f.read()

    with open(f"conversation/stage_{round}_c.txt", "r", encoding="utf-8") as f:
        text += f.read()

    tokens = re.findall(r"\b\w+\b", text)
    word_count = len(tokens)
    word.append(word_count)

    profanity_count = sum(profanity.contains_profanity(t) for t in tokens)
    prof.append(profanity_count)

print(word)
print(prof)
