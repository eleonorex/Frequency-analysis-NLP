import pymorphy2
import requests
from bs4 import BeautifulSoup
import nltk
import spacy
from collections import defaultdict as dd
from operator import itemgetter
import string
from rusenttokenize import ru_sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

nlp = spacy.load("ru_core_news_sm")
morph_analyzer = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words('russian')
punctuation = punctuation + "«»—"


def tokenize(text, remove_stop_words=True):
    text_preprocessed_tokenized = []

    for sentence in ru_sent_tokenize(text):
        clean_words = [word.strip(string.punctuation).lower() for word in word_tokenize(text)]

        clean_words = [word for word in clean_words if ((word not in russian_stopwords) | (not remove_stop_words))
                       and word != " " and word.strip() not in punctuation]

        clean_lemmas = [morph_analyzer.parse(word)[0].normal_form for word in clean_words]

        text_preprocessed_tokenized.extend(clean_lemmas)

    return text_preprocessed_tokenized


# _________________________________ WEB PAGE PARSING TO COLLECT TEXT

url = "https://www.chukfamily.ru/kornei/tales/barmalej"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}
page = requests.get(url, headers=headers)
parsed = BeautifulSoup(page.text, "html.parser")

tale = "\n".join([a.text for a in parsed.select(".proza-contents")])

with open("text.txt", 'w') as file:
    file.write(tale)


# _______________________________________ FREQUENCY ANALYSIS

document_tokenized = tokenize(tale)

text_words_frequencies = dd(int)

for word in document_tokenized:
    text_words_frequencies[word] += 1 / len(document_tokenized)

sorted_frequency_table = sorted(text_words_frequencies.items(), key=itemgetter(1), reverse=True)

frequencies = []
for lemma, freq in sorted_frequency_table[:10]:
    frequencies.append('\t'.join((lemma, str(freq))))


# ______________________________________ PART OF SPEECH ORDER
pos = []
for word in tokenize(tale, remove_stop_words=False):
    pos.append(morph_analyzer.parse(word)[0].tag.POS)

pos_pairs = [p for p in nltk.pairwise(pos)]
pairs_freq = dd(int)
for pair in pos_pairs:
    pairs_freq[pair] += 1 / len(pos_pairs)

freq_pos_table = sorted(pairs_freq.items(), key=itemgetter(1), reverse=True)
for p, freq in freq_pos_table[:10]:
    print('\t'.join((str(p), str(freq))))


# _______________________________________ INDEXES FOR PART 1

frag1 = open("text.txt", "r").readlines()[1:12]

frag1_tokenized = []
verbs = []
verbs_frequencies = dd(int)
for line in frag1:
    for word in tokenize(line):
        frag1_tokenized.append(word)
        if morph_analyzer.parse(word)[0].tag.POS == 'INFN':
            verbs.append(word)

for verb in verbs:
    verbs_frequencies[verb] += 1 / len(frag1_tokenized)

verbs_table = sorted(verbs_frequencies.items(), key=itemgetter(1), reverse=True)
for lemma, freq in verbs_table[:10]:
    print('\t'.join((lemma, str(freq))))

# _______________________________________ INDEXES FOR PART 2

frag2 = open("text.txt", "r").readlines()[30:34]

lemmas_frequencies = dd(int)
lemmas = []
for token in tokenize('\t'.join(frag2)):
    lemmas.append(morph_analyzer.parse(token)[0][0])

for lemma in lemmas:
    lemmas_frequencies[lemma] += 1 / len(tokenize('\t'.join(frag2)))

lemmas_table = sorted(lemmas_frequencies.items(), key=itemgetter(1), reverse=True)
for lemma, freq in lemmas_table[:10]:
    print('\t'.join((lemma, str(freq))))
