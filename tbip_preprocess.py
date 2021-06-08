import pandas as pd
import numpy as np
import os, csv, sys
from tqdm import tqdm

from random import sample
import collections

from scipy import sparse
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, string, re, os, sys
from nltk.corpus import stopwords
from stop_words import get_stop_words

# Generar stopwords
nltk.download('stopwords')
custom_stopwords = get_stop_words('spanish') + stopwords.words('spanish') + list(string.ascii_lowercase)
translation = str.maketrans('áéíóöúüñ', 'aeioouun')
custom_stopwords = list(set([sw.translate(translation) for sw in custom_stopwords]))
custom_stopwords.sort()

# Lista para agregar nuevas stopwords
new_stopwords = []

# Función para limpiar texto
from unicodedata import normalize
def clean_text(s):
    s = s.lower()
    s = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize('NFD', s), 0, re.I)
    s = normalize('NFC', s).replace('ñ','n')
    s = re.sub('[^a-z ]',' ', s)
    s = re.sub(' +',' ', s).strip()
    return s

def preprocess(df):
    '''
    :param df: df con variables "nombre" y "texto"
    :return: save npys
    '''
    df['clean_texto'] = df['texto'].apply(clean_text)

    # Extraer nombres para stopwords
    nombres = list(set(' '.join(df['nombre'].unique()).replace('-',' ').replace("'",'').split(' ')))

    # Extraer autores y textos en formato numpy
    authors = np.array(df['nombre'].apply(str))
    texts = np.array(df['texto'])

    # Generar id interno al autor
    author_to_author_id = dict(
        [(y.title(), x) for x, y in enumerate(sorted(set(authors)))])

    # Ids internos de autores
    author_indices = np.array(
        [author_to_author_id[s.title()] for s in authors])

    # Ids de autores
    author_map = np.array(list(author_to_author_id.keys()))

    # Definir contador tokens por cada documento
    count_vectorizer = CountVectorizer(ngram_range = (1, 3),
                                       min_df = 0.005,
                                       max_df = 0.5,
                                       stop_words = custom_stopwords + new_stopwords + nombres,
                                       token_pattern = '[a-z]+',
                                       lowercase = True,
                                       strip_accents = 'ascii')


    # Contar tokens y generar vocabulario
    counts = count_vectorizer.fit_transform(texts)
    vocabulary = np.array([k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), key=lambda kv: kv[1])])


    # Ajustar counts removiendo pares de unigram/n-gram que co-ocurren.
    # La función era esta:
    # counts_dense = utils.remove_cooccurring_ngrams(counts, vocabulary)
    # Pero utilizaba más de 25GB de ram
    # El código que sigue es la misma función pero optimizada en algunas partes

    # `n_gram_to_unigram` takes as key an index to an n-gram in the vocabulary
    # and its value is a list of the vocabulary indices of the corresponding
    # unigrams.
    n_gram_indices = np.where(
        np.array([len(word.split(' ')) for word in vocabulary]) > 1)[0]
    n_gram_to_unigrams = {}
    for n_gram_index in n_gram_indices:
        matching_unigrams = []
        for unigram in vocabulary[n_gram_index].split(' '):
            if unigram in vocabulary:
                matching_unigrams.append(np.where(vocabulary == unigram)[0][0])
        n_gram_to_unigrams[n_gram_index] = matching_unigrams

    # `n_grams_to_bigrams` now breaks apart trigrams and higher to find bigrams
    # as subsets of these words.
    n_grams_to_bigrams = {}
    for n_gram_index in n_gram_indices:
        split_n_gram = vocabulary[n_gram_index].split(' ')
        n_gram_length = len(split_n_gram)
        if n_gram_length > 2:
            bigram_matches = []
            for i in range(0, n_gram_length - 1):
                bigram = " ".join(split_n_gram[i:(i + 2)])
            if bigram in vocabulary:
                bigram_matches.append(np.where(vocabulary == bigram)[0][0])
            n_grams_to_bigrams[n_gram_index] = bigram_matches

    # Go through counts, and remove a unigram each time a bigram superset
    # appears. Also remove a bigram each time a trigram superset appears.
    # Note this isn't perfect: if bigrams overlap (e.g. "global health care"
    # contains "global health" and "health care"), we count them both. This
    # may introduce a problem where we subract a unigram count twice, so we also
    # ensure non-negativity.
    counts_dense = counts.toarray()
    for i in range(len(counts_dense)):
        n_grams_in_doc = np.where(counts_dense[i, n_gram_indices] > 0)[0]
        sub_n_grams = n_gram_indices[n_grams_in_doc]
        for n_gram in sub_n_grams:
            counts_dense[i, n_gram_to_unigrams[n_gram]] -= counts_dense[i, n_gram]
            if n_gram in n_grams_to_bigrams:
                counts_dense[i, n_grams_to_bigrams[n_gram]] -= counts_dense[i, n_gram]

    counts_dense[np.where(counts_dense < 0)] = 0
    counts = sparse.csr_matrix(counts_dense)

    # Eliminar documentos con 0 tokens en vocabulario
    tokens_by_doc = np.sum(counts > 0, axis = 1)
    keep = np.where(tokens_by_doc > 0)[0]

    # Eliminar autores cuyos documentos no poseen ningún token en el vocabulario
    authors = authors[keep]
    author_to_author_id = dict(
        [(y.title(), x) for x, y in enumerate(sorted(set(authors)))])
    author_indices = np.array(
        [author_to_author_id[s.title()] for s in authors])
    author_map = np.array(list(author_to_author_id.keys()))

    # Guardar datos en save_dir
    save_dir = 'tbip_data/clean'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sparse.save_npz(os.path.join(save_dir, "counts.npz"), sparse.csr_matrix(counts[keep]).astype(np.float32))
    np.save(os.path.join(save_dir, "author_indices.npy"), author_indices)
    np.savetxt(os.path.join(save_dir, "vocabulary.txt"), vocabulary, fmt="%s")
    np.savetxt(os.path.join(save_dir, "author_map.txt"), author_map, fmt="%s")
