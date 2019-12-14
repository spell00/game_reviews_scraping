import logging
import time
import warnings
import numpy as np
import gensim
import pandas as pd
import spacy
import os
from gensim import utils
from gensim.models import Word2Vec
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def display_closestwords_tsnescatterplot(model, word, topn=25,size=300):
    arr = np.empty((0, size), dtype='f')
    word_labels = [word]

    close_words = model.similar_by_word(word, topn=topn)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)


    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords, c="b")
    for i, (label, x, y) in enumerate(zip(word_labels, x_coords, y_coords)):
        if label is not word:
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        else:
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
            plt.scatter(x_coords[i], y_coords[i], c="r")

    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    # plt.vlines(0, colors="r")
    # plt.hlines(0, colors="r")
    plt.savefig("voisins_tsne_" + word + ".png")


def display_closestwords_pcascatterplot(model, word, topn=10,size=300):
    arr = np.empty((0, size), dtype='f')
    word_labels = [word]

    close_words = model.similar_by_word(word, topn=topn)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    pca = PCA()
    np.set_printoptions(suppress=True)
    Y = pca.fit_transform(arr)


    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for i, (label, x, y) in enumerate(zip(word_labels, x_coords, y_coords)):
        if label is not word:
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        else:
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
            plt.scatter(x_coords[i], y_coords[i], c="r")

    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    # plt.vlines(0, colors="r")
    # plt.hlines(0, colors="r")
    plt.savefig("voisins_pca_" + word + ".png")


def remove_out_of_context_from_list(list):
    pass



class Corpus:
    def __init__(self, path):
        self.path = path

        self.train_corpus = pd.read_csv(self.path, names=["id", "funny", "helpful", "recommend", "review"])

    def __iter__(self):
        for line in self.train_corpus["review"][1:]:
            try:
                yield utils.simple_preprocess(line)
            except:
                yield utils.simple_preprocess("")


class EmbeddingsModel:

    def __init__(self):
        self.sentences = None

        self.model = None

    def train(self, absolute_path, checkpoint=None, n_epochs=100):

        self.sentences = Corpus(absolute_path)
        tic = time.clock()
        if checkpoint is not None:
            from gensim.models import KeyedVectors

            self.model = Word2Vec(size=300, min_count=1)
            self.model.build_vocab(self.sentences)
            total_examples = self.model.corpus_count
            model = KeyedVectors.load_word2vec_format("saved_models/GoogleNews-vectors-negative300.bin", binary=True)
            self.model.build_vocab([list(model.vocab.keys())], update=True)
            self.model.intersect_word2vec_format("saved_models/GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)
            print("Fine-tuning for", )
            self.model.train(self.sentences,
                             total_examples=total_examples,
                             epochs=n_epochs,
                             compute_loss=True)
            print("Fine-tuning is over.")
        else:
            self.model = gensim.models.Word2Vec(
                self.sentences,
                min_count=5,
                size=400,
                iter=n_epochs,
                compute_loss=True,
                sg=1,
                hs=0,
                workers=10
            )
        toc = time.clock()
        print()
        print("Temps d'entrainement:", toc - tic, " secondes.")

    def save_model(self, absolute_path):
        self.model.save(absolute_path)

    def load_model(self, absolute_path):
        self.model = gensim.models.Word2Vec.load(absolute_path)

    def make_word2vec(self, absolute_path):
        from gensim.models import KeyedVectors
        self.model = KeyedVectors.load_word2vec_format(absolute_path, binary=True)
        self.model.vocab = self.sentences
        self.model.save(os.getcwd() + "/saved_models/word2vec.model")
        print("word2vec can now be loaded and fine-tuned")

    def load_spacy_model(self):
        self.nlp = spacy.load("./saved_models/spacy.word2vec.model/")

        def get_item_in_tuple(item):
            return item[1]

        dict = {}
        prune_vectors = self.nlp.vocab.prune_vectors(5000, 25799)

        for word in prune_vectors.keys():
            new_key = prune_vectors[word][0]

            new_tuple = (word, prune_vectors[word][1])
            try:
                dict[new_key].append(new_tuple)
            except:
                dict[new_key] = [new_tuple]

        for key in dict.keys():
            dict[key] = sorted(dict[key], key=get_item_in_tuple, reverse=True)

        dict_sorted = {}
        for k in sorted(dict, key=lambda k: len(dict[k]), reverse=True):
            dict_sorted[k] = dict[k]

        # for key in dict_sorted.keys():
        #     print(key, ":", dict[key])

        self.write_dict_to_file(dict_sorted)

    def write_dict_to_file(self, dict):
        f = open("voisins.txt", "w")
        for k in dict.keys():
            formated_string = str(k) + "\t" + str(len(dict[k])) + "\t"
            for tuple in dict[k][:15]:
                formated_string += (str(tuple[0]) + " ")
            formated_string += "\n"
            f.write(formated_string)
        f.close()

    # python -m spacy init-model en ./saved_models/spacy.word2vec.model --vectors-loc ./saved_models/word2vec.txt
    def save_model_for_spacy(self, absolute_path):
        self.model.wv.save_word2vec_format(absolute_path, binary=False)

    def get_vocab(self):
        return self.model.wv.vocab

    def get_training_loss(self):
        return self.model.get_latest_training_loss()

    def get_accuracy_questions_words(self):
        return self.model.accuracy("questions-words.txt")




if __name__ == '__main__':
    my_model = EmbeddingsModel()

    # my_model.make_word2vec(os.getcwd() + "/saved_models/GoogleNews-vectors-negative300.bin.gz")

    train = False
    if train:
        my_model.train("steam_australia.csv", checkpoint=os.getcwd() + "/saved_models/word2vec.model")
        my_model.save_model("./saved_models/saved_gensim_model_word2vec")
    elif not train:
        draw = False
        stats = True

        #game_names = []
        #with open("game_names.txt") as fp:
        #    for cnt, line in enumerate(fp):
        #        game_names += [line.strip("\n").replace(" ", "")]
        path = os.path.abspath(os.getcwd() + "/saved_models/saved_gensim_model_word2vec")
        my_model.load_model(path)
        #vocab = list(my_model.get_vocab().keys())
        #w = open("vocab_word2vec.txt", "w+")

        #with tqdm(total=len(vocab)) as pbar:
        #    for v in vocab:
        #        w.write(v + "\n")
        #        pbar.update(1)
        #w.close()

        # These names are in game_names, obtained in draft.py from taking words before the pattern "is a * game"
        # The list is small and noisy. We will take a single popular game,
        # top100 = my_model.model.most_similar('Borderlands', topn=100)
        # top100_gow = my_model.model.most_similar('god', topn=100)
        # top100_gow2 = my_model.model.most_similar('GOW', topn=100)
        # top100_ff = my_model.model.most_similar('FF', topn=100)
        # top100_ff7 = my_model.model.most_similar('FF7', topn=100)
        display_closestwords_tsnescatterplot(my_model.model, "Borderlands")
        display_closestwords_tsnescatterplot(my_model.model, "Portals")
        display_closestwords_tsnescatterplot(my_model.model, "FFVII")
        display_closestwords_pcascatterplot(my_model.model, "Borderlands")
        display_closestwords_pcascatterplot(my_model.model, "Portals")
        display_closestwords_pcascatterplot(my_model.model, "FFVII")

        # my_model.save_model_for_spacy(os.getcwd() + "/saved_models/word2vec.txt")
        # my_model.load_spacy_model()
        # my_model.write_annotations_file()

        if stats:
            print(my_model.get_training_loss())
            # print(my_model.get_accuracy_questions_words())
            print(len(my_model.get_vocab()))

        if draw:
            x_vals, y_vals, labels = my_model.reduce_dimensions()

            my_model.plot_with_matplotlib(x_vals, y_vals, labels)
