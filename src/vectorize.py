import logging
import time
import warnings
import numpy as np
import gensim
import spacy
import os
from gensim import utils
from gensim.models import Word2Vec
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pattern.text.en import parse, tag
import argparse
import pandas as pd
from src.utils import create_missing_folders
from gensim.models import KeyedVectors
from nltk import word_tokenize

warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def display_closestwords_tsnescatterplot(model, word, dest, topn=25,size=300):
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
    plt.savefig("images/neighbors/{}/tsne_{}.png".format(dest, word))


def display_closestwords_pcascatterplot(model, word, dest, topn=10,size=300):
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
    create_missing_folders(os.getcwd() + "/images/neighbors/{}".format(dest))
    plt.savefig("images/neighbors/{}/pca_{}.png".format(dest, word))


def remove_out_of_context_from_list(list):
    pass

def get_word2vec_vocab():
    model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
    vocab = model.wv.vocab
    w = open("extracted/lists/GoogleNews-vectors-negative300_NNP", "w+")
    with tqdm(total=len(vocab)) as pbar:
        for v in vocab:
            if parse(v + "\n").split("/")[1] == "NNP" and " ".join(v.lower().split("_").capitalize()) == " ".join(v.split("_")):
                w.write(v + "\n")
            pbar.update(1)
    w.close()


class Corpus:
    def __init__(self, path):
        self.path = path

        self.train_corpus = pd.read_csv(self.path, names=["id", "funny", "helpful", "recommend", "review"])["review"][1:]
        self.train_corpus = [str(sent) for sent in self.train_corpus]

    def __iter__(self):
        for line in self.train_corpus:
            try:
                yield word_tokenize(line)
            except:
                yield ""


class EmbeddingsModel:

    def __init__(self):
        self.sentences = None

        self.model = None

    def train(self, absolute_path, checkpoint=None, n_epochs=100, size=400, min_count=1, sg=1, hs=0,
              base_dir="saved_models"):

        self.sentences = Corpus(absolute_path)
        tic = time.clock()
        if checkpoint is not None:
            print("Uning weights for pre-trained model:", checkpoint)
            self.model = Word2Vec(size=300, min_count=5)
            self.model.build_vocab(self.sentences)
            # self.model.build_vocab(self.sentences.train_corpus)
            total_examples = self.model.corpus_count
            model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
            self.model.build_vocab([list(model.vocab.keys())], update=True)
            self.model.intersect_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)
            print("Fine-tuning for", )
            self.model.train(self.sentences,
                             total_examples=total_examples,
                             epochs=n_epochs,
                             compute_loss=True)
            print("Fine-tuning is over.")
        else:
            print("No checkpoint. Training one of size" + str() + " for", n_epochs, "epochs")
            self.model = gensim.models.Word2Vec(
                self.sentences,
                min_count=min_count,
                size=size,
                iter=n_epochs,
                compute_loss=True,
                sg=sg,
                hs=hs,
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
        f = open("../extracted/voisins.txt", "w")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="GoogleNews-vectors-negative300.bin", help="Name of pretrained model file (default: none)")
    parser.add_argument("--train", default="false", help="Train?")
    parser.add_argument("--draw", default="false", help="Draw?")
    parser.add_argument("--stats", default="false", help="Stats?")
    parser.add_argument("--save_name", default="gensim_model", help="Name of the model checkpoint that will be saved")
    parser.add_argument("--dir", default="saved_models", help="Name of the model checkpoint that will be saved")
    parser.add_argument("--csv_fname", default="steam_australia_norm_w2v.csv", help="Name of the csv file that will be saved")
    parser.add_argument("--vocab_fname", default="vocab_word2vec_steam.txt", help="Name of the model vocab file that will be saved")
    parser.add_argument("--save_fname", default="steam_australia_norm_w2v", help="Name of the model vocab file that will be saved")
    args = parser.parse_args()
    if args.train == "true":
        train = True
    else:
        train = False
    if args.draw == "true":
        draw = True
    else:
        draw = False
    if args.stats == "true":
        stats = True
    else:
        stats = False
    if args.pretrained == "none":
        pretrained = None
    else:
        pretrained = args.pretrained
    checkpoint = "{}/{}/{}/pretrained_{}".format(os.getcwd(), args.dir, args.save_name, str(args.pretrained).split(".bin")[0])
    create_missing_folders(checkpoint)
    checkpoint = "{}/{}.model".format(checkpoint, args.save_fname)

    if train:
        my_model.train("extracted/dataframes/{}".format(args.csv_fname), checkpoint=pretrained)
        my_model.save_model(checkpoint)
    elif not train:
        my_model.load_model(checkpoint)
        vocab = list(my_model.get_vocab().keys())

        w = open("extracted/lists/" + args.vocab_fname, "w+")
        with tqdm(total=len(vocab)) as pbar:
            for v in vocab:
                w.write(v + "\n")
                pbar.update(1)
        w.close()

        w = open("extracted/lists/vocab_word2vec_POS.txt", "w+")
        with tqdm(total=len(vocab)) as pbar:
            for v in vocab:
                w.write(tag(v) + "\n")
                pbar.update(1)
        w.close()

        w = open("extracted/lists/" + checkpoint + "_parse.txt", "w+")
        with tqdm(total=len(vocab)) as pbar:
            for v in vocab:
                w.write(parse(str(v) + "\n") + "\n")
                pbar.update(1)
        w.close()


        # These names are in game_names, obtained in scrape.py from taking words before the pattern "is a * game"
        # The list is small and noisy. We will take a single popular game,
        # top100 = my_model.model.most_similar('Borderlands', topn=100)
        tops_adj = {
            'Borderlands': None,
            'facets': None,
            'adjective': None,
            'type': None,
            'multiplayer': None,
            'characteristics': None,
            'gameplay': None,   # Good example for 3- find adjectives (or other information) characteristics of a facet of a game
            'god': None,
            'FF7': None,
            'rts': None,
        }

        tops_nn = {
            'Borderlands': None,
            'facets': None,
            'facet': None,
            'adjective': None,
            'collection': None,
            'descriptor': None,
            'description': None,
            'gametype': None,  # GOOD ONE FOR 4- identify types of games
            'classification': None,
            'category': None,
            'genre': None,
            'genres': None,
            'gametypes': None,
            'multiplayer': None,
            'characteristics': None,
            'gameplay': None,
            'aspects': None,
            'features': None,   # GOOD ONE FOR 2- identify aspects (features) of a game
            'feature': None,
            'fortress': None,
            'style': None,
            'Team_Fortress_2': None,
        }

        for top in tops_adj:
            tops_adj[top] = my_model.model.most_similar(top, topn=1000)
            top_words = np.array(tops_adj[top])[:, 0].tolist()
            scores = np.array(tops_adj[top])[:, 1].tolist()
            top_words = [parse(str(k) + "\n") for k in top_words]
            top_jj_indices = [i for i, x in enumerate(top_words) if x.split("/")[1] in ["JJ", "JJR", "JJS"]]

            top_words = np.concatenate((np.array([top_words[i].split("/")[0] for i in top_jj_indices]).reshape(-1, 1),
                                        np.array([np.round(float(scores[i]), 3) for i in top_jj_indices]).reshape(-1,
                                                                                                                  1)),
                                       1)
            tops_adj[top] = pd.DataFrame(top_words, columns=["adjectives", "similarity"])
            tops_adj[top].to_csv("extracted/adjectives/{}".format(top), index=False)

        for top in tops_nn:
            tops_nn[top] = my_model.model.most_similar(top, topn=1000)
            top_words = np.array(tops_nn[top])[:, 0].tolist()
            scores = np.array(tops_nn[top])[:, 1].tolist()
            top_words = [parse(str(k) + "\n") for k in top_words]
            top_jj_indices = [i for i, x in enumerate(top_words) if x.split("/")[1] in ["NN", "NNS"]]

            top_words = np.concatenate((np.array([top_words[i].split("/")[0] for i in top_jj_indices]).reshape(-1, 1),
                                        np.array([np.round(float(scores[i]), 3) for i in top_jj_indices]).reshape(-1,
                                                                                                                  1)),
                                       1)
            tops_nn[top] = pd.DataFrame(top_words, columns=["adjectives", "similarity"])
            tops_nn[top].to_csv("extracted/adjectives/{}".format(top), index=False)

        for top in tops_adj:
            display_closestwords_pcascatterplot(my_model.model, top, dest="adjectives")

        for top in tops_nn:
            display_closestwords_pcascatterplot(my_model.model, top, dest="nouns")
            #display_closestwords_pcascatterplot(my_model.model, "Portals")
            #display_closestwords_pcascatterplot(my_model.model, "FFVII")
            #display_closestwords_pcascatterplot(my_model.model, "fortress")
            #display_closestwords_pcascatterplot(my_model.model, "Team_Fortress_2")

        if stats:
            print(my_model.get_training_loss())
            # print(my_model.get_accuracy_questions_words())
            print(len(my_model.get_vocab()))

        if draw:
            x_vals, y_vals, labels = my_model.reduce_dimensions()

            my_model.plot_with_matplotlib(x_vals, y_vals, labels)
