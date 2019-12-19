import jsbeautifier
import numpy as np
import pandas as pd
from tqdm import tqdm
from pattern.text.en import polarity, positive, subjectivity
from gensim.parsing.preprocessing import remove_stopwords
import string
from src.preprocess import preprocessing, SymSpell
from nltk.corpus import stopwords
import spacy
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from pattern.text.en import parse

spacy_nlp = spacy.load('en_core_web_lg')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
stop_words = set(list(set(stopwords.words('english'))) + list(spacy_stopwords))


def get_word2vec_vocab():
    """
    Gets the vocabulary from pretrained word2vec and writes a new file with all the vocabulary in a new file
    The file contains a single word per line. Over 2 billion words
    :return: Nothing
    """
    print("Extracting word2vec vocabulary")
    model = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin", binary=True)
    vocab = model.wv.vocab
    w = open("extracted/lists/GoogleNews-vectors-negative300", "w+")
    with tqdm(total=len(vocab)) as pbar:
        for v in vocab:
            if parse(v + "\n").split("/")[1] == "NNP" and " ".join(v.lower().split("_").capitalize()) == " ".join(v.split("_")):
                w.write(v + "\n")
            pbar.update(1)
    w.close()


def count_words(filepath ='extracted/lists/australian_user_reviews.txt',
                names_file="extracted/lists/games_abbreviations.txt"):
    """
    Looks for games in the filepath and writes the
    list of games that were found extracted/lists/game_lines.csv"

    :param filepath: data in which the words are counted
    :param names_file: file containing the name in a csv file. In if a single column of words it should work as a single coulmn csv
    :return:
    """
    games_dict = {}
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            with open(names_file) as n:
                for i, name in enumerate(n):
                    name = name.strip("\n")
                    name, abbr = name.split(",")
                    if name in line or (abbr in line and len(abbr) > 1):
                        if name not in games_dict:
                            games_dict.update({name: [cnt]})
                        else:
                            games_dict[name] += [cnt]
    w = open("extracted/lists/game_lines.csv", "w+")
    for game in games_dict.keys():
        w.write("{},{}\n".format(game, games_dict[game]))
    w.close()


# Other startegies:
# - If the First few words with Caps Might Be Games is... ( . League of Legend is ... )
# - Words with only caps might be accronyms. people i think go much more with accronyms if name is long ()
#       * Will need to verify what it is an accronym for, or at least what it is not
#       * faire un dictionnaire de vocabulaire avec texte des tp precedents. Les mots ALL CAPS trouvés devront être
#         vérifiés s'ils sont des vrais mots (e.g. AAAAAH -> ah, WWWWHHHHYYYYYYY -> why, FFIV -> ffiv is not a word,
#         thus a game accronym)


def look_for_words(filepath ='extracted/lists/australian_user_reviews.txt',
                   write_to="extracted/lists/game_names.txt",
                   write_rejected="extracted/lists/game_names_rejected_for_length.txt",
                   what_is_it=" is a ",
                   what_it_is="game",
                   cutoff=5,
                   this_list=['Now', 'Game', 'game', 'the game', 'Now', 'now', "There", "there", "This", "this",
                                      "this game", "", "it", "It"]):
    """
    Looks for games in filepath and writes the list of games that were found extracted/lists/game_lines.csv"

    :param filepath: Path to the file containing the documents to analyse (1 doc per line)
    :param write_to: Name of the file the patterns found
    :param write_rejected: list of words that were rejected, but made it close (for further inspection)
    :param what_is_it: e.g. " is a " to be followed by "game" . In a corpus about hockey, it might often be "is a * player"
    :param what_it_is: e.g. game
    :param cutoff: If the sequence if too long the word is rejected
    :param this_list: list of words that might be identified to be avoided
    :return:
    """
    print("Finding games...")

    w = open(write_to, "w+")
    w2 = open(write_rejected, "w+")
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            line_split = line.split(what_is_it)
            if len(line_split) > 1:
                line_split1 = line_split[1].split(".")[-1].split(",")[-1].split("!")[-1].split("?")[-1].replace("'",
                                                                                                                "").replace(
                    '"', "").replace("-", "").replace("_", "").replace("*", "")
                line_split = line_split[0].split(".")[-1].split(",")[-1].split("!")[-1].split("?")[-1].replace("'",
                                                                                                               "").replace(
                    '"', "").replace("-", "").replace("_", "").replace("*", "")
                if line_split is "":
                    continue
                if line_split not in this_list and (
                        what_it_is in line_split1.split(" ")):
                    flag = False
                    for word in line_split.split(" "):
                        if word in ["This", "this", "", "it", "It"]:
                            flag = True
                    if flag is False:
                        if len(line_split.split(" ")) <= cutoff:
                            line_split = " ".join([l.capitalize() for l in line_split.split(" ")])
                            w.write(line_split + '\n')
                        else:
                            w2.write(line_split + '\n')

    w.close()
    w2.close()


def unique_words(filepath="extracted/lists/game_names.txt", write_to="extracted/lists/games_unique.txt"):
    """
    Takes the list in extracted/lists/game_names.txt and writes the file extracted/lists/games_unique.txt
    that will have all the unique games names in it, one name per line
    """
    gamenames = {}
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            if line not in gamenames:
                gamenames.update({line: 0})
            else:
                gamenames[line] += 1
    w = open(write_to, "w+")
    for game in gamenames.keys():
        w.write("{}".format(game))
    w.close()


def words_abbreviation(filepath="extracted/lists/games_unique.txt", write_to="extracted/lists/games_abbreviations.txt"):
    """
    Takes the list in  and writes the file games_unique.txt that will have all the
    unique games names in it
    """
    gamenames = {}
    with open(filepath) as fp:
        for cnt, game in enumerate(fp):
            game = game.split("\n")[0]
            abb = "".join(''.join(x for x in game.replace("'", "") if not x.islower()).split(" "))
            if game not in gamenames:
                gamenames.update({game: abb})
            else:
                exit("whoops")
    w = open(write_to, "w+")
    for game in gamenames.keys():
        w.write("{},{}\n".format(game, gamenames[game]))
    w.close()


def get_user_ids(line):
    """
    Finds the id of a user in a line of a json file
    :param line: The line containing the id
    :return:
    """
    try:
        id = line.split("'user_id': ")[1]
        if id[0] == "'":
            id = id.split(",")[0].strip("'")
        elif id[0] == '"':
            id = id.split('\"\n')[0]
        else:
            exit("unexpected start of user_id")
    except:
        # print(cnt, "has an empty user_id section")
        id = ''
    return id


def get_funny(funny):
    try:
        if funny[0] == "'":
            funny = funny.split(",")[0].strip("'")
        elif funny[0] == '"':
            funny = funny.split('\"\n')[0]
        else:
            exit("unexpected start of funny")
    except:
        # print(cnt, "has an empty funny section")
        funny = ''
    return funny


def get_helpful(helpful):
    try:
        # helpful = line.split("'helpful': ")[1]
        if helpful[0] == "'":
            helpful = helpful.split(",")[0].strip("'")
        elif helpful[0] == '"':
            helpful = helpful.split('\"\n')[0]
        else:
            exit("unexpected start of helpful")
    except:
        # print(cnt, "has an empty helpful section")
        helpful = ''
    return helpful


def get_recommend(recommend):
    try:
        # recommend = line.split("'recommend': ")[1]
        recommend = recommend.split(",")[0]
        if recommend == 'True':
            recommend = 1
        else:
            recommend = 0
    except:
        # print(cnt, "has an empty recommend section")
        recommend = ''
    return recommend


def complete_game_list_with_word2vec():
    from src.vectorize import EmbeddingsModel
    my_model = EmbeddingsModel()
    direc = "saved_models/"
    checkpoint = "gensim_model_pretrainedGoogleNews-vectors-negative300.bin.model"
    my_model.load_model(direc + checkpoint)


def get_review(line, cnt):
    helpfuls = line.split("'reviews': ")[1].split("'helpful': ")[1:]
    reviews = line.split("'reviews': ")[1].split("'review': ")[1:]
    funnys = line.split("'reviews': ")[1].split("'funny': ")[1:]
    recommends = line.split("'reviews': ")[1].split("'recommend': ")[1:]
    revs, polars, positivs, subjs, funs, helps, recoms = [], [], [], [], [], [], []
    for review, fun, recomm, help in zip(reviews, funnys, recommends, helpfuls):
        try:
            if review[0] == "'":
                review = review.split("\'\n")[0]
                review = review[1:]
            elif review[0] == '"':
                review = review.split('\"\n')[0]
                review = review[1:]
            else:
                exit("unexpected start of review")
        except:
            # print(cnt, "has an empty review section")
            review = ''
        revs += [review]
        polars += [polarity(review)]
        positivs += [positive(review)]
        subjs += [subjectivity(review)]
        funs += [get_funny(fun)]
        helps += [get_helpful(help)]
        recoms += [get_recommend(recomm)]
    return revs, polars, positivs, subjs, funs, helps, recoms


def build_dataframe(json_fname="australian_user_reviews", csv_fname="steam_australia", get_more_infos=False):
    """
    Builds the dataframe to do cool data science with
    :param json_fname: json file containing the information to scrape for infos
    :param csv_fname: name of the csv filename that will contain the dataframe
    :param get_more_infos: Get more infos?
    :return:
    """
    print("Building the dataframe in extracted/dataframes...")
    filepath = 'data/{}.json'.format(json_fname)
    w = open("extracted/lists/{}.txt".format(json_fname), "w+")
    reviews = []
    ids = []
    funnys = []
    helpfuls = []
    recommends = []
    polarities = []
    positives = []
    subjectivities = []
    n_reviews_per_user = []
    n_positive_reviews_per_user = []
    n_negative_reviews_per_user = []
    n_reviews_per_user_with_at_least_1neg = []
    n_reviews_per_user_with_at_least_1pos = []
    n_positive_reviews_per_user_with_at_least_1neg = []
    n_negative_reviews_per_user_with_at_least_1neg = []
    n_positive_reviews_per_user_with_at_least_1pos = []
    n_negative_reviews_per_user_with_at_least_1pos = []
    with tqdm(total=25799) as pbar:
        with open(filepath) as fp:
            for cnt, line in enumerate(fp):
                line = jsbeautifier.beautify(line)
                revs, polars, positivs, subjs, funs, helps, recoms = get_review(line, cnt)
                reviews.extend(revs)
                polarities.extend(polars)
                positives.extend(positivs)
                subjectivities.extend(subjs)
                ids.extend([get_user_ids(line) for _ in range(len(revs))])
                funnys.extend(funs)
                helpfuls.extend(helps)
                recommends.extend(recoms)
                n_reviews_per_user += [len(revs)]
                if get_more_infos:
                    if len([x for x in recoms if str(x) == "0"]) > 0:
                        n_reviews_per_user_with_at_least_1neg += [len(revs)]
                        n_negative_reviews_per_user_with_at_least_1neg += [len([x for x in recoms if str(x) == "0"])]
                        n_positive_reviews_per_user_with_at_least_1neg += [len([x for x in recoms if str(x) == "1"])]
                    if len([x for x in recoms if str(x) == "1"]) > 0:
                        n_reviews_per_user_with_at_least_1pos += [len(revs)]
                        n_positive_reviews_per_user_with_at_least_1pos += [len([x for x in recoms if str(x) == "1"])]
                        n_negative_reviews_per_user_with_at_least_1pos += [len([x for x in recoms if str(x) == "0"])]
                n_positive_reviews_per_user += [len([x for x in recoms if str(x) == "1"])]
                n_negative_reviews_per_user += [len([x for x in recoms if str(x) == "0"])]
                w.write(reviews[-1] + '\n')
                pbar.update(1)
    w.close()
    n_positive_reviews_per_user = np.array(n_positive_reviews_per_user)
    n_negative_reviews_per_user = np.array(n_negative_reviews_per_user)
    n_negative_reviews_per_user_with_at_least_1pos = np.array(n_negative_reviews_per_user_with_at_least_1pos)
    n_negative_reviews_per_user_with_at_least_1neg = np.array(n_negative_reviews_per_user_with_at_least_1neg)
    n_reviews_per_user = np.array(n_reviews_per_user)
    plt.boxplot(n_reviews_per_user)
    plt.savefig("images/boxplot_reviews_per_id")
    plt.boxplot([n_positive_reviews_per_user, n_negative_reviews_per_user])
    plt.savefig("images/boxplot_negpos_reviews_per_id")
    plt.boxplot([n_negative_reviews_per_user_with_at_least_1pos, n_negative_reviews_per_user_with_at_least_1neg])
    plt.savefig("images/boxplot_negpos_reviews_per_id")
    print("Max n_reviews for 1 user:", np.max(n_reviews_per_user))
    print("Max n positive reviews for 1 user:", np.max(n_positive_reviews_per_user),
          "/", n_reviews_per_user[np.argmax(n_positive_reviews_per_user)])
    print("Max n negative reviews for 1 user:", np.max(n_negative_reviews_per_user),
          "/", n_reviews_per_user[np.argmax(n_negative_reviews_per_user)])

    print("Mean n_reviews for 1 user:", np.mean(n_reviews_per_user))
    print("Mean n positive reviews for 1 user:", np.mean(n_positive_reviews_per_user))
    print("Mean n negative reviews for 1 user:", np.mean(n_negative_reviews_per_user))

    print("Median n_reviews for 1 user:", np.median(n_reviews_per_user))
    print("Median n positive reviews for 1 user:", np.median(n_positive_reviews_per_user))
    print("Median n negative reviews for 1 user:", np.median(n_negative_reviews_per_user))

    if get_more_infos:
        print("\n\nWITH at least 1 positive review\n\n")
        print("Number of reviewers", len(n_reviews_per_user_with_at_least_1pos), "/25799")
        print("Max n_reviews for 1 user:", np.max(n_reviews_per_user_with_at_least_1pos))
        print("Max n positive reviews for 1 user:", np.max(n_positive_reviews_per_user_with_at_least_1pos),
              "/", n_reviews_per_user[np.argmax(n_positive_reviews_per_user_with_at_least_1pos)])
        print("Max n negative reviews for 1 user:", np.max(n_negative_reviews_per_user_with_at_least_1pos),
              "/", n_reviews_per_user[np.argmax(n_negative_reviews_per_user_with_at_least_1pos)])

        print("Mean n_reviews for 1 user:", np.mean(n_reviews_per_user_with_at_least_1pos))
        print("Mean n positive reviews for 1 user:", np.mean(n_positive_reviews_per_user_with_at_least_1pos))
        print("Mean n negative reviews for 1 user:", np.mean(n_negative_reviews_per_user_with_at_least_1pos))

        print("Median n_reviews for 1 user:", np.median(n_reviews_per_user_with_at_least_1pos))
        print("Median n positive reviews for 1 user:", np.median(n_positive_reviews_per_user_with_at_least_1pos))
        print("Median n negative reviews for 1 user:", np.median(n_negative_reviews_per_user_with_at_least_1pos))

        print("\n\nWITH at least 1 negative review\n\n")
        print("Number of reviewers", len(n_reviews_per_user_with_at_least_1neg),"/25799")
        print("Max n_reviews for 1 user:", np.max(n_reviews_per_user_with_at_least_1neg))
        print("Max n positive reviews for 1 user:", np.max(n_positive_reviews_per_user_with_at_least_1neg),
              "/", n_reviews_per_user[np.argmax(n_positive_reviews_per_user_with_at_least_1neg)])
        print("Max n negative reviews for 1 user:", np.max(n_negative_reviews_per_user_with_at_least_1neg),
              "/", n_reviews_per_user[np.argmax(n_negative_reviews_per_user_with_at_least_1neg)])

        print("Mean n_reviews for 1 user:", np.mean(n_reviews_per_user_with_at_least_1neg))
        print("Mean n positive reviews for 1 user:", np.mean(n_positive_reviews_per_user_with_at_least_1neg))
        print("Mean n negative reviews for 1 user:", np.mean(n_negative_reviews_per_user_with_at_least_1neg))

        print("Median n_reviews for 1 user:", np.median(n_reviews_per_user_with_at_least_1neg))
        print("Median n positive reviews for 1 user:", np.median(n_positive_reviews_per_user_with_at_least_1neg))
        print("Median n negative reviews for 1 user:", np.median(n_negative_reviews_per_user_with_at_least_1neg))



    ids = np.array(ids)
    funnys = np.array(funnys)
    helpfuls = np.array(helpfuls)
    recommends = np.array(recommends)
    reviews_arr = np.array(reviews, dtype='<U32')
    polarities = np.array(polarities)
    positives = np.array(positives)
    subjectivities = np.array(subjectivities)
    df = pd.DataFrame(columns=["id",
                               "funny",
                               "helpful",
                               "recommend",
                               "polarity",
                               "positive",
                               "subjectivity",
                               "review"],
                      data=np.concatenate((ids.reshape(-1, 1),
                                           funnys.reshape(-1, 1),
                                           helpfuls.reshape(-1, 1),
                                           recommends.reshape(-1, 1),
                                           polarities.reshape(-1, 1),
                                           positives.reshape(-1, 1),
                                           subjectivities.reshape(-1, 1),
                                           reviews_arr.reshape(-1, 1)), 1))
    df["review"] = reviews
    df.to_csv("extracted/dataframes/{}.csv".format(csv_fname), index=False)
    print()


def normalize_csv(df_name="steam_australia.csv", norm_fname="steam_australia_norm", spell_check=True, correct=False):
    """
    Care must be taken to remove the lowercases at the good time or it can ruin the normalization of named entities
    1- Usual text normalization. Lowercase is applied here only!
    2- names of games are turned into Capitalized version and underscored version (e.g. Team_Fortress_2)
    3- Words composed of more than one word are in word2vec are replace with the Capitalized and underscored version (e.g. New_York)
    """
    print("Normalizing csv...")

    names_file = "games_abbreviations.txt"
    names = pd.read_csv("extracted/lists/" + names_file, names=["names", "abbr"])
    df = pd.read_csv("extracted/dataframes/" + df_name)
    w = open("extracted/lists/normalized_names.csv", "w+")
    words_dict = None
    ss = None
    if spell_check:
        print("Normalizing text...")
        if correct:
            ss = SymSpell(max_edit_distance=2)
            vocabulary = "data/english_words_479k.txt"
            with open(vocabulary) as f:
                words = f.readlines()
            eng_words = [word.strip() for word in words]
            print('Creating dictionary for symspell')
            _ = ss.create_dictionary_from_arr(eng_words, token_pattern=r'.+')
            words_dict = {k: 0 for k in eng_words}
        with tqdm(total=len(df)) as pbar:
            for l, line in enumerate(df["review"]):
                # Stop words might need to be removed later if a classification task is done.
                # We need them for normalize_csv_for_word2vec, which is done after normalization
                # to impove the number of hits in that next step. Stop words are part of expressions of named
                # entities, e.g. The_Hague is a city
                line = str(line).lower().translate(str.maketrans('', '', string.punctuation))
                line = preprocessing(line, stop_words, ss, words_dict, remove_stopwords=False)
                df.set_value(l, "review", line)
                pbar.update(1)
    with tqdm(total=len(df["review"])) as pbar:
        for l, line in enumerate(df["review"]):
            # line = str(line).lower()
            for i, (name, abbr) in enumerate((zip(names["names"], names["abbr"]))):
                if name.lower() in line:
                    # Replace the name, lowercase or not, not Capitalized
                    line = line.replace(name.lower(), name.replace(" ", "_"))
                    w.write(name.replace(" ", "_") + ',"' + line + '"\n')
                # if abbr.lower() in line and len(abbr) > 1:
                # Replace the abbreviations, lowercase or not with Capitalized full name
                #    line = line.replace(abbr.lower(), name.replace(" ", "_"))
                #    w.write(name.replace(" ", "_") + ',"' + line + '"\n')

                df.set_value(l, "review", line)
            pbar.update(1)

    w.close()
    df.to_csv("extracted/dataframes/{}.csv".format(norm_fname), index=False)


def remove_stop_words_from_list(list_vocab):
    # Do not remove stop words in "composed" words like New_York
    print("Removing stopwords from vocabulary...")
    cleared_list = []
    stopwords_removed = []
    list_vocab = list(np.array(list_vocab).reshape(-1))
    with tqdm(total=len(list_vocab)) as pbar:
        for vocab in list_vocab:
            if len(remove_stopwords(str(vocab).lower())) > 0:
                cleared_list += [str(vocab)]
            else:
                stopwords_removed += [str(vocab)]
            pbar.update(1)
    print("Removed", len(stopwords_removed), "stopwords")
    assert len(stopwords_removed) + len(cleared_list) == len(list_vocab)
    return cleared_list, stopwords_removed


def normalize_csv_for_word2vec(df_name="steam_australia_norm.csv", w2v_norm_fname="steam_australia_norm_w2v"):
    """
    word2vec has a lot of vocabulary that cannot be detected in the sentences we will use to fine-tune word2vec

    :param df_name:
    :param w2v_norm_fname:
    :return:
    """
    print("Normalizing name entities in accordance to the word2vec vocabulary")
    vocab = "extracted/lists/GoogleNews-vectors-negative300"
    df = pd.read_csv("extracted/dataframes/" + df_name)
    voc = np.array(pd.read_table(vocab))
    voc, stopwords = remove_stop_words_from_list(voc)
    w = open("extracted/lists/normalized_words_w2v.txt", "w+")

    print("Normalisation for pretrained word2vec started. This will take a while...")
    with tqdm(total=len(voc)) as pbar:
        for word in voc:
            word = str(word.strip("\n"))
            if (  # and len(word.split("_")) > 1
                    "_".join([w.lower().capitalize() for w in word.split("_")]) == word or "_".join(
                [w.lower() for w in word.split("_")]) == word):
                if word != word.lower():
                    if len(word) > 1 and "" not in word.split("_"):
                        for l, line in enumerate(df["review"]):
                            line = str(line)
                            # The spaces ensure it is tow complete words, not the end of one and begining of the other
                            expr = " ".join(word.lower().split("_"))
                            if " " + expr + " " in line:
                                # Replace the name, lowercase or not, not Capitalized
                                line = line.replace(" " + expr + " ", " " + word + " ")
                                w.write(word + '\n')
                                df.set_value(l, "review", line)
                            # start of sentence
                            if expr + " " in line and line.split(expr)[0] == "":
                                # Replace the name, lowercase or not, not Capitalized
                                line = line.replace(expr + " ", word + " ")
                                w.write(word + '\n')
                                df.set_value(l, "review", line)
                            # end of sentence
                            if " " + expr in line and line.split(expr)[-1] == "":
                                # Replace the name, lowercase or not, not Capitalized
                                line = line.replace(" " + expr, " " + word)
                                w.write(word + '\n')
                                df.set_value(l, "review", line)
            pbar.update(1)
    w.close()
    df.to_csv("extracted/dataframes/{}.csv".format(w2v_norm_fname), index=False)


if __name__ == "__main__":
    look_for_words()
    unique_words()
    words_abbreviation()
    count_words()
    build_dataframe(get_more_infos=True)
    normalize_csv(spell_check=True, correct=False)
    get_word2vec_vocab()
    normalize_csv_for_word2vec()
