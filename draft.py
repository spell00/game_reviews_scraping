import jsbeautifier
import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_names():
    pass


def find_all_similar_games():
    """
    Finds games that are similar
    strategies:

        1- Look for games that were mentionned in the same comment (often means it is compared one way or another)
        2- Look for games that have very similar characteristics (like a recommendation system)
        3- Find if all the games that were mentionned together actually have similar characteristics
           (especially same type)

    :return:
    """


def descriptive_stats_raters():
    """
    Finds the distribution of number of reviews per reviewer. Try poisson distribution
    :return:
    """
    pass


def create_labels_recommended():
    """
    Creates a .csv with the first column being the json item and second column if the person recommended or not
    :return:
    """
    pass


def find_games_types():
    pass


def find_games_types_descritors():
    pass


def are_players_haters():
    """
    Function to test if negative reviews come from haters or they also like things
    :return:
    """
    pass


def rating_and_time():
    """
    Function to test if the rating of a game increase of decline through time
    :return:
    """
    pass


def is_positive():
    pass


def game_ontology(games):
    pass


def find_characteristics(games):
    """
    Characteristics:
        1- multiplayer
        2- gameplay
        3- graphics
        4- story
        5- issues
        6- characters
        7- features
        8- soundtracks
    :param games:
    :return:
    """
    pass


def find_characteristics_infos():
    """
    Characteristics:
        1- multiplayer
        2- gameplay
        3- graphics
        4- story
        5- issues
        6- characters
        7- features
        8- soundtracks
    :return:
    """
    pass


def predict_game_appreciation():
    pass


def look_for_games():
    filepath = 'australian_user_reviews.txt'
    names = "games_abbreviations.txt"
    games_dict = {}
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            # if cnt % 1000 == 0:
            #    print(cnt)
            with open(names) as n:
                for i, name in enumerate(n):
                    name = name.strip("\n")
                    name, abbr = name.split(",")
                    if name in line or (abbr in line and len(abbr) > 1):
                        if name not in games_dict:
                            games_dict.update({name: [cnt]})
                        else:
                            games_dict[name] += [cnt]
    w = open("game_lines.csv", "w+")
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
def is_a_game():
    filepath = 'australian_user_reviews.txt'
    w = open("game_names.txt", "w+")
    w2 = open("game_names_rejected_for_length.txt", "w+")
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            line_split = line.split(" is a ")
            if len(line_split) > 1:
                line_split1 = line_split[1].split(".")[-1].split(",")[-1].split("!")[-1].split("?")[-1].replace("'",
                                                                                                                "").replace(
                    '"', "").replace("-", "").replace("_", "").replace("*", "")
                line_split = line_split[0].split(".")[-1].split(",")[-1].split("!")[-1].split("?")[-1].replace("'",
                                                                                                               "").replace(
                    '"', "").replace("-", "").replace("_", "").replace("*", "")
                if line_split is "":
                    continue
                while line_split[-1] == " ":
                    line_split = line_split[:-1]
                while line_split[0] == " ":
                    line_split = line_split[1:]
                if line_split not in ["There", "there", "This", "this", "", "it", "It"] and (
                        "game" in line_split1.split(" ")):
                    flag = False
                    for word in line_split.split(" "):
                        if word in ["This", "this", "", "it", "It"]:
                            flag = True
                    if flag is False:
                        if len(line_split.split(" ")) <= 5:
                            line_split = " ".join([l.capitalize() for l in line_split.split(" ")])
                            w.write(line_split + '\n')
                        else:
                            w2.write(line_split + '\n')

    w.close()


def unique_games():
    filepath = 'game_names.txt'
    gamenames = {}
    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            if line not in gamenames:
                gamenames.update({line: 0})
            else:
                gamenames[line] += 1
    w = open("games_unique.txt", "w+")
    for game in gamenames.keys():
        w.write("{}".format(game))
    w.close()


def games_abbreviation():
    filepath = 'games_unique.txt'
    gamenames = {}
    with open(filepath) as fp:
        for cnt, game in enumerate(fp):
            game = game.split("\n")[0]
            abb = "".join(''.join(x for x in game.replace("'", "") if not x.islower()).split(" "))
            if game not in gamenames:
                gamenames.update({game: abb})
            else:
                exit("whoops")
    w = open("games_abbreviations.txt", "w+")
    for game in gamenames.keys():
        w.write("{},{}\n".format(game, gamenames[game]))
    w.close()


def get_user_ids(line, cnt):
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


def get_funny(line, cnt):
    try:
        funny = line.split("'funny': ")[1]
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


def get_helpful(line, cnt):
    try:
        helpful = line.split("'helpful': ")[1]
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


def get_recommend(line, cnt):
    try:
        recommend = line.split("'recommend': ")[1]
        if recommend[0] == "'":
            recommend = recommend.split(",")[0].strip("'")
        elif recommend[0] == '"':
            recommend = recommend.split('\"\n')[0]
        else:
            exit("unexpected start of recommend")
    except:
        # print(cnt, "has an empty recommend section")
        recommend = ''
    return recommend


def get_review(line, cnt):
    try:
        review = line.split("'reviews': ")[1].split("'review': ")[1]
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
    return review


def main():
    filepath = 'australian_user_reviews.json'
    w = open("australian_user_reviews.txt", "w+")
    names_file = "games_abbreviations.txt"
    names = pd.read_csv(names_file, names=["names", "abbr"])

    reviews = []
    ids = []
    funnys = []
    helpfuls = []
    recommends = []
    with tqdm(total=25799) as pbar:
        with open(filepath) as fp:
            for cnt, line in enumerate(fp):
                line = jsbeautifier.beautify(line)
                reviews += [get_review(line, cnt)]
                ids += [get_user_ids(line, cnt)]
                funnys += [get_funny(line, cnt)]
                helpfuls += [get_helpful(line, cnt)]
                recommends += [get_recommend(line, cnt)]

                w.write(reviews[-1] + '\n')
                pbar.update(1)
    w.close()
    ids = np.array(ids)
    funnys = np.array(funnys)
    helpfuls = np.array(helpfuls)
    recommends = np.array(recommends)
    reviews = np.array(reviews)
    df = pd.DataFrame(columns=["id", "funny", "helpful", "recommend", "review"],
                      data=np.concatenate((ids.reshape(-1, 1),
                                           funnys.reshape(-1, 1),
                                           helpfuls.reshape(-1, 1),
                                           recommends.reshape(-1, 1),
                                           reviews.reshape(-1, 1)), 1))
    df.to_csv("steam_australia.csv", index=False)


def normalize():
    names_file = "games_abbreviations.txt"
    w = open("australian_user_reviews_normalized.txt", "w+")

    filepath = 'australian_user_reviews.txt'
    names = pd.read_csv(names_file, names=["names", "abbr"])
    # ids = reviews["id"].to_numpy()
    # funnys = np.array(funnys)
    # helpfuls = np.array(helpfuls)
    # recommends = np.array(recommends)

    with open(filepath) as fp:
        for cnt, line in enumerate(fp):
            for i, (name, abbr) in enumerate((zip(names["names"], names["abbr"]))):
                if len(name.split(" ")) == 1:
                    continue
                if name in line:
                    line.replace(name, name.replace(" ", ""))
            w.write(line)
    w.close()

def normalize_line(line, names):
    # ids = reviews["id"].to_numpy()
    # funnys = np.array(funnys)
    # helpfuls = np.array(helpfuls)
    # recommends = np.array(recommends)

    for i, (name, abbr) in enumerate((zip(names["names"], names["abbr"]))):
        if len(name.split(" ")) == 1:
            continue
        if name in line:
            line.replace(name, name.replace(" ", ""))
    return line

# 1- Find a list of games
# 2- Find all reviews about each game
# 3? - Predict which game it is when the name of the game is not included
# 4- Find what other informations are associated with the game
# 5- Find who is more likely to have a positive review and who is not

# 6- Game ontology

# 7- Game clustering


# TEXT NORMALISATION:
#   -Replace game names by always the same thing, e.g. RL -> Rocket_League.
#       Reason: to make game names the same and only one word (to use in word2vec and find neighbors)


if __name__ == "__main__":
    main()
    normalize()
    is_a_game()
    unique_games()
    games_abbreviation()
    look_for_games()

# Terreria: Starbound
