import logging
import warnings

from pattern.text.en import singularize, pluralize, comparative, superlative, lemma, lexeme, tenses, suggest, wordnet

warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class EmbeddingsModel:

    def __init__(self):
        self.sentences = None
        self.model = None

    def write_annotations_file(self):

        def preprocess_and_split_line(line):
            line = line.replace("\n", "")
            line = line.strip()
            columns = line.split("\t")
            word = columns[0]
            count = columns[1]

            synonyms = []
            if int(count) != 0:
                synonyms = columns[2].split(" ")

            return word, count, synonyms

        def write_hypo(parent, count, list_of_neighbors):

            return_dict = {}

            for index in range(0, len(list_of_neighbors)):
                s = wordnet.synsets(list_of_neighbors[index])
                if len(s) > 0:
                    s = s[0]

                    synomyms = s.synonyms
                    hypernyms = s.hypernyms()
                    hyponyms = s.hyponyms()
                    holonyms = s.holonyms()
                    meronyms = s.meronyms()
                    singulars = [singularize(list_of_neighbors[index])]
                    plurals = [pluralize(list_of_neighbors[index])]
                    comparatives = [comparative(list_of_neighbors[index])]
                    superlatives = [superlative(list_of_neighbors[index])]
                    lemmas = [lemma(list_of_neighbors[index])]
                    lexemes = [lexeme(list_of_neighbors[index])]
                    tensess = [tenses(list_of_neighbors[index])]
                    suggests = [suggest(list_of_neighbors[index])]

                    neighbors_with_link_string = None

                    if parent in synomyms:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[SYNO]"
                    elif parent in hypernyms:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[HYPER]"
                    elif parent in hyponyms:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[HYPO]"
                    elif parent in holonyms:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[HOLO]"
                    elif parent in meronyms:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[MERO]"
                    elif parent in singulars:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[PLURAL]"
                    elif parent in plurals:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[SINGULAR]"
                    elif parent in comparatives:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[COMPA]"
                    elif parent in superlatives:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[SUPERLA]"
                    elif parent in lemmas:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[LEMMA]"
                    elif parent in lexemes:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[LEXEME]"
                    elif parent in tensess:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[TENSE]"
                    elif parent in suggests:
                        neighbors_with_link_string = str(list_of_neighbors[index]) + "[MISPELL]"

                    if neighbors_with_link_string:
                        try:
                            return_dict[word][1].append(neighbors_with_link_string)
                        except:
                            return_dict[word] = (count, [neighbors_with_link_string])
            return return_dict

        f_in = open('voisins.txt', "r")
        f_out = open("annotations.txt", "w")
        for line in f_in:
            word, count, synomyms = preprocess_and_split_line(line)
            dict = write_hypo(word, count, synomyms)

            if dict:
                for key in dict.keys():
                    formated_string = key + "\t" + str(dict[key][0]) + "\t"

                    for word in dict[key][1]:
                        formated_string += (str(word) + " ")

                    formated_string += "\n"

                    f_out.write(formated_string)
        f_in.close()
        f_out.close()


if __name__ == '__main__':
    my_model = EmbeddingsModel()

    my_model.write_annotations_file()
