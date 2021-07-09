#!/usr/bin/env python3
#coding:utf-8
import json
import numpy as np
from tqdm import tqdm
from nltk import tokenize
from rake_nltk import Rake
import re
import argparse


'''import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text, token.pos_, token.dep_)'''


def Plots(in_file_story, in_file_sensplit, out_file_name):


    from allennlp.predictors.predictor import Predictor
    ent_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
    conref_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
    srl_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

    with open(in_file_sensplit, "r") as f1:
        SenSplitting = json.load(f1)

    def count_line(file_name):
        with open(file_name) as f:
            for count, _ in enumerate(f, 1):
                pass
        return count

    def get_cluster_indices_list(clusters):

        cluster_indices_list = []
        for cluster in clusters:
            cluster_index_list = []
            for indices in cluster:
                if indices[0] == indices[1]:
                    cluster_index_list.append(indices[0])
                else:
                    for n in range(indices[1] - indices[0] + 1):
                        cluster_index_list.append(n + indices[0])
            cluster_indices_list.append(cluster_index_list)
        return sum(cluster_indices_list, [])


    total_story = count_line(in_file_story)
    with open(in_file_story,"r") as f:
        for n_story, story in tqdm(enumerate(f.readlines())):
            try:
                # SRL
                sens = SenSplitting[n_story]
                story_slr = []
                n_v = 0
                for sen in sens:
                    sen_srls = srl_predictor.predict(sen.replace("<newline> ", ""))
                    sen_slr = [[] for _ in sen_srls["words"]]
                    for tier in sen_srls["verbs"]:
                        for n_tag, tag in enumerate(tier["tags"]):
                            if tag != 'O':
                                sen_slr[n_tag].append(str(n_v) + tag[tag.rfind("-") + 1:])
                        n_v += 1
                    story_slr += sen_slr


                # conreference & entity detection
                conref_dict = conref_predictor.predict(story.replace("<newline> ", ""))
                ent_dict = ent_predictor.predict(story.replace("<newline> ", ""))
                clusters_indices = get_cluster_indices_list(conref_dict["clusters"])
                ent_indices = [n_ent for n_ent, ent in enumerate(ent_dict["tags"]) if
                               ent != 'O' and n_ent not in clusters_indices]

                for index in ent_indices:
                    conref_dict["clusters"].append([[index, index]])
                    conref_dict["clusters"].sort()

                for n_cluster, cluster in enumerate(conref_dict["clusters"]):
                    for conref in cluster:
                        for n_token, token in enumerate(conref_dict["document"][:-1]):
                            if n_token in range(conref[0], conref[1] + 1):
                                if story_slr[n_token] == []:
                                    conref_dict["document"][n_token] = "ent{}".format(n_cluster)
                                elif re.match(r"^\d+V$", story_slr[n_token][0]) is None:
                                    conref_dict["document"][n_token] = "ent{}".format(n_cluster)


                # key words extraction
                r = Rake()
                r.extract_keywords_from_text(story.replace("<newline> ", ""))
                key_phrases_list = r.get_ranked_phrases()
                key_words_indices = []
                for phrase in key_phrases_list:
                    word_list = list(phrase.split(" "))
                    for start in range(len(ent_dict["words"]) - len(word_list) + 1):
                        if ent_dict["words"][start:start + len(word_list)] == word_list:
                            key_words_indices += [index for index in range(start, start + len(word_list))]


                plot = []
                # exclude "\n"
                for n_word, word in enumerate(conref_dict["document"][:-1]):
                    if story_slr[n_word] != [] and \
                            (n_word in key_words_indices[:int(len(conref_dict["document"][:-1])*0.2)] or # reduce key words
                             re.match(r"^\d+V$", story_slr[n_word][0]) != None or
                             re.match(r"^ent\d+$", word) != None):
                        plot.append("<{}>".format(story_slr[n_word][0]))
                        plot.append(word)

            except:
                # conreference & entity detection
                conref_dict = conref_predictor.predict(story.replace("<newline> ", ""))
                ent_dict = ent_predictor.predict(story.replace("<newline> ", ""))
                clusters_indices = get_cluster_indices_list(conref_dict["clusters"])
                ent_indices = [n_ent for n_ent, ent in enumerate(ent_dict["tags"]) if
                               ent != 'O' and n_ent not in clusters_indices]

                for index in ent_indices:
                    conref_dict["clusters"].append([[index, index]])
                    conref_dict["clusters"].sort()

                for n_cluster, cluster in enumerate(conref_dict["clusters"]):
                    for conref in cluster:
                        for n_token, token in enumerate(conref_dict["document"][:-1]):
                            if n_token in range(conref[0], conref[1] + 1):
                                if story_slr[n_token] == []:
                                    conref_dict["document"][n_token] = "ent{}".format(n_cluster)
                                elif re.match(r"^\d+V$", story_slr[n_token][0]) is None:
                                    conref_dict["document"][n_token] = "ent{}".format(n_cluster)

                # key words extraction
                r = Rake()
                r.extract_keywords_from_text(story.replace("<newline> ", ""))
                key_phrases_list = r.get_ranked_phrases()
                key_words_indices = []
                for phrase in key_phrases_list:
                    word_list = list(phrase.split(" "))
                    for start in range(len(ent_dict["words"]) - len(word_list) + 1):
                        if ent_dict["words"][start:start + len(word_list)] == word_list:
                            key_words_indices += [index for index in range(start, start + len(word_list))]

                plot = []
                # exclude "\n"
                for n_word, word in enumerate(conref_dict["document"][:-1]):
                    if n_word in key_words_indices[:int(len(conref_dict["document"][:-1])*0.2)] or \
                            re.match(r"^ent\d+$", word) != None:
                        plot.append(word)


            with open("Dataset/Plots/{}.txt".format(out_file_name),"a") as f2:
                f2.write(" ".join(plot) + "\n")

            with open("Dataset/EntityAnonymizedStories/{}.txt".format(out_file_name),"a") as f3:
                f3.write(" ".join(conref_dict["document"]))

            print("{}/{} is Done!".format(n_story,total_story))


def SenSplitting(in_file,out_file_name):
    '''
    args: in_file: input file address, raw story
        out_file_name: output file name, e.g., "train", "valid"
    output: list of sentences, json file
            [["sentence 1.1","sentence 1.2",sentence 1.3,...],
            ["sentence 2.1","sentence 2.2", "sentence 3.3",...],
            ...]
    '''

    Sentences = []
    with open(in_file, "r") as f:
        print("Start sentence splitting")
        for line in tqdm(f.readlines()):
            Sentences.append(tokenize.sent_tokenize(line))

    with open("../Dataset/SenSplitting/{}_target_sensplit.json".format(out_file_name), "w") as w:
        json.dump(Sentences,w)




def SenEmbeding(in_file,out_file_name):
    '''
    args: in_file: input file address, json file with sentences split in each story
        out_file_name: output file name, e.g., "train", "valid"
    output: arrays of sentence embeddings, npy file, shape of sentence embeddings (1*768)
            [[[senembedding 1.1],[senembedding 1.2],[senembedding 1.3],...,[]],
            [[senembedding 2.1],[senembedding 2.2],[senembedding 2.3],...,[]],
            ...]
    '''
    # Source code: https://github.com/UKPLab/sentence-transformers

    # import model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # load sentences-split file
    with open(in_file, "r") as f:
        Sentences = json.load(f)

    # generate sentence embeddings
    sentence_embeddings = []
    print("Start getting sentence embeddings")
    for n in tqdm(range(len(Sentences))):
        sentence_embeddings.append(model.encode(Sentences[n]))

    # save array of sentence embeddings
    np.save("../Dataset/SenEmbedding/{}.npy".format(out_file_name), sentence_embeddings)

    # load npy data
    #c = np.load("../Dataset/SenEmbedding/train.npy", allow_pickle=True)
    #print(c)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODE', type=str, choices=['Plots', 'SenSplitting', 'SenEmbeding'])
    parser.add_argument('--IN_FILE', type=str)
    parser.add_argument('--IN_FILE_SENSPLIT', type=str)
    parser.add_argument('--OUT_FILE_NAME', type=str)
    args = parser.parse_args()

    if args.MODE == 'Plots':
        Plots(args.IN_FILE, args.IN_FILE_SENSPLIT, args.OUT_FILE_NAME)
        #Plots("Dataset/WritingPrompts/smallest.wp_target", "smallest.txt.wp_target")
        #Plots("Dataset/WritingPrompts/train.wp_target", "Dataset/SenSplitting/train_target_sensplit.json", "train")
        #Plots("Dataset/WritingPrompts/valid.wp_target", "Dataset/SenSplitting/valid_target_sensplit.json", "valid")
        #Plots("Dataset/WritingPrompts/test.wp_target", "Dataset/SenSplitting/test_target_sensplit.json", "test")

    elif args.MODE == 'SenSplitting':
        SenSplitting(args.IN_FILE, args.OUT_FILE_NAME)
        #SenSplitting("../Dataset/WritingPrompts/train.wp_target","train")
        #SenSplitting("../Dataset/WritingPrompts/valid.wp_target","valid")
        #SenSplitting("../Dataset/WritingPrompts/test.wp_target","test")

    elif args.MODE == 'SenEmbeding':
        SenEmbeding(args.IN_FILE, args.OUT_FILE_NAME)
        #SenEmbeding("../Dataset/SenSplitting/train_target_sensplit.json", "train")
        #SenEmbeding("../Dataset/SenSplitting/valid_target_sensplit.json", "valid")
        #SenEmbeding("Dataset/SenSplitting/test_target_sensplit.json", "test")




