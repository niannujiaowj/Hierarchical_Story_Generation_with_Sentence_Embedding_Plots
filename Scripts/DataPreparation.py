#!/usr/bin/env python3
#coding:utf-8
import json
import os
import numpy as np
from tqdm import tqdm
from nltk import tokenize
from rake_nltk import Rake
import re
import argparse
from Scripts.util import *


'''import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text, token.pos_, token.dep_)'''


def Plots(in_file_story, in_file_sensplit, out_file_name):
    #open("./tmp/Plots/{}".format(out_file_name), 'w').close()
    #open("./tmp/EntityAnonymizedStories/{}".format(out_file_name), 'w').close()

    from allennlp.predictors.predictor import Predictor
    ent_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
    conref_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
    srl_predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    try:
        ent_predictor._model = ent_predictor._model.cuda()
        conref_predictor._model = conref_predictor._model.cuda()
        srl_predictor._model = srl_predictor._model.cuda()
    except:
        pass

    #tmp_sensplit_file = get_file_from_google_cloud_storage(in_file_sensplit)
    with open(in_file_sensplit, "r") as f1:
        SenSplitting = json.load(f1)[135499:135500]

    #tmp_story_file = get_file_from_google_cloud_storage(in_file_story)
    total_story = count_line(in_file_story)
    with open(in_file_story,"r") as f:
        for n_story, story in tqdm(enumerate(f.readlines()[135499:135500])):
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
                            if re.findall(r"ARG[\d\w]*", tag) != []:
                                sen_slr[n_tag].append(str(n_v) + re.findall(r"ARG[\d\w]*", tag)[0])
                            elif re.match(r"\w+-V", tag):
                                sen_slr[n_tag].append(str(n_v) + "V")
                            else:
                                pass
                        n_v += 1
                    story_slr += sen_slr
                print("story_slr = ",story_slr )

                # conreference & entity detection
                conref_dict = conref_predictor.predict(story.replace("<newline> ", ""))
                ent_dict = ent_predictor.predict(story.replace("<newline> ", ""))
                clusters_indices = get_cluster_indices_list(conref_dict["clusters"])
                ent_indices = [n_ent for n_ent, ent in enumerate(ent_dict["tags"]) if
                               ent != 'O' and n_ent not in clusters_indices]

                for index in ent_indices:
                    conref_dict["clusters"].append([[index, index]])
                    conref_dict["clusters"].sort()

                print("conref_dict[clusters] = ",conref_dict["clusters"])

                for n_cluster, cluster in enumerate(conref_dict["clusters"]):
                    for conref in cluster:
                        for n_token, token in enumerate(conref_dict["document"][:-1]):
                            if n_token in range(conref[0], conref[1] + 1):
                                if story_slr[n_token] == []:
                                    conref_dict["document"][n_token] = "ent{}".format(n_cluster)
                                elif re.match(r"^\d+V$", story_slr[n_token][0]) is None:
                                    conref_dict["document"][n_token] = "ent{}".format(n_cluster)
                print("conref_dict[document] = ", conref_dict["document"])

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

                print("key_words_indices = ",key_words_indices)

                plot = []
                EntityAnonymizedStory = []
                # exclude "\n"
                for n_word, word in enumerate(conref_dict["document"][:-1]):
                    try: # cut off the same ent group
                        if word == conref_dict["document"][n_word-1]:
                            same_group_label = "True"
                        else:
                            same_group_label = "False"
                            EntityAnonymizedStory.append(word)
                    except:
                        same_group_label = "False"
                        EntityAnonymizedStory.append(word)
                    if story_slr[n_word] != [] and \
                            same_group_label == "False" and \
                            (n_word in key_words_indices[:int(len(conref_dict["document"][:-1])*0.2)] or # reduce key words
                             re.match(r"^\d+V$", story_slr[n_word][0]) != None or
                             re.match(r"^ent\d+$", word) != None):
                        for n in range(len(story_slr[n_word])):
                            plot.append("<{}>".format(story_slr[n_word][n]))
                        plot.append(word)
                print("plot = ", plot)
                print("EntityAnonymizedStory = ",EntityAnonymizedStory)

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
                EntityAnonymizedStory = []
                # exclude "\n"
                for n_word, word in enumerate(conref_dict["document"][:-1]):
                    try: # cut off the same ent group
                        if word == conref_dict["document"][n_word-1]:
                            same_group_label = "True"
                        else:
                            same_group_label = "False"
                            EntityAnonymizedStory.append(word)
                    except:
                        same_group_label = "False"
                        EntityAnonymizedStory.append(word)
                    if same_group_label == "False" and \
                            (n_word in key_words_indices[:int(len(conref_dict["document"][:-1])*0.2)] or
                            re.match(r"^ent\d+$", word) != None):
                        plot.append(word)

            if not os.path.exists("Dataset/Plots"):
                os.makedirs("Dataset/Plots")
            with open("Dataset/Plots/{}".format(out_file_name),"a") as f2:
                f2.write(" ".join(plot) + "\n")


            if not os.path.exists("Dataset/EntityAnonymizedStories"):
                os.makedirs("Dataset/EntityAnonymizedStories")
            with open("Dataset/EntityAnonymizedStories/{}".format(out_file_name),"a") as f3:
                f3.write(" ".join(EntityAnonymizedStory))

            print("{}/{} is Done!".format(n_story,total_story))

    #subprocess.check_call(
    #    ['gsutil', 'cp', "./tmp/Plots/{}".format(out_file_name), os.path.join("gs://my_dissertation/Dataset/Plots", out_file_name)])

    #subprocess.check_call(
    #    ['gsutil', 'cp', "./tmp/EntityAnonymizedStories/{}".format(out_file_name), os.path.join("gs://my_dissertation/Dataset/EntityAnonymizedStories", out_file_name)])

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

    with open("Dataset/SenSplitting/{}_target_sensplit.json".format(out_file_name), "w") as w:
        json.dump(Sentences,w)




def SenEmbedding(in_file,out_file_name):
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
    np.save("Dataset/SenEmbedding/{}.npy".format(out_file_name), sentence_embeddings)

    # load npy data
    #c = np.load("../Dataset/SenEmbedding/train.npy", allow_pickle=True)
    #print(c)





if __name__ == '__main__':
    Plots("Dataset/WritingPrompts/train.wp_target", "Dataset/SenSplitting/train_target_sensplit.json", "train_135499_135500")
    '''parser = argparse.ArgumentParser()
    parser.add_argument('--MODE', type=str, choices=['Plots', 'SenSplitting', 'SenEmbeding'])
    parser.add_argument('--IN_FILE', type=str)
    parser.add_argument('--IN_FILE_SENSPLIT', type=str)
    parser.add_argument('--OUT_FILE', type=str)
    parser.add_argument('--OUT_FILE_NAME', type=str)
    args = parser.parse_args()

    if args.MODE == 'Plots':
        Plots(args.IN_FILE, args.IN_FILE_SENSPLIT, args.OUT_FILE_NAME)
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
        #SenEmbeding("Dataset/SenSplitting/test_target_sensplit.json", "test")'''




