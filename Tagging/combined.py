from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, Bidirectional, MaxPooling1D,\
    Flatten, concatenate, LSTM
from keras.initializers import RandomUniform
from keras.optimizers import Adam
import numpy as np
import requests
import json
import re
from common import prepare_corpus, prepare_dict_simple, prepare_sets, create_lstm, create_conv, tag_text_combined, \
    predict, create_lstm2



def train(conv_model_name="../Models/model_conv_c32e8.hf5", bilstm_model_name="../Models/model_lstm_c32e8.hf5",
          corpus_path="http://localhost:8983/solr/mikes", excluded_file="exluded.txt"):
    excluded = []
    file = open(excluded_file, encoding="utf-8", mode="r")
    for w in file:
        excluded.append(w.strip("\n"))
    file.close()

    word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag, example_array, excluded, test_array \
        = prepare_corpus(corpus_path+"/select", 0.0, excluded)

    d, l = prepare_dict_simple(corpus_path+"/select")

    model = create_conv(word_len, char_len, vec2tag, conv_model_name)
    for i in range(32):
        (train_xw, valid_xw, test_xw), (train_xc, valid_xc, test_xc), (train_y, valid_y, test_y), _ = \
            prepare_sets(word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag, example_array,
                         (0.95, 0.0), (i*10000, (i+1)*10000))
    
        model.fit([train_xw, train_xc], train_y, batch_size=128, epochs=8, validation_split=0.05)
    
        scores = model.evaluate([test_xw, test_xc], test_y)
        model.save_weights('model_conv'+str(i)+'.hf5')
    
        print(scores)

    model1 = create_lstm(word_len, char_len, vec2tag, bilstm_model_name)
    for i in range(32):
    
        (train_xw, valid_xw, test_xw), (train_xc, valid_xc, test_xc), (train_y, valid_y, test_y), _ = \
            prepare_sets(word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag, example_array,
                         (0.95, 0.0), (i*10000, (i+1)*10000))
        model1.fit([train_xw, train_xc], train_y, batch_size=128, epochs=8, validation_split=0.05)
    
        scores = model1.evaluate([test_xw, test_xc], test_y)
        model1.save_weights('model_lstm'+str(i)+'.hf5')
    
        print(scores)

    (_, _, test_xw), (_, _, test_xc), (_, _, test_y), (_, _, test_sw) = \
        prepare_sets(word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag, test_array, (0.0, 0.0),
                     (0, len(test_array)))
    
    scores = model.evaluate([test_xw, test_xc], test_y)
    print(scores)
    scores = model1.evaluate([test_xw, test_xc], test_y)
    print(scores)
    hits = 0
    pred = model.predict([test_xw, test_xc])
    pred1 = model1.predict([test_xw, test_xc])
    bywho = [0, 0, 0, 0, 0, 0, 0] # abc, ab, ac, bc, a, b, c
    for i in range(len(test_array)):
        if i % 100 == 1:
            print(str(i) + ". sentence, " + str(hits*100/i) + " % acc., ")
        pred2 = predict(test_sw[i], d, l)
        if pred[i].argmax(axis=-1) == pred1[i].argmax(axis=-1):
            chosen = vec2tag[pred[i].argmax(axis=-1)]
        else:
            chosen = pred2
        if chosen == test_array[i][1]:
            hits += 1

        if test_array[i][1] == vec2tag[pred[i].argmax(axis=-1)]:
            if test_array[i][1] == vec2tag[pred1[i].argmax(axis=-1)]:
                if test_array[i][1] == pred2:
                    bywho[0] += 1
                else:
                    bywho[1] += 1
            elif test_array[i][1] == pred2:
                bywho[2] += 1
            else:
                bywho[4] += 1
        elif test_array[i][1] == vec2tag[pred1[i].argmax(axis=-1)]:
            if test_array[i][1] == pred2:
                bywho[3] += 1
            else:
                bywho[5] += 1
        elif test_array[i][1] == pred2:
            bywho[6] += 1
    print(hits / len(test_array))
    print(bywho)
    return


def parse(text, conv_model_name="../Models/model_conv_c32e8.hf5", bilstm_model_name="../Models/model_lstm_c32e8.hf5",
          corpus_path="http://localhost:8983/solr/mikes", excluded_file="exluded.txt"):
    excluded = []
    file = open(excluded_file, encoding="utf-8", mode="r")
    for w in file:
        excluded.append(w.strip("\n"))
    file.close()

    word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag, example_array, excluded, test_array \
        = prepare_corpus(corpus_path+"/select", 0.0, excluded)

    d, l = prepare_dict_simple(corpus_path+"/select")
    model = create_conv(word_len, char_len, vec2tag, conv_model_name)
    model1 = create_lstm(word_len, char_len, vec2tag, bilstm_model_name)

    pred = tag_text_combined(text, model, model1, d, l, (word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag))
    result = ""
    preword = ""
    pretag = ""
    f = True
    for w, p in pred:
        word = w
        query = {"q": "b:" + word, "fl": "u b"}
        response = requests.get(corpus_path+"/select", params=query)
        jstr = json.loads(response.text)
        lut = set()
        for u in jstr["response"]["docs"]:
            if word in u["b"]:
                lut.add(u["u"][0])
        if len(lut) > 0:
            lemmas = ""
            for lemma in lut:
                if lemma == p:
                    lemmas += " !U:{0}".format(lemma)
                else:
                    lemmas += " U:{0}".format(lemma)
            if not f:
                result += " <{0} ?U:{1}>".format(preword, pretag)
            result += " <" + word + lemmas + ">"
            f = True
        else:
            if f:
                f = False
            else:
                query = {"q": "b:" + preword + " " + word, "fl": "u b"}
                response = requests.get(corpus_path+"/select", params=query)
                jstr = json.loads(response.text)
                lut = set()
                for u in jstr["response"]["docs"]:
                    if preword + " " + word in u["b"]:
                        lut.add(u["u"][0])
                if len(lut) > 0:
                    lemmas = ""
                    for lemma in lut:
                        lemmas += " U:{0}".format(lemma)
                    result += "<" + preword + " " + word + lemmas + ">"
                    f = True
                else:
                    result += " <{0} ?U:{1}>".format(preword, pretag)
                    f = False
        preword = word
        pretag = p
    return result

