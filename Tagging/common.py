from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, Bidirectional, MaxPooling1D, \
    Flatten, concatenate, LSTM
from keras.initializers import RandomUniform
from keras.optimizers import Adam
import numpy as np
import requests
import json
import re
import Levenshtein


def one_hot_encoder(n, m):
    oh = np.zeros(m)
    oh[n] = 1
    return oh


def purify_sentence(example):
    sentence = []
    b = True
    sent_array = example.split(" ")
    for word in sent_array:
        stripped = re.sub(r"[!;?,.1234567890()_:/{}…†@+„”\"<>’*\n]", "", word).lower()
        if len(stripped) > 0:
            sentence.append(stripped)
    return sentence


def encode_sentence(word2vec, char2vec, char_len, tag2vec, tag_len, sentence, tag, index):
    encoded = [0 for _ in range(15 - index)]
    encoded.extend([(word2vec[word] if word in word2vec.keys() else 0) for word in sentence])
    encoded.extend([0 for _ in range(31 - (len(sentence) + 15 - index))])
    sword = sentence[index]
    encoded[15] = 0  # exclude the word itself
    ch_encoded = [one_hot_encoder(char2vec[c], char_len) for c in sentence[index]]
    missing = 40 - len(ch_encoded)
    ch_encoded.extend([one_hot_encoder(0, char_len) for _ in range(missing)])
    label = tag2vec[tag]
    return encoded, ch_encoded, label, sword


def prepare_corpus(url, exclude_chance=0, excluded=[]):
    query = {"q": "*:*", "rows": 15843}
    response = requests.get(url, params=query)
    word_array = json.loads(response.text)["response"]["docs"]

    word_set = set()
    char_set = set()
    tag_set = set()
    example_array = []
    test_array = []
    ignored = excluded
    for word_object in word_array:
        if "example" in word_object.keys() and not word_object["u"][0][0] == "+":
            if "q" in word_object.keys():
                for q in word_object["q"]:
                    if np.random.uniform() < exclude_chance:
                        ignored.append(q)
            for example in word_object["example"]:
                sentence = []
                b = True
                temp = example.split("_")
                if len(temp) > 3:
                    continue
                elif len(temp) == 3 and ' ' in temp[1]:
                    continue
                elif len(temp) < 3:
                    continue
                sent_array = example.split(" ")[:-2]
                for word in sent_array:
                    stripped = re.sub(r"[!;?,.1234567890()_:/{}…†@+„”\"<>’*]", "", word).lower()
                    if len(stripped) > 0:
                        if "_" in word and b:
                            stripped = "_" + stripped
                            b = False
                        sentence.append(stripped)

                for i in range(len(sentence)):
                    if "_" == sentence[i][0]:
                        sentence[i] = sentence[i][1:]
                        if not sentence[i] in ignored:
                            example_array.append((sentence, word_object["u"][0], i))
                        else:
                            test_array.append((sentence, word_object["u"][0], i))
                        break

                for word in sentence:
                    word_set.add(word)
                    for char in word:
                        char_set.add(char)

            tag_set.add(word_object["u"][0])

    word2vec = {}
    char2vec = {}
    tag2vec = {}
    vec2tag = {}
    word_len = len(word_set)
    char_len = len(char_set) + 1
    tag_len = len(tag_set)
    wordlist = list(word_set)
    wordlist.sort()
    charlist = list(char_set)
    charlist.sort()
    taglist = list(tag_set)
    taglist.sort()
    for word in wordlist:
        word2vec[word] = len(word2vec) + 1  # 0 - unknown

    for char in charlist:
        char2vec[char] = len(char2vec) + 1  # 0 - padding

    for tag in taglist:
        tag2vec[tag] = len(tag2vec)
        vec2tag[len(vec2tag)] = tag
    tag2vec[""] = 0
    vec2tag[0] = ""

    example_array = np.random.permutation(example_array)

    return word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag, example_array, ignored, test_array


def prepare_sets(word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag, example_array, split, indexes):
    sentence_list = list()
    word_list = list()
    for sentence, tag, index in example_array[indexes[0]:indexes[1]]:
        w_encoded, ch_encoded, label, sword = encode_sentence(word2vec, char2vec, char_len, tag2vec, tag_len, sentence,
                                                              tag, index)
        sentence_list.append((w_encoded, ch_encoded, label))
        word_list.append(sword)

    split1 = int(len(sentence_list) * split[0])
    split2 = int(len(sentence_list) * split[0]) + int(len(sentence_list) * split[1])

    train_xw = np.array([a for a, _, _ in sentence_list[:split1]])
    valid_xw = np.array([a for a, _, _ in sentence_list[split1:split2]])
    test_xw = np.array([a for a, _, _ in sentence_list[split2:]])

    train_xc = np.array([a for _, a, _ in sentence_list[:split1]])
    valid_xc = np.array([a for _, a, _ in sentence_list[split1:split2]])
    test_xc = np.array([a for _, a, _ in sentence_list[split2:]])

    train_y = np.array([a for _, _, a in sentence_list[:split1]])
    valid_y = np.array([a for _, _, a in sentence_list[split1:split2]])
    test_y = np.array([a for _, _, a in sentence_list[split2:]])

    train_sw = word_list[:split1]
    valid_sw = word_list[split1:split2]
    test_sw = word_list[split2:]

    return (train_xw, valid_xw, test_xw), (train_xc, valid_xc, test_xc), (train_y, valid_y, test_y), (
    train_sw, valid_sw, test_sw)


def tag_text(text, model, enc_data):
    word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag = enc_data
    p_text = purify_sentence(text)
    e_words = []
    e_chars = []
    for i in range(len(p_text)):
        sentence = p_text[max(0, i - 10):min(i + 10, len(p_text) - 1)]
        word_x, char_x, _, _ = \
            encode_sentence(word2vec, char2vec, char_len, tag2vec, tag_len, sentence, "",
                            min(i - max(0, i - 10), len(sentence) - 1))
        e_words.append(word_x)
        # print(word_x)
        e_chars.append(char_x)
    pred = model.predict([np.array(e_words), np.array(e_chars)])
    pred = pred.argmax(axis=-1)
    return [p_text[i] + "[" + vec2tag[pred[i]] + "]" for i in range(len(pred))]


def create_conv(word_len, char_len, vec2tag, file=None):
    words_input = Input(shape=(31,), dtype='int32', name='words_input')
    words_embendding = Embedding(word_len, 128)(words_input)
    conv1d_out1 = Conv1D(20, 3, activation='relu')(words_embendding)
    maxpool_out1 = MaxPooling1D(20)(conv1d_out1)
    words = Flatten()(maxpool_out1)

    char_input = Input(shape=(40, char_len,), name='char_input')
    conv1d_out = Conv1D(30, 3, activation='relu')(char_input)
    maxpool_out = MaxPooling1D(30)(conv1d_out)
    char = Flatten()(maxpool_out)

    output = concatenate([words, char])
    output = Dense(512, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(4096, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(len(vec2tag), activation="softmax")(output)

    model = Model(inputs=[words_input, char_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="nadam",
                  metrics=["accuracy"])

    model.summary()
    if file is not None:
        model.load_weights(file)
    return model


def create_lstm(word_len, char_len, vec2tag, file=None):
    words_input = Input(shape=(31,), dtype='int32', name='words_input')
    words_embendding = Embedding(word_len, 128)(words_input)
    lstm_out1 = Bidirectional(LSTM(32))(words_embendding)
    words = Dense(64)(lstm_out1)

    char_input = Input(shape=(40, char_len,), name='char_input')
    conv1d_out = Conv1D(30, 3, activation='relu')(char_input)
    maxpool_out = MaxPooling1D(30)(conv1d_out)
    char = Flatten()(maxpool_out)

    output = concatenate([words, char])
    output = Dense(512, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(4096, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(len(vec2tag), activation="softmax")(output)

    model = Model(inputs=[words_input, char_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="nadam",
                  metrics=["accuracy"])

    model.summary()
    model.summary()
    if file is not None:
        model.load_weights(file)
    return model


def create_lstm2(word_len, char_len, vec2tag, file=None):
    words_input = Input(shape=(31,), dtype='int32', name='words_input')
    words_embendding = Embedding(word_len, 128)(words_input)
    lstm_out1 = Bidirectional(LSTM(32))(words_embendding)
    words = Dense(64)(lstm_out1)

    char_input = Input(shape=(40, char_len,), name='char_input')
    lstm_out2 = Bidirectional(LSTM(32))(char_input)
    char = Dense(64)(lstm_out2)

    output = concatenate([words, char])
    output = Dense(512, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(4096, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(len(vec2tag), activation="softmax")(output)

    model = Model(inputs=[words_input, char_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="nadam",
                  metrics=["accuracy"])

    model.summary()
    model.summary()
    if file is not None:
        model.load_weights(file)
    return model


def prepare_dict_simple(url):
    query = {"q": "*:*", "rows": 15843, "fl": "u, b"}
    response = requests.get(url, params=query)
    word_array = json.loads(response.text)["response"]["docs"]

    dict_ = dict()
    list_ = []
    for word_object in word_array:
        if "b" in word_object.keys() and not word_object["u"][0][0] == "+":
            for b in word_object["b"]:
                list_.append(b)
                dict_[len(list_)] = word_object["u"][0]

    return dict_, list_


def predict(word, lut, words):
    min = 1000
    pred = []
    for i in range(len(words)):
        _word = words[i]
        di = Levenshtein.distance(word, _word)
        if di < min:
            pred = [i]
            min = di
        if di == min:
            pred.append(i)
    max = len(pred)
    index = int(np.random.uniform(0, max))
    r = pred[index]
    return lut[r]


def tag_text_combined(text, model_conv, model_lstm, wdict, wlist, enc_data):
    word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag = enc_data
    p_text = purify_sentence(text)
    e_words = []
    e_chars = []
    for i in range(len(p_text)):
        sentence = p_text[max(0, i - 10):min(i + 10, len(p_text) - 1)]
        word_x, char_x, _, _ = \
            encode_sentence(word2vec, char2vec, char_len, tag2vec, tag_len, sentence, "",
                            min(i - max(0, i - 10), len(sentence) - 1))
        e_words.append(word_x)
        e_chars.append(char_x)
    pred = model_conv.predict([np.array(e_words), np.array(e_chars)])
    pred = pred.argmax(axis=-1)
    pred1 = model_lstm.predict([np.array(e_words), np.array(e_chars)])
    pred1 = pred1.argmax(axis=-1)
    o = []
    for i in range(len(p_text)):
        if pred[i] == pred1[i]:
            o.append((p_text[i], vec2tag[pred[i]]))
        else:
            o.append((p_text[i], predict(p_text[i], wdict, wlist)))
    return o


def tag_word_combined(text, model_conv, model_lstm, wdict, wlist, enc_data):
    word2vec, word_len, char2vec, char_len, tag2vec, tag_len, vec2tag = enc_data
    p_text = purify_sentence(text)
    e_words = []
    e_chars = []
    for i in range(len(p_text)):
        sentence = p_text[max(0, i - 10):min(i + 10, len(p_text) - 1)]
        word_x, char_x, _, _ = \
            encode_sentence(word2vec, char2vec, char_len, tag2vec, tag_len, sentence, "",
                            min(i - max(0, i - 10), len(sentence) - 1))
        e_words.append(word_x)
        e_chars.append(char_x)
    pred = model_conv.predict([np.array(e_words), np.array(e_chars)])
    pred = pred.argmax(axis=-1)
    pred1 = model_lstm.predict([np.array(e_words), np.array(e_chars)])
    pred1 = pred1.argmax(axis=-1)
    o = []
    for i in range(len(p_text)):
        if pred[i] == pred1[i]:
            chosen = (p_text[i], vec2tag[pred[i]])
        else:
            chosen = (p_text[i], predict(p_text[i], wdict, wlist))
        indices = [i for i, x in enumerate(wlist) if x == p_text[i]]
        words = [wdict[wlist[i]] for i in indices]
        if len(words) > 0:
            if chosen[1] in words:
                o.append(chosen)
            else:
                o.append(words[0])
        else:
            o.append(chosen)
    return o


def prepare_flair_csv(url, exclude_chance=0.0):
    excluded = []
    file = open("exluded.txt", encoding="utf-8", mode="r")
    for w in file:
        excluded.append(w.strip("\n"))
    file.close()

    query = {"q": "*:*", "rows": 15843}
    response = requests.get(url, params=query)
    word_array = json.loads(response.text)["response"]["docs"]

    example_array = []
    test_array = []
    ignored = excluded
    for word_object in word_array:
        if "example" in word_object.keys() and not word_object["u"][0][0] == "+":
            if "q" in word_object.keys():
                for q in word_object["q"]:
                    if np.random.uniform() < exclude_chance:
                        ignored.append(q)
            for example in word_object["example"]:
                sentence = []
                b = True
                temp = example.split("_")
                if len(temp) > 3:
                    continue
                elif len(temp) == 3 and ' ' in temp[1]:
                    continue
                elif len(temp) < 3:
                    continue
                sent_array = example.split(" ")[:-2]
                for word in sent_array:
                    stripped = re.sub(r"[!;?,.1234567890()_:/{}…†@+„”\"<>’*]", "", word)
                    if len(stripped) > 0:
                        if "_" in word and b:
                            stripped = "_" + stripped
                            b = False
                        sentence.append(stripped)

                for i in range(len(sentence)):
                    if "_" == sentence[i][0]:
                        sentence[i] = sentence[i][1:]
                        if not sentence[i] in ignored:
                            example_array.append((" ".join(sentence), word_object["u"][0], i))
                        else:
                            test_array.append((" ".join(sentence), word_object["u"][0], i))
                        break

    example_array = np.random.permutation(example_array)
    print(len(test_array))
    print(len(example_array))

    file2 = open("../Data/flair/train.csv", encoding="utf-8", mode="w")
    for s, t, i in example_array[0:300000]:
        file2.write(s + "\t" + t + "\t" + str(i) + "\n")
    file2.close()
    file3 = open("../Data/flair/dev.csv", encoding="utf-8", mode="w")
    for s, t, i in example_array[300000:360000]:
        file3.write(s + "\t" + t + "\t" + str(i) + "\n")
    file3.close()
    file4 = open("../Data/flair/test.csv", encoding="utf-8", mode="w")
    for s, t, i in test_array:
        file4.write(s + "\t" + t + "\t" + str(i) + "\n")
    file4.close()

