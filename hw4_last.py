import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv
from collections import Counter
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from math import nan
UNK = "<UNK>"
PAD = "<PAD>"


train_df = pd.read_csv("data/train", sep = ' ', header=None, names = ["ind", "word", "ner"], quoting=3)
val_df = pd.read_csv("data/dev", sep = ' ', header=None, names = ["ind","word", "ner"], quoting=3)
test_df = pd.read_csv("data/test", sep = ' ', header=None, names = ["ind","word"], quoting=3)


v = Counter(train_df.word)
ner = Counter(train_df.ner)
vocab = {}

for i,w in enumerate(v):
    vocab[str(w)] = i
    
vocab[UNK] = len(vocab)
vocab[PAD] = len(vocab)
ner_ids = {}
ids_ner = []

for i,n in enumerate(ner):
    ner_ids[n] = i
    ids_ner.append(n)
    
ner_ids[PAD] = len(ner_ids)
ids_ner.append(PAD)
train_df.ner = train_df.ner.apply(lambda s: ner_ids[s])
train_df


j = -1
training_data = []
for i, row in train_df.iterrows():
    if row.ind == 1:
        training_data.append([[vocab[row.word]], [row.ner]])
        j += 1
    else:
        if isinstance(row.word, str):
            training_data[j][0].append(vocab[row.word])
            training_data[j][1].append(row.ner)


def prepare_dev_data(vocabulary, val_df):
    j = -1
    dev_data = []
    count_unk = 0
    for i, row in val_df.iterrows():
        if row.word in vocabulary:
            word_index = vocabulary[row.word]
        else:
            word_index = vocabulary[UNK]
            count_unk += 1
        if row.ind == 1:
            dev_data.append([[word_index], [row.ner]])
            j += 1
        else:
            if isinstance(row.word, str):
                dev_data[j][0].append(word_index)
                dev_data[j][1].append(row.ner)
            else:
                dev_data[j][0].append(vocabulary[UNK])
                dev_data[j][1].append(row.ner)
    
    
    return dev_data



training_data = sorted(training_data, key=lambda x: len(x[1]))



def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.tensor(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    
    return emb_layer


class bilstm_model(nn.Module):
    def __init__(self, input_dim, vocab_size, weights_matrix,output_dim=len(ner_ids), is_glove = False):
        super(bilstm_model, self).__init__()
        self.is_glove = is_glove
        
        self.dropout = nn.Dropout(0.33)
        if is_glove:
            self.word_vectors = create_emb_layer(weights_matrix, True)
        else:
            self.word_vectors = nn.Embedding(vocab_size+1, input_dim)
        self.lstm = nn.LSTM(input_dim,256,1, batch_first=True, bidirectional=True)
        self.lin_1 = nn.Linear(256*2, 128)
        self.elu = nn.ELU()
        self.lin_2 = nn.Linear(128, output_dim)
        
    def forward(self, sentences):
        vecs = self.word_vectors(sentences)
        vecs = self.dropout(vecs)
        lstm_out, _ = self.lstm(vecs) 
        lstm_out = self.dropout(lstm_out)
        linear_out = self.lin_1(lstm_out)
        elu_out = self.elu(linear_out)
        tag_scores = self.lin_2(elu_out)
        return tag_scores



def write_pred_file(df, model, file, dev_data):
    i=0
    model.eval()
    with open(file, 'w') as f:
        for sentence, tags in dev_data:
            s = torch.tensor(sentence)
            tag_scores = model(s.view(1, s.size(0))).view(s.size(0), 10)
            predicted = torch.max(tag_scores.data, 1)[1].cpu().numpy() 
        
            for p in predicted:
                f.write(f"{val_df.ind[i]} {val_df.word[i]} {val_df.ner[i]} {ids_ner[p]}\n")

                if p == "" or p == " " or val_df.ner[i] == "" or val_df.ner[i]== " " or ids_ner[p] == "" or ids_ner[p] == " ":
                    print("error bantu")
                i=i+1
            f.write("\n")

def prepare_test_data(vocabulary, test_df):
    j = -1
    test_data = []
    count_unk = 0
    for i, row in test_df.iterrows():
        if row.word in vocabulary:
            word_idx = vocabulary[row.word]
        else:
            word_idx = vocabulary[UNK]
            count_unk += 1
        if row.ind == 1:
            test_data.append([word_idx])
            j += 1
        else:
            if isinstance(row.word, str):
                test_data[j].append(word_idx)
            else:
                test_data[j].append(vocabulary[UNK])
    
    return test_data

def pred_test(df, model, file, test_data):
    i=0
    model.eval()
    with open(file, 'w') as f:
        for sentence in test_data:
            s = torch.tensor(sentence)
            tag_scores = model(s.view(1, s.size(0))).view(s.size(0), 10)
            predicted = torch.max(tag_scores.data, 1)[1].cpu().numpy() 
        
            for p in predicted:
                f.write(f"{test_df.ind[i]} {test_df.word[i]} {ids_ner[p]}\n")
                i=i+1
            f.write("\n")


def model_load(embedding_length, vocab_size, file,is_glove=False, weights_matrix=None):
        model = bilstm_model(is_glove=is_glove, input_dim=embedding_length,vocab_size=vocab_size, weights_matrix=weights_matrix)
        model.load_state_dict(torch.load(file))
        return model


model = model_load(embedding_length=100, vocab_size=len(vocab), file='blstm1.pt')


dev_data = prepare_dev_data(vocab, val_df)

write_pred_file(val_df, model, "data/dev1.out", dev_data)

test_data = prepare_test_data(vocab, test_df)

pred_test(test_df, model, "data/test1.out",test_data)



v = Counter(train_df.word)
v += Counter(val_df.word)
v += Counter(test_df.word)
vocab_size_glove = len(v)
embedding_length_glove = 101
vocab_glove = {}
for i,w in enumerate(v):
    vocab_glove[str(w)] = i
vocab_glove[UNK] = len(vocab_glove)
vocab_glove[PAD] = len(vocab_glove)



glove = {}
with open("glove.6B.100d.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        emb = line.rstrip().split()
        emb.append(0)
        glove[emb[0]] = list(map(float, emb[1:]))


weights_matrix = np.zeros((len(vocab_glove), 101))
words_found = 0
words_found_after_lower = 0
for key, value in vocab_glove.items():
    try:
        weights_matrix[value] = glove[key]
        words_found += 1
    except KeyError:
        try:
            upper_case_emb = glove[key.lower()]
            upper_case_emb[-1] = 1.0
            weights_matrix[value] = upper_case_emb
            words_found_after_lower += 1
        except KeyError:
            if key == PAD:
                weights_matrix[value] = np.zeros(101) 
            else:
                weights_matrix[value] = np.random.normal(scale=0.6, size=(101, ))



model_glove = model_load(embedding_length=101,
                   vocab_size=len(vocab_glove), is_glove=True, file='blstm2.pt',weights_matrix = weights_matrix)

dev_data = prepare_dev_data(vocab_glove, val_df)

write_pred_file(val_df, model_glove, "data/dev2.out",dev_data)

test_data = prepare_test_data(vocab_glove, test_df)

pred_test(test_df, model_glove, "data/test2.out",test_data)




    
    

    








