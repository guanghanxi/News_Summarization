import os
import numpy as np
import pandas as pd
import nltk.tokenize
import re
import random
from nltk.util import ngrams
import tqdm
from nltk.tokenize import RegexpTokenizer
from bert_score import score
import torch


tokenizer = RegexpTokenizer(r'\w+')

# Load all text in the directory path

def read_text(path):
    files= os.listdir(path) 
    results = {'text':[], 'highlight_1':[], 'highlight_2':[], 'highlight_3':[], 'highlight_4':[]}
    for file in tqdm.tqdm(files):
        if not os.path.isdir(file):
            file_name = path + '/'+file
            with open(file_name, encoding="utf-8") as f:
                text = (f.read()).replace('\n', " ").replace("(CNN)", "").replace("--", "")
                if len(text)<1000:
                    continue
                text_highlights = text.split("@highlight")
                final_text = text_highlights[0]
                results['text'].append(nltk.tokenize.sent_tokenize(final_text.strip()))
                for i in range(1, 5):
                    key = 'highlight_'+str(i)
                    if i<len(text_highlights):
                        results[key].append(text_highlights[i])
                    else:
                        results[key].append("")
    return pd.DataFrame(results)


# lead_3 model, return all words in the first three sentence of the text

def lead_3(sent_list):   
    result = []
    result_sent = ""
    for i in range(3):
        result.append(tokenizer.tokenize(sent_list[i]))
        result_sent += sent_list[i]
    return result, result_sent

# random model

def random_baseline(sent_list):   
    samples = random.sample(sent_list,3)
    result = []
    for sent in samples:
        result.append(tokenizer.tokenize(sent))
    return result, " ".join(samples)

# return the reference token

def reference_token(row):
    result = []
    result_sent = ""
    for i in range(1,5):
        key = 'highlight_' + str(i)
        result.append(tokenizer.tokenize(row[key]))
        result_sent += row[key]
    return result, result_sent

def rouge_1(predict, refr):
    predict_list = [word for sent in predict for word in sent]
    refr_list = [word for sent in refr for word in sent]
    temp = 0
    for word in predict_list:
        if word in refr_list:
            temp += 1
    return 2*temp/(len(predict_list)+len(refr_list))

def rouge_n(predict, refr, n):
    n_gram_pred = []
    n_gram_ref = []
    for sent in predict:
        n_gram_pred = n_gram_pred + list(ngrams(sent,n))
    for sent in refr:
        n_gram_ref = n_gram_ref + list(ngrams(sent,n))
    temp = 0
    for pair in n_gram_pred:
        if pair in n_gram_ref:
            temp += 1
    return 2*temp/(len(n_gram_pred) + len(n_gram_ref))


if __name__ == '__main__':
    test_dir = 'test'
    s = read_text(test_dir)
    num = len(s)
    score_lead_1 = 0
    score_lead_2 = 0
    score_rand_1 = 0
    score_rand_2 = 0
    n = 2
    lead3_predict_list = []
    rand_predict_list = []
    ref_predict_list = []
    for index, row in tqdm.tqdm(s.iterrows()):
        lead_predict, lead_sent = lead_3(row['text'])
        rand_predict, rand_sent = random_baseline(row['text'])
        reference, ref_sent = reference_token(row)
        lead3_predict_list.append(lead_sent)
        rand_predict_list.append(rand_sent)
        ref_predict_list.append(ref_sent)
        score_lead_1 += rouge_1(lead_predict, reference)
        score_rand_1 += rouge_1(rand_predict, reference)
        score_lead_2 += rouge_n(lead_predict, reference, n)
        score_rand_2 += rouge_n(rand_predict, reference, n)
        
    lead3_P, lead3_R, lead3_F1  = score(lead3_predict_list, ref_predict_list, lang = "en", verbose = True)
    rand_P, rand_R, rand_F1 = score(rand_predict_list, ref_predict_list, lang = "en", verbose = True)
    print('Rouge-1 Score for Lead-3 Model: ', score_lead_1)
    print('Rouge-2 Score for Lead-3 Model: ', score_lead_2)
    print('Rouge-1 Score for Random Model: ', score_rand_1)
    print('Rouge-2 Score for Random Model: ', score_rand_2)
    print('BERT Precision for Lead-3 Model: ',  torch.sum(lead3_P))
    print('BERT Recall for Lead-3 Model: ',  torch.sum(lead3_R))
    print('BERT F-1 Score for Lead-3 Model: ',  torch.sum(lead3_F1))
    print('BERT Precision for Random Model: ',  torch.sum(rand_P))
    print('BERT Recall for Random Model: ',  torch.sum(rand_R))
    print('BERT F-1 Score for Random Model: ',  torch.sum(rand_F1))