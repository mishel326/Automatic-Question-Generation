# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:59:40 2020

@author: -
"""

import json
import re
import operator
#import numpy as np
from functools import reduce
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27')

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"cannot", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub("[^a-z0-9 $%-]+", '', text.replace('.',' ').replace(',',' ').replace(':',' ').replace('-',' - ').replace('$',' $ ').replace('%',' % '))
    return text

def read_and_clean(paragraphs,question,file_name):
    with open(file_name) as file:
        data = json.loads(file.read())
        for i in data:
            temp_data = i['paragraphs']
            for j in temp_data:
                paragraphs.append('<BOS> '+clean_text(j['context'])+' <EOS>')
                list_of_questions = []
                tdata = j['qas']
                for k in tdata:
                    list_of_questions.append('<BOS> '+clean_text(k['question'])+' <EOS>')
                question.append(list_of_questions)
                
def func(paragrphs,questions):
    new_paragrphs = []
    new_questions = []
    for i in range(len(paragrphs)):
        for j in range(len(questions[i])):
            new_paragrphs.append(paragrphs[i])
    new_questions = reduce(operator.concat, questions)
    return new_paragrphs,new_questions
      
#train_paragraphs = []
#train_questions = []
#test_paragraphs = []
#test_questions = []
dev_paragraphs = []
dev_questions = []
#read_and_clean(train_paragraphs, train_questions, desktop_path + "\train.json")
#read_and_clean(test_paragraphs, test_questions, desktop_path + "\test.json")
read_and_clean(dev_paragraphs, dev_questions, "dataset\dev.json")

dev_paragraphs,dev_questions = func(dev_paragraphs,dev_questions)

dictionry = {"pad":0,"<BOS>":1,"<EOS>":2}  
 
def build_vocab(paragraphs):
    for paragraph in paragraphs:
        if paragraph not in dictionry:
            dictionry[paragraph] = len(dictionry)
    
def build_vocab_list(list_words):
    for paragraph in list_words:
        build_vocab(paragraph.split())
            
build_vocab_list(dev_paragraphs)
build_vocab_list(dev_questions)

def sentence_to_index(sentences):
    for sentence in range(len(sentences)):
        sentences[sentence] = dictionry[sentences[sentence]]
    return sentences

def list_to_index(_list):
    for i in range(len(_list)):
        _list[i] = sentence_to_index(_list[i].split())

def padding(_list):
    pad_size = reduce(lambda x,y:x if x > len(y) else len(y),_list,0)
    for i in range(len(_list)):
        _list[i] = _list[i]+[0]*(pad_size - len(_list[i]))


list_to_index(dev_paragraphs)
list_to_index(dev_questions)

padding(dev_paragraphs)
padding(dev_questions)


nlp.close()