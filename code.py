<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:59:40 2020

@author: -
"""

import json
import re
import operator
import numpy as np
import logging
from functools import reduce
from stanfordcorenlp import StanfordCoreNLP

#nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27')

print("pass")

paragraph_pad_size = [0]
question_pad_size = [0]

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    text = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'em", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"cannot", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    #text = re.sub(r'[^\w\s]','',text)
    #text = re.sub("[^a-z0-9 $%-]+", '', text.replace('.',' ').replace(',',' ').replace(':',' ').replace('-',' - ').replace('$',' $ ').replace('%',' % ').replace('/',' / '))
    return text
                

def read_and_clean(paragraphs,question,file_name):
    with open(file_name,encoding='utf-8') as file:
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
    for paragrph in range(len(paragrphs)):
        for question in range(len(questions[paragrph])):
            new_paragrphs.append(paragrphs[paragrph])
    new_questions = reduce(operator.concat, questions)
    return new_paragrphs,new_questions

def stanford(_list):
    with StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27', quiet=False, logging_level=logging.DEBUG) as snlp:
        new_list = []
        for sent in _list :
            sentence = []
            stanford_obj = snlp.annotate(sent, properties={'annotators': "ssplit",'outputFormat': 'json', 'timeout':'5000000'})
            stanford_obj = json.loads(stanford_obj)
            for x in stanford_obj['sentences']:
                for j in x['tokens']:
                    sentence.append( j['word'] ) 
            new_list.append(sentence)
    return new_list
                    
    
      
#train_paragraphs = []
#train_questions = []
#test_paragraphs = []
#test_questions = []
dev_paragraphs = []
dev_questions = []
#read_and_clean(train_paragraphs, train_questions,r"dataset\train.json")
#read_and_clean(test_paragraphs, test_questions, r"dataset\test.json")
read_and_clean(dev_paragraphs, dev_questions, "dataset\dev.json")

dev_paragraphs, dev_questions = func(dev_paragraphs,dev_questions)
dev_paragraphs, dev_questions = stanford(dev_paragraphs),stanford(dev_questions)
#dev_paragraphs, dev_questions = func(dev_paragraphs,dev_questions)
#train_paragraphs, train_questions = func(train_paragraphs, train_questions)
#test_paragraphs, test_questions = func(test_paragraphs, test_questions)

dictionry = {"pad":0,"<BOS>":1,"<EOS>":2}  

def build_vocab(paragraphs):
    for i in paragraphs:
        for j in i:
            if j not in dictionry:
                dictionry[j] = len(dictionry)
     
def build_vocab_list(list_words):
    for paragraph in list_words:
        build_vocab(paragraph)
             
build_vocab(dev_paragraphs)
build_vocab(dev_questions)
#build_vocab(train_paragraphs)
#build_vocab(train_questions)
#build_vocab(test_paragraphs)
#build_vocab(test_questions)

def list_to_index(_list):
    for paragraph in _list:
        _list[_list.index(paragraph)] = sentence_to_index(paragraph)

def sentence_to_index(sentences):
    for sentence in range(len(sentences)):
        sentences[sentence] = dictionry[sentences[sentence]]
    return sentences 

def padding(_list,pad):
    pad[0] = reduce(lambda x,y:x if x > len(y) else len(y),_list,pad[0])
    for i in range(len(_list)):
        _list[i] = _list[i]+[0]*(pad[0] - len(_list[i]))

#list_to_index(train_paragraphs)
#list_to_index(train_questions) 
#list_to_index(test_paragraphs)
#list_to_index(test_questions)      
list_to_index(dev_paragraphs)
list_to_index(dev_questions)
 
#padding(train_paragraphs,paragraph_pad_size)
#padding(train_questions,question_pad_size)
#padding(test_paragraphs,paragraph_pad_size)
#padding(test_questions,question_pad_size)

padding(dev_paragraphs,paragraph_pad_size)
padding(dev_questions,question_pad_size)

dev_paragraphs = np.array(dev_paragraphs)
dev_questions = np.array(dev_questions)

#train_paragraphs = np.array(train_paragraphs)
#train_questions = np.array(train_questions)
#test_paragraphs = np.array(test_paragraphs)
#test_questions = np.array(test_questions)

=======
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:59:40 2020

@author: -
"""

import json
import re
import operator
import numpy as np
import logging
from functools import reduce
from stanfordcorenlp import StanfordCoreNLP

#nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27')

print("pass")

paragraph_pad_size = [0]
question_pad_size = [0]

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    text = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", text)
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
    #text = re.sub("[^a-z0-9 $%-]+", '', text.replace('.',' ').replace(',',' ').replace(':',' ').replace('-',' - ').replace('$',' $ ').replace('%',' % ').replace('/',' / '))
    return text

#def stanford(paragraph):
#    while True:
#        try:
#            para = nlp.word_tokenize(paragraph)
#            if len(para) > 0:
#                return para
#        except:
#            print("ERROR")
#            time.sleep(5)
            
        

def read_and_clean(paragraphs,question,file_name):
    with open(file_name,encoding='utf-8') as file:
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
    for paragrph in range(len(paragrphs)):
        for question in range(len(questions[paragrph])):
            new_paragrphs.append(paragrphs[paragrph])
    new_questions = reduce(operator.concat, questions)
    return new_paragrphs,new_questions
      
#train_paragraphs = []
#train_questions = []
#test_paragraphs = []
#test_questions = []
dev_paragraphs = []
dev_questions = []
#read_and_clean(train_paragraphs, train_questions,r"dataset\train.json")
#read_and_clean(test_paragraphs, test_questions, r"dataset\test.json")
read_and_clean(dev_paragraphs, dev_questions, "dataset\dev.json")

dev_paragraphs, dev_questions = func(dev_paragraphs,dev_questions)
#train_paragraphs, train_questions = func(train_paragraphs, train_questions)
#test_paragraphs, test_questions = func(test_paragraphs, test_questions)

dictionry = {"pad":0,"<BOS>":1,"<EOS>":2}  

def build_vocab(paragraphs):
    with StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27', quiet=False, logging_level=logging.DEBUG) as snlp:
        for sent in paragraphs : 
            stanford_obj = snlp.annotate(sent, properties={'annotators': "ssplit",'outputFormat': 'json', 'timeout':'5000000'})
            stanford_obj = json.loads(stanford_obj)
            for x in stanford_obj['sentences']:
                for j in x['tokens']:
                    if j['word'] not in dictionry:
                        dictionry[j['word']] = len(dictionry)
     
def build_vocab_list(list_words):
    for paragraph in list_words:
        build_vocab(paragraph.split())
             
build_vocab(dev_paragraphs)
build_vocab(dev_questions)
#build_vocab(train_paragraphs)
#build_vocab(train_questions)
#build_vocab(test_paragraphs)
#build_vocab(test_questions)
 
def sentence_to_index(sentences):
    for sentence in range(len(sentences)):
        sentences[sentence] = dictionry[sentences[sentence]]
    return sentences
 
def list_to_index(_list):
    for i in range(len(_list)):
        _list[i] = sentence_to_index(_list[i].split())

def padding(_list,pad):
    pad[0] = reduce(lambda x,y:x if x > len(y) else len(y),_list,pad[0])
    for i in range(len(_list)):
        _list[i] = _list[i]+[0]*(pad[0] - len(_list[i]))

#list_to_index(train_paragraphs)
#list_to_index(train_questions) 
#list_to_index(test_paragraphs)
#list_to_index(test_questions)      
list_to_index(dev_paragraphs)
list_to_index(dev_questions)
 
#padding(train_paragraphs,paragraph_pad_size)
#padding(train_questions,question_pad_size)
#padding(test_paragraphs,paragraph_pad_size)
#padding(test_questions,question_pad_size)
padding(dev_paragraphs,paragraph_pad_size)
padding(dev_questions,question_pad_size)

dev_paragraphs = np.array(dev_paragraphs)
dev_questions = np.array(dev_questions)
#train_paragraphs = np.array(train_paragraphs)
#train_questions = np.array(train_questions)
#test_paragraphs = np.array(test_paragraphs)
#test_questions = np.array(test_questions)

>>>>>>> 9c464f9ea52905c7d78d07c78d615e4f7b207dd9
