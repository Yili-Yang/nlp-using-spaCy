
# coding: utf-8

# In[2]:

import gzip
import json
from spacy.matcher import Matcher
import spacy
from spacy.attrs import *
import random
from spacy.gold import GoldParse
from spacy.language import EntityRecognizer
import pandas as pd
from operator import itemgetter
#!pip install fuzzywuzzy
from fuzzywuzzy import fuzz
import pickle
from pathlib import Path


# In[40]:

class Train_spacy():

    def __init__(self, path_of_obj, custom_exact_dict,custom_fuzz_dict, language='en'):
        '''init with:
        the path of obj which contains the truth
        a dictionary containing the relationship between the key of obj and the actural customized label that you want to do exact match
        a dictionary containing the relationship between the key of obj and the actural customized label that you want to fuzz match'''
        with open(path_of_obj, 'r') as file:
            self.obj = json.load(file)
        self.nlp = spacy.load(language)
        self.custom_fuzz_dict = custom_fuzz_dict
        self.custom_exact_dict = custom_exact_dict
    def find_sub_list(self, sl, l):
        '''find exact match'''
        results = []
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind:ind + sll] == sl:
                results.append((ind, ind + sll - 1))

        return results

    def find_company_name(self, doc, company_name, nlp):
        '''take the doc and the job_tiltes in excel and return a list of tuple like
        [(start,end),...]'''
        result_list = list()
        doc_list = list()
        for toke in doc:
            doc_list.append(toke.text)
        re_ = list()
        doc1 = nlp.make_doc(company_name)
        list_name = list()
        for token1 in doc1:
            list_name.append(token1.text)
        re_ = find_sub_list(list_name, doc_list)
        if re_:
            result_list.extend(re_)
            re_ = list()
        return result_list

    def fuzzmatch(self, str1, str2, n):
        ratio = fuzz.ratio(str1, str2)
        if ratio >= n:
            return 1
        else:
            return 0

    def find_sub_list_fuzz(self, sl, l, first_accuracy, total_accuracy):
        # find fuzz match with 3 incides
        results = []
        sll = len(sl)
        lenl = len(l)
        for ind in (i for i, e in enumerate(l) if self.fuzzmatch(e, sl[0], first_accuracy) == 1):
            if self.fuzzmatch(l[ind:ind + sll], sl, total_accuracy) == 1:
                results.append((ind, ind + sll - 1))
            elif ind + sll - 1 >= ind and self.fuzzmatch(l[ind:ind + sll - 1], sl, total_accuracy) == 1:
                results.append((ind, ind + sll - 2))
            elif ind + sll - 2 >= ind and self.fuzzmatch(l[ind:ind + sll - 2], sl, total_accuracy) == 1:
                results.append((ind, ind + sll - 3))
            elif ind + sll - 3 >= ind and self.fuzzmatch(l[ind:ind + sll - 3], sl, total_accuracy) == 1:
                results.append((ind, ind + sll - 4))
            elif ind + sll + 1 <= lenl - 1 and self.fuzzmatch(l[ind:ind + sll + 1], sl, total_accuracy) == 1:
                results.append((ind, ind + sll))
            elif ind + sll + 2 <= lenl - 1 and self.fuzzmatch(l[ind:ind + sll + 2], sl, total_accuracy) == 1:
                results.append((ind, ind + sll + 1))
            elif ind + sll + 3 <= lenl - 1 and self.fuzzmatch(l[ind:ind + sll + 3], sl, total_accuracy) == 1:
                results.append((ind, ind + sll + 2))

        return results

    def find_company_name_fuzz(self, doc, company_name, first_accuracy, total_accuracy, nlp):
        # take the doc and the entity str and return a list of tuple like
        # [(start,end),...]
        result_list = list()
        doc_list = list()
        for toke in doc:
            doc_list.append(toke.text)
        re_ = list()
        doc1 = nlp.make_doc(company_name)
        list_name = list()
        for token1 in doc1:
            list_name.append(token1.text)
        re_ = self.find_sub_list_fuzz(
            list_name, doc_list, first_accuracy, total_accuracy)
        if re_:
            result_list.extend(re_)
            re_ = list()
        for i in range(len(result_list) - 1, -1, -1):
            ele_ = result_list[i]
            if ele_[1] < len(doc):
                if doc[ele_[1]].is_punct:
                    result_list[i] = (ele_[0], ele_[1] - 1)
            else:
                if doc[-1].is_punct:
                    result_list[i] = (ele_[0], len(doc) - 2)
        return result_list

    def pos_tuple_to_pos_list(self, start, end):
        # change tuple to postion list i.e change (1,4) to [1,2,3,4]
        result = list()
        for i in range(end - start + 1):
            result.append(i + start)
        return result

    def custom_label_priority(self, item, custom_label_list, check_pos_tuple_list):
        # check if the custom label has some overlap of the ori labels
        if item in custom_label_list:
            return False
        else:
            start = item[0]
            end = item[1]
            range_list = self.pos_tuple_to_pos_list(start, end)
            for posTuple in check_pos_tuple_list:
                compare_list = self.pos_tuple_to_pos_list(
                    posTuple[0], posTuple[1])
                test_set = set(range_list) & set(compare_list)
                if test_set:
                    return True
            return False
    
    def get_training_data_in_article_with_default_label(self):
        '''get training data with original entities and using article to train.'''
        custom_label_list = list()
        for key in self.custom_fuzz_dict:
            custom_label_list.append(key)
        check_pos_tuple_list = list()
        train_data = list()
        for record in self.obj:
            sentence = self.nlp(record['raw_article_text'])
            list_ = list()
            cus_list_ = list()
            # re-index the sentence token
            for token in sentence:
                if token.ent_type:
                    list_.append(
                        (token.idx, token.idx + len(token), token.ent_type_))
            # label the custom entity using fuzzmatch
            for key, value in self.custom_fuzz_dict.items():
                for buyer in record[key]:
                    list_fuzz_buyers = self.find_company_name_fuzz(
                        sentence, buyer, 90, 90, self.nlp)
                    for pos_tuple in list_fuzz_buyers:
                        if pos_tuple[1] >= len(sentence):
                            train_tuple = (sentence[pos_tuple[0]].idx, len(
                                sentence.text), value)
                            cus_list_.append(train_tuple)
                            check_pos_tuple_list.append(
                                (train_tuple[0], train_tuple[1]))
                        else:
                            train_tuple = (sentence[pos_tuple[0]].idx, sentence[pos_tuple[1]
                                                                                ].idx + len(sentence[pos_tuple[1]]), value)
                            cus_list_.append(train_tuple)
                            check_pos_tuple_list.append(
                                (train_tuple[0], train_tuple[1]))
            for key,value in self.custom_exact_match.items():
                for job in record[key]:
                    list_fuzz_buyers = self.find_company_name_fuzz(
                        sentence, buyer, 100, 100, self.nlp)
                    for pos_tuple in list_exact_jobs:
                        if pos_tuple[1] >= len(sentence):
                            train_tuple = (sentence[pos_tuple[0]].idx, len(
                                sentence.text), value)
                            cus_list_.append(train_tuple)
                            check_pos_tuple_list.append(
                                (train_tuple[0], train_tuple[1]))
                        else:
                            train_tuple = (sentence[pos_tuple[0]].idx, sentence[pos_tuple[1]
                                                                                ].idx + len(sentence[pos_tuple[1]]), value)
                            cus_list_.append(train_tuple)
                            check_pos_tuple_list.append(
                                (train_tuple[0], train_tuple[1])) 
            # make sure that custom label is over-riding the original ones.
            list_ = [x for x in list_ if not self.custom_label_priority(
                x, custom_label_list, check_pos_tuple_list)]
            list_.extend(cus_list_)
            # sort the list by starting position
            list_.sort(key=lambda tup: tup[0])
            if list_:
                train_data.append((sentence.text, list_))
        return train_data
    
    
    def get_training_data_in_sentence_with_default_label(self):
        '''get training data with original entities and using sentence to train.'''
        custom_label_list = list()
        for key in self.custom_fuzz_dict:
            custom_label_list.append(key)
        check_pos_tuple_list = list()
        train_data = list()
        for record in self.obj:
            doc = self.nlp(record['raw_article_text'])
            for sentence in doc.sents:  # loop through all the sentence in doc
                list_ = list()
                cus_list_ = list()
                # re-index the sentence token
                sentence = self.nlp(sentence.text)
                for token in sentence:
                    if token.ent_type:
                        list_.append(
                            (token.idx, token.idx + len(token), token.ent_type_))
                # label the custom entity using fuzzmatch
                for key, value in self.custom_fuzz_dict.items():
                    for buyer in record[key]:
                        list_fuzz_buyers = self.find_company_name_fuzz(
                            sentence, buyer, 90, 90, self.nlp)
                        for pos_tuple in list_fuzz_buyers:
                            if pos_tuple[1] >= len(sentence):
                                train_tuple = (sentence[pos_tuple[0]].idx, len(
                                    sentence.text), value)
                                cus_list_.append(train_tuple)
                                check_pos_tuple_list.append(
                                    (train_tuple[0], train_tuple[1]))
                            else:
                                train_tuple = (sentence[pos_tuple[0]].idx, sentence[pos_tuple[1]
                                                                                    ].idx + len(sentence[pos_tuple[1]]), value)
                                cus_list_.append(train_tuple)
                                check_pos_tuple_list.append(
                                    (train_tuple[0], train_tuple[1]))
                for key,value in self.custom_exact_match.items():
                    for job in record[key]:
                        list_fuzz_buyers = self.find_company_name_fuzz(
                            sentence, buyer, 100, 100, self.nlp)
                        for pos_tuple in list_exact_jobs:
                            if pos_tuple[1] >= len(sentence):
                                train_tuple = (sentence[pos_tuple[0]].idx, len(
                                    sentence.text), value)
                                cus_list_.append(train_tuple)
                                check_pos_tuple_list.append(
                                    (train_tuple[0], train_tuple[1]))
                            else:
                                train_tuple = (sentence[pos_tuple[0]].idx, sentence[pos_tuple[1]
                                                                                    ].idx + len(sentence[pos_tuple[1]]), value)
                                cus_list_.append(train_tuple)
                                check_pos_tuple_list.append(
                                    (train_tuple[0], train_tuple[1])) 
                # make sure that custom label is over-riding the original ones.
                list_ = [x for x in list_ if not self.custom_label_priority(
                    x, custom_label_list, check_pos_tuple_list)]
                list_.extend(cus_list_)
                # sort the list by starting position
                list_.sort(key=lambda tup: tup[0])
                if list_:
                    train_data.append((sentence.text, list_))
        return train_data

    @staticmethod
    def save_to_directory(train_data, path):
        '''save training data to '''
        output = open(path + '.pkl', 'wb')
        pickle.dump(train_data, output)
        output.close()

    @staticmethod
    def load_from_directory(path):
        pkl_file = open(path + '.pkl', 'rb')
        train_data = pickle.load(pkl_file)
        pkl_file.close()
        return train_data

    def train_ner(self, nlp, train_data, output_dir, training_rate, recurrent_time):
        # Add new words to vocab
        for raw_text, _ in train_data:
            doc = nlp.make_doc(raw_text)
            for word in doc:
                _ = nlp.vocab[word.orth]
        random.seed(0)
        # You may need to change the learning rate. It's generally difficult to
        # guess what rate you should set, especially when you have limited
        # data.
        nlp.entity.model.learn_rate = training_rate
        for itn in range(recurrent_time):
            random.shuffle(train_data)
            loss = 0.
            for raw_text, entity_offsets in train_data:
                gold = GoldParse(doc, entities=entity_offsets)
                # By default, the GoldParse class assumes that the entities
                # described by offset are complete, and all other words should
                # have the tag 'O'. You can tell it to make no assumptions
                # about the tag of a word by giving it the tag '-'.
                # However, this allows a trivial solution to the current
                # learning problem: if words are either 'any tag' or 'ANIMAL',
                # the model can learn that all words can be tagged 'ANIMAL'.
                # for i in range(len(gold.ner)):
                # if not gold.ner[i].endswith('ANIMAL'):
                #    gold.ner[i] = '-'
                doc = nlp.make_doc(raw_text)
                nlp.tagger(doc)
                # As of 1.9, spaCy's parser now lets you supply a dropout probability
                # This might help the model generalize better from only a few
                # examples.
                loss += nlp.entity.update(doc, gold)
            if loss == 0:
                break
        # This step averages the model's weights. This may or may not be good for
        # your situation --- it's empirical.
        nlp.end_training()
        if output_dir:
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.save_to_directory(output_dir)

    def train_based_on_default(self, train_data, model_name, start,end, training_rate, recurrent_time, output_directory=None):
        '''train a new model from the basic default model.
        input:
        training data that should be used to train the model
        languge that are being used
        start and the end of the indice training sample that is actually used
        training rate,recurrent time of the sample
        the path where the customized model is to be stored
        '''
        print("Loading initial model", model_name)
        nlp = spacy.load(model_name)
        if output_directory is not None:
            output_directory = Path(output_directory)
        else:
            print('model is not saved, please specifiy a path to store the model')
        # add labels
        for value in self.custom_fuzz_dict.values():
            nlp.entity.add_label(value)
        for value in self.custom_exact_dict.values():
            nlp.entity.add_label(value)
        self.train_ner(
            nlp, train_data[start:end], output_directory, training_rate, recurrent_time)
        print('Training end, model saved to %s' % (output_directory))

    def update(self, train_data, start, end, training_rate, recurrent_time, input_directory,output_directory = None,model_name='en'):
        '''update the model use train_data and save to a path(can be same or not)
        input:
        training data that should be used to update the model
        start and the end of the indice training sample that is actually used
        training rate,recurrent time of the sample
        the path of the model that is to be updated
        the path where the updated model is to be stored, default to the input_directory
        default language is english
        '''
        print("Loading initial model", model_name)
        print('path:', input_directory)
        nlp = spacy.load(model_name, path=Path(input_directory))
        if output_directory == None:
            output = input_directory
        else:
            output = output_directory
        output_directory = Path(output)
        for value in self.custom_fuzz_dict.values():
            nlp.entity.add_label(value)
        for value in self.custom_exact_dict.values():
            nlp.entity.add_label(value)
        self.train_ner(
            nlp, train_data[start:end], output_directory, training_rate, recurrent_time)
        print('Updating end, model saved to %s' % (output_directory))

    def count_train_matches(self, train_data, path_test, language='en'):
        '''compare the performance of training between the truth and the trained result
        input: 
        training data that is the truth, 
        the path where the trained model is stored, 
        default language is english'''
        nlp_test = spacy.load(language, path=path_test)
        right_match = 0
        wrong_label = 0
        wrong_text_label = 0
        total_test_ents = 0
        total_ori_ents = 0
        obvious_error = 0
        custom_label_list = list()
        custom_label_count = 0
        ori_label_count = 0
        truth_custom_label_count = 0
        truth_ori_label_count = 0
        for value in self.custom_fuzz_dict.values():
            custom_label_list.append(value)
        for value in self.custom_exact_dict.values():
            custom_label_list.append(value)
            #print(custom_label_list)
        for text, train_list in train_data:
            total_ori_ents += len(train_list)
            doc_test = nlp_test(text)
            test_label_list = list()
            for token in doc_test:
                if token.ent_type_:
                    # change position labels to position format
                    test_label_list.append(
                        (token.idx, token.idx + len(token), token.ent_type_))
                    if token.ent_type_ in custom_label_list:
                        custom_label_count += 1
                    else:
                        ori_label_count += 1
                    if token.is_punct:
                        obvious_error += 1
            total_test_ents += len(test_label_list)
            for item in train_list:
                if item[2] in custom_label_list:
                    truth_custom_label_count += 1
                else:
                    truth_ori_label_count += 1
            for i in range(len(train_list) - 1, -1, -1):
                ele_ = train_list[i]
                if ele_ in test_label_list:
                    right_match += 1
                    del train_list[i]
            for i in train_list:
                for j in test_label_list:
                    if i[0] == j[0] and i[1] == j[1]:
                        train_list.remove(i)
                        test_label_list.remove(j)
                        wrong_label += 1
            wrong_text_label += len(test_label_list)

        count_dict = {'right_match': right_match, 'wrong_label': wrong_label, 'wrong_text_label': wrong_text_label,
                      'total_test_ents': total_test_ents, 'total_ori_ents': total_ori_ents, 'obvious_error': obvious_error,
                      'custom_label_count': custom_label_count, 'ori_label_count': ori_label_count,
                      'truth_custom_label_count': truth_custom_label_count, 'truth_ori_label_count': truth_ori_label_count}
        return count_dict

    def count_difference_between_two_models(self, path_test, path_ori=None, obj =None,language='en'):
        '''compare the performance of training between two trained models
        input: 
        the path where the trained model are stored,
        test obj which contain the 'raw_article_text',default is the obj that init the class
        default language is english'''
        if obj == None:
            obj =self.obj
        right_match = 0
        wrong_label = 0
        wrong_text_label = 0
        total_test_ents = 0
        total_ori_ents = 0
        test_obvious_error = 0
        compare_obvious_error = 0
        custom_label_list = list()
        test_custom_label_count = 0
        test_ori_label_count = 0
        compare_custom_label_count = 0
        compare_ori_label_count = 0
        nlp_test = spacy.load(language, path=path_test)
        nlp_ori = spacy.load(language, path=path_ori)
        for value in self.custom_fuzz_dict.values():
            custom_label_list.append(value)
        for value in self.custom_exact_dict.values():
            custom_label_list.append(value)
        for i in obj:
            text = i['raw_article_text']
            doc_test = nlp_test(text)
            test_label_list = list()
            for token in doc_test:
                if token.ent_type_:
                    # change position labels to position format
                    test_label_list.append(
                        (token.idx, token.idx + len(token), token.ent_type_))
                    if token.ent_type_ in custom_label_list:
                        test_custom_label_count += 1
                    else:
                        test_ori_label_count += 1
                    if token.is_punct:
                        test_obvious_error += 1
            total_test_ents += len(test_label_list)
            # print(len(test_label_list))
            doc_ori = nlp_ori(text)
            ori_label_list = list()
            for token in doc_ori:
                if token.ent_type_:
                    # change position labels to position format
                    ori_label_list.append(
                        (token.idx, token.idx + len(token), token.ent_type_))
                    if token.ent_type_ in custom_label_list:
                        compare_custom_label_count += 1
                    else:
                        compare_ori_label_count += 1
                    if token.is_punct:
                        compare_obvious_error += 1
            total_ori_ents += len(ori_label_list)
            # print(len(ori_label_list))
            for i in range(len(test_label_list) - 1, -1, -1):
                ele_ = test_label_list[i]
                if ele_ in ori_label_list:
                    right_match += 1
                    del test_label_list[i]
                    
                   
            for i in test_label_list:
                for j in ori_label_list:
                    if i[0] == j[0] and i[1] == j[1]:
                        #print(i,j)
                        test_label_list.remove(i)
                        wrong_label += 1
            wrong_text_label += len(test_label_list)
        count_dict = {'same_match': right_match, 'different_label': wrong_label, 'different_text_label': wrong_text_label,
                      'test_model_ents': total_test_ents, 'compare_model_ents': total_ori_ents,
                      'obvious_error_of_test_model': test_obvious_error,
                      'obvious_error_of_compare_model': compare_obvious_error,
                      'custom_label_count_of_test_model': test_custom_label_count, 'ori_label_count_of_test_model': test_ori_label_count,
                      'custom_label_count_of_compare_model': compare_custom_label_count,
                      'ori_label_count_of_compare_model': compare_ori_label_count}
        return count_dict


# In[41]:
if __name__ = 'main':
	fuzz_dict = {'buyers': 'BUYERS', 'sellers': 'SELLERS'}


	# In[42]:

	# init with the manully corrected data and language
	model = Train_spacy('DataWithSource.json', {},fuzz_dict, 'en')


	# In[ ]:

	# get training data in the format of (start,end,label)
	train_data = model.get_training_data_in_sentence_with_default_label()
	#train_data = model.get_training_data_in_article_with_default_label()

	# In[ ]:

	model.save_to_directory(train_data, 'test')  # save training data to a path


	# In[45]:

	train_data = model.load_from_directory('test')  # load training data from a path


	# In[ ]:

	help(model.train_based_on_default)


	# In[ ]:

	model.train_based_on_default(
	    train_data, 'en', 1,100, 0.0001, 1, 'new_training_100_0001_1')  # train a new model


	# In[ ]:

	model.update(train_data, 100, 200, 0.0001, 1, 'new_training_100_0001_1')  # update a existed model
	#can take a output path if you want to store the updated model else where.

	#compare between training data and the testing data, using training as benchmark:
	#	count_train_matches()

	#compare between two models, using first model as benchmark:
	#count_difference_between_two_models


# In[ ]:



