#!/usr/bin/python

import math
import json
import random
import logging
import numpy as np
from numpy import zeros
from gensim import models
from gensim import matutils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from time import time

label_file_path = "Hygiene/hygiene.dat.labels"
text_file_path = "Hygiene/hygiene.dat"
additional_file_path = "Hygiene/hygiene.dat.additional"
testing_label_path = "data/testing.additional.labels"

def getLabel(file_path):
    """
        parse label
        return list of labels, 1 means passed, 2 means not
    """
    return_labels = []
    with open(file_path , "r") as label_file:
        for line in label_file:
            try:
                if( int( line.strip() ) == 1):
                    return_labels.append(1)
                else:
                    return_labels.append(0)
            except:
                break
    return return_labels

def getText(file_path):
    """
        parse textual information (need some transform)
        return list of texts
    """
    return_texts = []
    with open(file_path , "r") as text_file:
        for line in text_file:
            return_texts.append( line.strip() )
    return return_texts

def getAdditional(file_path):
    """
        parse non-textual information (need some transform)
        return list of additional information dict
    """
    return_additional = []
    with open(file_path , "r") as additional_file:
        for line in additional_file:
            tmp = line.strip().split('"')
            cat_in_string = tmp[1]
            cat_in_list = cat_in_string[2:-2].split("', '")
            numbers = tmp[2].split(",")[1:]
            single = {
                "cat":cat_in_list,
                "zip":numbers[0],
                "rev_num":numbers[1],
                "avg_stars":numbers[2]
            }
            return_additional.append(single)

    return return_additional

def main():

    #parse additional
    hy_additional = getAdditional(additional_file_path)

    #make categories be vector
    cat_text = []
    for single in hy_additional:
        catgories = single['cat']
        cat_text.append( " ".join(catgories) )

    cat_vectorizer = CountVectorizer(binary=True)
    t0 = time()
    cat_vec = cat_vectorizer.fit_transform(cat_text)
    print("vectorize done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % cat_vec.shape)

    t0 = time()
    cat_corpus = matutils.Sparse2Corpus(cat_vec,  documents_columns=False)
    print("transform category corpus done in %fs" % (time() - t0))

    cat_lda = models.ldamodel.LdaModel(cat_corpus, num_topics=10)
    print("category lda done in %fs" % (time() - t0))
    cat_topics = cat_lda.get_document_topics(cat_corpus)

    #parse label
    hy_labels = getLabel(label_file_path)
    print(str(len( hy_labels)) + " labels")

    #parse text
    hy_text = getText(text_file_path)
    print(str(len( hy_text)) + " texts")

    #tfidf vectorize text
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)
    t0 = time()
    X = vectorizer.fit_transform(hy_text)
    print("vectorize done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    t0 = time()
    corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    print("transform corpus done in %fs" % (time() - t0))

    #lda topic modeling on text
    t0 = time()
    lda = models.ldamodel.LdaModel(corpus, num_topics=100)
    print("lda done in %fs" % (time() - t0))
    text_topics = lda.get_document_topics(corpus)

    #mixture of cat_topic_prob and text_topic_prob
    mix_topics_list = []
    for k, top_dis in enumerate(text_topics):
        top_dis_array = zeros(110)
        for (index, value) in top_dis:
            top_dis_array[index] = value*9/10

        mix_topics_list.append(top_dis_array)

    for k, cat_top_dis in enumerate(cat_topics):
        for (index, value) in cat_top_dis:
            mix_topics_list[k][100+index] = value/10
    
    with open( 'mix_topic_distribution.log', 'w') as f:
        for k, top_dis in enumerate(mix_topics_list):
            f.write(str(top_dis))
            f.write("\n==========\n")

    training_data = np.array( mix_topics_list[:len(hy_labels)] )

    #train classifier, support vector machine or KNN (with topic probability)
    clf = KNeighborsClassifier( n_neighbors=5, weights='distance' )
    #clf = SVC(probability=True)
    t0 = time()
    clf.fit( training_data, np.array(hy_labels) )
    print("classifier training done in %fs" % (time() - t0))

    #predict label of the testing_text and categories mixture
    testing_label = clf.predict( np.array(mix_topics_list) )
    testing_proba = clf.predict_proba( np.array(mix_topics_list) )

    #log file for debug
    with open("testing_mix_proba.log","w") as log_file:
        for class_proba in testing_proba:
            log_file.write( str(class_proba) + "\n" )

    #classifier result
    with open(testing_label_path,"w") as testing_label_file:
        for label in testing_label:
            testing_label_file.write( str(label) + "\n" )

if __name__=="__main__":
    main()