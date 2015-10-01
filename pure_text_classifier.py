#!/usr/bin/python

import math
import json
import random
import numpy as np
from numpy import zeros
from gensim import models
from gensim import matutils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from time import time

label_file_path = "Hygiene/hygiene.dat.labels"
text_file_path = "Hygiene/hygiene.dat"
additional_file_path = "Hygiene/hygiene.dat.additional"
testing_label_path = "data/testing.labels"

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

    #parse label
    hy_labels = getLabel(label_file_path);
    print(str(len( hy_labels)) + " labels")

    #parse text
    hy_text = getText(text_file_path);
    print(str(len( hy_text)) + " texts")

    #tfidf vectorize text
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=30000, min_df=2, stop_words='english', use_idf=True)
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

    text_topics_list = []
    with open( 'topic_distribution.log', 'w') as f:
        for k, top_dis in enumerate(text_topics):
            f.write(str(top_dis))
            f.write("\n")

            top_dis_array = zeros(100)
            for (index, value) in top_dis:
                top_dis_array[index] = value

            text_topics_list.append(top_dis_array)

    # the first N topic prob. distributions are traing_data
    training_data = np.array( text_topics_list[:len(hy_labels)] )

    #train classifier, support vector machine or KNN (with topic probability)
    clf = KNeighborsClassifier( n_neighbors=5, weights='distance' )
    #clf = SVC(probability=True)
    t0 = time()
    clf.fit( training_data, np.array(hy_labels) )
    print("classifier training done in %fs" % (time() - t0))

    #predict label of the testing_text
    testing_label = clf.predict( np.array(text_topics_list) )
    testing_proba = clf.predict_proba( np.array(text_topics_list) )

    #log file for debug
    with open("testing_proba.log","w") as log_file:
        for class_proba in testing_proba:
            log_file.write( str(class_proba) + "\n" )

    #classifier result
    with open(testing_label_path,"w") as testing_label_file:
        for label in testing_label:
            testing_label_file.write( str(label) + "\n" )

if __name__=="__main__":
    main()