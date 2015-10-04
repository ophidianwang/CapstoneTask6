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
testing_label_path = "data/testing.mixture.labels"

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

    mixture_proba = []

    #parse label
    hy_labels = getLabel(label_file_path)
    print(str(len( hy_labels)) + " labels")

    #parse text
    hy_text = getText(text_file_path)
    print(str(len( hy_text)) + " texts")

    #parse additional
    hy_additional = getAdditional(additional_file_path)

    cat_text = []   #make categories be vector
    rating = [] #rating
    review_num = [] #number of reviews
    for single in hy_additional:
        catgories = single['cat']
        cat_text.append( " ".join(catgories) )
        rating.append( single['avg_stars'] )
        review_num.append( single['rev_num'] )

    #calculate proba from categories
    cat_vectorizer = CountVectorizer(binary=True)
    t0 = time()
    cat_vec = cat_vectorizer.fit_transform(cat_text)
    print("vectorize done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % cat_vec.shape)

    clf = KNeighborsClassifier( n_neighbors=5, weights='distance' )
    t0 = time()
    clf.fit( cat_vec[:len(hy_labels)], np.array(hy_labels) )
    print("category knn classifier training done in %fs" % (time() - t0))
    testing_proba = clf.predict_proba( cat_vec )

    for i,class_proba in enumerate(testing_proba):
        single = { "cat_knn":class_proba }
        mixture_proba.append(single)

    #calculate proba from rating
    reshape_rating = np.array( rating ).reshape(-1,1)
    clf = SVC(probability=True)
    t0 = time()
    clf.fit( reshape_rating[:len(hy_labels)] , np.array(hy_labels) )
    print("rating svm classifier training done in %fs" % (time() - t0))
    testing_proba = clf.predict_proba( reshape_rating )

    for i,class_proba in enumerate(testing_proba):
        mixture_proba[i]["rat_svm"] = class_proba

    #calculate proba from review number
    reshape_review_num = np.array( review_num ).reshape(-1,1)
    clf = SVC(probability=True)
    t0 = time()
    clf.fit( reshape_review_num[:len(hy_labels)] , np.array(hy_labels) )
    print("review number svm classifier training done in %fs" % (time() - t0))
    testing_proba = clf.predict_proba( reshape_review_num )

    for i,class_proba in enumerate(testing_proba):
        mixture_proba[i]["num_svm"] = class_proba

    #tfidf vectorize text
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english', use_idf=True)
    t0 = time()
    X = vectorizer.fit_transform(hy_text)
    print("vectorize done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    training_data = X[:len(hy_labels)]

    #train classifier, support vector machine or KNN (with topic probability)
    clf = KNeighborsClassifier( n_neighbors=5, weights='distance' )
    #clf = SVC(probability=True)
    t0 = time()
    clf.fit( training_data, np.array(hy_labels) )
    print("text knn classifier training done in %fs" % (time() - t0))

    #predict label of the testing_text and categories mixture
    testing_proba = clf.predict_proba( X )

    for i,class_proba in enumerate(testing_proba):
        mixture_proba[i]["text_knn"] = class_proba

    #log file for debug
    with open("testing_mixture_proba.log","w") as log_file:
        for single in mixture_proba:
            log_file.write( str(single) + "\n" )

    #classifier result
    with open(testing_label_path,"w") as testing_label_file:
        for single in mixture_proba:
            fail_chance = 1
            pass_chance = 1
            for key in single:
                if(key == "cat_knn"):
                    continue
                fail_chance = fail_chance * float(single[key][0])
                pass_chance = pass_chance * float(single[key][1])
            if(pass_chance >= fail_chance):
                testing_label_file.write( str(1) + "\n" )
            else:
                testing_label_file.write( str(0) + "\n" )


if __name__=="__main__":
    main()