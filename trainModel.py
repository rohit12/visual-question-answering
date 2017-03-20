import sys
from random import shuffle
import argparse

import numpy as np
import logging
import scipy.io
from keras.utils import np_utils, generic_utils
from keras.models import load_model

from sklearn import preprocessing
from sklearn.externals import joblib

from spacy.en import English

from features import get_questions_matrix_sum, get_images_matrix, get_answers_matrix
from utils import grouper, selectFrequentAnswers

logging.basicConfig(level=logging.DEBUG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-activation', type=str, default='tanh')
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-model_save_interval', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=128)
    args = parser.parse_args()

    questions_train = open('./data/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
    answers_train = open('./data/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
    images_train = open('./data/images_train2014.txt', 'r').read().decode('utf8').splitlines()

    logging.debug("Length of questions_train %d", len(questions_train))
    logging.debug("Length of answers_train %d", len(answers_train))
    logging.debug("Length of images_train %d", len(images_train))

    maxAnswers = 1000
    questions_train, answers_train, images_train = selectFrequentAnswers(questions_train, answers_train, images_train,
                                                                         maxAnswers)

    logging.debug("Length of the lists after select Frequent Answers")
    logging.debug("Length of questions_train %d", len(questions_train))
    logging.debug("Length of answers_train %d", len(answers_train))
    logging.debug("Length of images_train %d", len(images_train))

    # generates numerical labels for all the answers in answers_train.
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(answers_train)
    nb_classes = len(list(labelencoder.classes_))
    joblib.dump(labelencoder, 'labelencoder.pkl')

    # TODO Get vectors for each image from Sherlock and load them into an array here
    image_ids = open("./id_map.txt").read().splitlines()
    id_map = {}
    for ids in image_ids:
        id_split = ids.split()
        id_map[int(id_split[0])] = int(id_split[1])

    sherlock_features = np.load('./sherlock_features.npy')

    # Load Word2Vec
    nlp = English()
    logging.debug("Word2Vec is loaded")
    img_dim = 900
    word_vec_dim = 300

    model = load_model('NaiveSherlock.hd5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    logging.debug("Model compiled")

    for i in xrange(args.num_epochs):
        progbar = generic_utils.Progbar(len(questions_train))
        for qu_batch, an_batch, im_batch in zip(
                grouper(questions_train, args.batch_size, fillvalue=questions_train[-1]),
                grouper(answers_train, args.batch_size, fillvalue=answers_train[-1]),
                grouper(images_train, args.batch_size, fillvalue=images_train[-1])):
            logging.debug("One batch done")
            x_q_batch = get_questions_matrix_sum(qu_batch, nlp)
            logging.debug("length of qu_batch is %d", len(qu_batch))
            logging.debug("Shape of x_q_batch is: %s", x_q_batch.shape)
            x_i_batch = get_images_matrix(im_batch, id_map, sherlock_features)
            logging.debug("shape of x_i_batch is %s", x_i_batch.shape)
            x_batch = np.hstack((x_q_batch, x_i_batch))
            y_batch = get_answers_matrix(an_batch, labelencoder)
            loss = model.train_on_batch(x_batch, y_batch)
            progbar.add(args.batch_size, values=[("train_loss", loss)])


if __name__ == '__main__':
    main()
