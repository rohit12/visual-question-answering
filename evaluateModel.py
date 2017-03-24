import sys
import argparse
from progressbar import Bar, ETA, Percentage, ProgressBar
from keras.models import load_model

from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

from features import get_questions_matrix_sum, get_images_matrix, get_answers_matrix
from utils import grouper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-results', type=str, required=True)
    args = parser.parse_args()

    model = load_model(args.model)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    questions_val = open('./data/questions_val2014.txt',
                         'r').read().decode('utf8').splitlines()
    answers_val = open('./data/answers_val2014_all.txt',
                       'r').read().decode('utf8').splitlines()
    images_val = open('./data/images_val2014_all.txt',
                      'r').read().decode('utf8').splitlines()

    print 'Model compiled, weights loaded...'
    labelencoder = joblib.load('./labelencoder.pkl')

    image_ids = open("./id_map.txt").read().splitlines()
    id_map = {}
    for ids in image_ids:
        id_split = ids.split()
        id_map[int(id_split[0])] = int(id_split[1]) - 1

    sherlock_features = np.load('./sherlock_features_temp.npy')

    nlp = English()
    print 'loaded word2vec features'

    nb_classes = 1000
    y_predict_text = []
    batchSize = 128
    widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#', left='[', right=']'),
               ' ', ETA()]
    pbar = ProgressBar(widgets=widgets)

    for qu_batch, an_batch, im_batch in pbar(zip(grouper(questions_val, batchSize, fillvalue=questions_val[0]),
                                                 grouper(answers_val, batchSize, fillvalue=answers_val[0]),
                                                 grouper(images_val, batchSize, fillvalue=images_val[0]))):
        X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
        X_i_batch = get_images_matrix(im_batch, id_map, sherlock_features)
        X_batch = np.hstack((X_q_batch, X_i_batch))
        y_predict = model.predict_classes(X_batch, verbose=0)
        y_predict_text.extend(labelencoder.inverse_transform(y_predict))

    correct_val = 0.0
    total = 0
    f1 = open(args.results, 'w')

    for prediction, truth, question, image in zip(y_predict_text, answers_val, questions_val, images_val):
        temp_count = 0
        for _truth in truth.split(';'):
            if prediction == _truth:
                temp_count += 1

        if temp_count > 2:
            correct_val += 1
        else:
            correct_val += float(temp_count) / 3

        total += 1
        f1.write(question.encode('utf-8'))
        f1.write('\n')
        f1.write(image.encode('utf-8'))
        f1.write('\n')
        f1.write(prediction)
        f1.write('\n')
        f1.write(truth.encode('utf-8'))
        f1.write('\n')
        f1.write('\n')

    f1.write('Final Accuracy is ' + str(correct_val / total))
    f1.close()
    f1 = open('../results/overall_results.txt', 'a')
    f1.write(args.weights + '\n')
    f1.write(str(correct_val / total) + '\n')
    f1.close()
    print 'Final Accuracy on the validation set is', correct_val / total


if __name__ == "__main__":
    main()
