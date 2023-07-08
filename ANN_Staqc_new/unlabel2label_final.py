import os
import tensorflow as tf
import numpy as np
import random
import pickle
import argparse
import logging
from sklearn.metrics import *

# Set random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

from configs import *
from models_code import CodeMF

class StandaloneCode:
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self._buckets = conf.get('buckets', [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)])
        self._buckets_text_max = (max([i for i, _, _, _ in self._buckets]), max([j for _, j, _, _ in self._buckets]))
        self._buckets_code_max = (max([i for _, _, i, _ in self._buckets]), max([j for _, _, _, j in self._buckets]))
        self.path = self.conf.get('workdir', './data/')
        self.train_params = conf.get('training_params', dict())
        self.data_params = conf.get('data_params', dict())
        self.model_params = conf.get('model_params', dict())
        self._eval_sets = None

    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            word_dict = pickle.load(f)
        return word_dict

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    def save_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        if not os.path.exists(self.path + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.path + 'models/' + self.model_params['model_name'] + '/')
        model.save("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
            self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch), overwrite=True)

    def load_model_epoch(self, model, epoch, d12, d3, d4, d5, r):
        print(self.path)
        print("{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
            self.path, d12, d3, d4, d5, r, epoch))
        assert os.path.exists("{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
            self.path, d12, d3, d4, d5, r, epoch)), "Weights at epoch {:d} not found".format(epoch)
        model.load("{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
            self.path, d12, d3, d4, d5, r, epoch))

    def del_pre_model(self, prepoch, d12, d3, d4, d5, r):
        if len((prepoch)) >= 2:
            lenth = len(prepoch)
            epoch = prepoch[lenth - 2]
            if os.path.exists("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
                    self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch)):
                os.remove("{}models/{}/pysparams:d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(
                    self.path, self.model_params['model_name'], d12, d3, d4, d5, r, epoch))

    def process_instance(self, instance, target, maxlen):
        w = self.pad(instance, maxlen)
        target.append(w)

    def process_matrix(self, inputs, trans1_length, maxlen):
        inputs_trans1 = np.split(inputs, trans1_length, axis=1)
        processed_inputs = []
        for item in inputs_trans1:
            item_trans2 = np.squeeze(item, axis=1).tolist()
            processed_inputs.append(item_trans2)
        return processed_inputs

    def get_data(self, path):
        data = self.load_pickle(path)
        text_S1 = []
        text_S2 = []
        code = []
        queries = []
        labels = []
        ids = []

        text_block_length, text_word_length, query_word_length, code_token_length = 2, 100, 25, 350
        text_blocks = self.process_matrix(np.array([samples_term[1] for samples_term in data]),
                                          text_block_length, 100)

        text_S1 = text_blocks[0]
        text_S2 = text_blocks[1]

        code_blocks = self.process_matrix(np.array([samples_term[2] for samples_term in data]),
                                          text_block_length - 1, 350)
        code = code_blocks[0]

        queries = [samples_term[3] for samples_term in data]
        labels = [samples_term[5] for samples_term in data]
        ids = [samples_term[0] for samples_term in data]

        return text_S1, text_S2, code, queries, labels, ids

    def eval(self, model, path):
        text_S1, text_S2, code, queries, labels, ids = self.get_data(path)
        labelpred = model.predict([np.array(text_S1), np.array(text_S2)