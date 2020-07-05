from keras import Model
from keras.optimizers import Adam
import pandas as pd
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np

'''
This class defines a general Model, which can be inherited to specific models.
Here you can init the parameters used in models and build a Keras Model to be trained and evaluated. 
Note that, the metric_model() function is totally same as the DCASE 2017 AED challenge, which the F1
and ER are calculated based on a segment of one second.
'''


class MODELS:
    def __init__(self, options):
        self.lr = options.learning_rate
        self.nlatents = options.latents_dim
        self.batch_size = options.batch_size
        self.epoch = options.epoch
        self.verbose = options.verbose
        self.nevents = options.num_events
        self.epochs = options.epoch
        self.input_dim = options.feature_dim
        self.timestep = options.time_step
        self.result_path = options.result_path
        self.name = options.name
        self.patience = options.patience
        os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu_device

    def train_model(self, model, x_train, y_train, fold, x_val=None, y_val=None, new_weights=None):
        # Sence this is a multi-outputs model, we should construct the ground truth feed-dict.
        y_true_trian = {'output': x_train}
        for n in range(self.nevents):
            y_true_trian['e' + str(n + 1) + '_out'] = y_train[self.timestep - 1:, n]

        if new_weights is not None:
            weights_path = new_weights
        else:
            weights_path = self.result_path + model.name + '/fold_' + str(fold) + '_last_weight.h5'

        if x_val is not None and y_val is not None:
            cp = ModelCheckpoint(self.result_path + model.name + '/fold_' + str(fold) + '_cp_weight.h5',
                                 save_best_only=True, save_weights_only=True, verbose=1)
            early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, mode='min')
            y_true_val = {'output': x_val}
            for n in range(self.nevents):
                y_true_val['e' + str(n + 1) + '_out'] = y_val[self.timestep - 1:, n]
            hist = model.fit(x_train, y_true_trian, epochs=self.epoch, verbose=self.verbose,
                             validation_data=(x_val, y_true_val), batch_size=self.batch_size, shuffle=True,
                             callbacks=[cp, early_stop])
        else:
            hist = model.fit(x_train, y_true_trian, epochs=self.epoch, verbose=self.verbose,
                             batch_size=self.batch_size, shuffle=True)
        hist_path = self.result_path + model.name + '/hist.pkl'
        print(weights_path)
        model.save_weights(weights_path)
        with open(hist_path, 'wb') as f:
            pkl.dump(hist.history, f)
        print(model.name + ' Model trained Done!')

    # parameter 'new_weight_path' is prepared for Data augmentation.
    #           'supervised': True for our supervised beta-VAE, False for others.
    def metric_model(self, model, x, y, supervised=False, new_weight_path=None, fold=0):
        y_true_test = {'output': x}
        for n in range(self.nevents):
            y_true_test['e' + str(n + 1) + '_out'] = y[self.timestep - 1:, n]

        def frame_to_segments(event_label):
            total_s = int(np.ceil(len(event_label) / 50))  # 0.02s per frame, 50 frames = 1.0s
            segments = np.zeros((total_s, self.nevents), np.int8)

            for i in range(total_s):
                if (i + 1) * 50 > len(event_label):
                    res = np.zeros(((i + 1) * 50 - len(event_label), self.nevents))
                    event_label = np.concatenate((event_label, res))
                per_s = event_label[i * 50:(i + 1) * 50]
                for j in range(self.nevents):
                    if sum(per_s[:, j]) > 0:
                        segments[i, j] = 1
            return segments

        def count_factors(y_pers, y_hat_pers, overall):
            Ntp = sum(y_hat_pers + y_pers > 1)
            Ntn = sum(y_hat_pers + y_pers == 0)
            Nfp = sum(y_hat_pers - y_pers > 0)
            Nfn = sum(y_pers - y_hat_pers > 0)

            Nref = sum(y_pers)
            Nsys = sum(y_hat_pers)

            S = min(Nref, Nsys) - Ntp
            D = max(0, Nref - Nsys)
            I = max(0, Nsys - Nref)

            overall['Ntp'] += Ntp
            overall['Ntn'] += Ntn
            overall['Nfp'] += Nfp
            overall['Nfn'] += Nfn
            overall['Nref'] += Nref
            overall['Nsys'] += Nsys
            overall['S'] += S
            overall['D'] += D
            overall['I'] += I

        if not new_weight_path:
            model.load_weights(self.result_path + model.name + '/fold_' + str(fold) + '_cp_weight.h5')
        else:
            model.load_weights(self.result_path + model.name + '/' + new_weight_path)
        y_hat = model.predict(x, batch_size=self.batch_size, verbose=0)
        if supervised:
            events_num = len(y_hat[1:])
            y_hat_ = y_hat[1]
            for i in range(2, events_num + 1):
                y_hat_ = np.concatenate([y_hat_, y_hat[i]], axis=1)
            y_hat = y_hat_
        y_hat_filted = np.zeros((len(y_hat), self.nevents), np.int8)
        for i in range(self.nevents):
            activaty_array = y_hat[:, i] > 0.5
            event_label = medfilt(volume=activaty_array.T, kernel_size=27)  # 0.54s做滤波
            event_label = np.array(event_label, np.int8)
            y_hat_filted[:, i] = event_label
        y_hat_segments = frame_to_segments(y_hat_filted)
        y_segments = frame_to_segments(y)
        overall = {
            'Ntp': 0.0,
            'Ntn': 0.0,
            'Nfp': 0.0,
            'Nfn': 0.0,
            'Nref': 0.0,
            'Nsys': 0.0,
            'ER': 0.0,
            'S': 0.0,
            'D': 0.0,
            'I': 0.0,
        }
        for i in range(len(y_hat_segments)):
            count_factors(y_segments[i], y_hat_segments[i], overall)
        # calculate F1
        if overall['Nsys'] == 0:
            precision = 0
        precision = overall['Ntp'] / float(overall['Nsys'])
        recall = overall['Ntp'] / float(overall['Nref'])
        f1_score = 2 * precision * recall / (precision + recall)
        # calculate error
        eps = np.spacing(1)
        substitution_rate = float(overall['S'] / (overall['Nref'] + eps))
        deletion_rate = float(overall['D'] / (overall['Nref'] + eps))
        insertion_rate = float(overall['I'] / (overall['Nref'] + eps))
        error_rate = float(substitution_rate + deletion_rate + insertion_rate)
        return f1_score, error_rate
