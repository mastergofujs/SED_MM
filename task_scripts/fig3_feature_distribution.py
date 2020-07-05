import sys
sys.path.append('..')
print(sys.path)
import numpy as np
from MainClasses.AttSBetaVAE import AttSBetaVAE
from MainClasses.Dataset import Dataset
import keras.backend as K
import random
from argparse import ArgumentParser


def setup_args():
    parser = ArgumentParser(description='Fig. 3: visualize the distribution of features learned from the proposed method')
    # NOTE: parameters below are not supposed to change.
    DATASET = 'tut_data'
    TIMESTEP = 5
    NAME = 'att_s_beta_vae'
    NFOLDS = 4
    NUM_EVENTS = 6

    parser.add_argument('-name', "--name", type=str, default=NAME)
    parser.add_argument('-nfolds', "--nfolds", type=int, default=NFOLDS)
    parser.add_argument("-dt", "--dataset", type=str, default=DATASET)
    parser.add_argument('-t', "--time_step", type=int, default=TIMESTEP)
    parser.add_argument('-o', "--result_path", type=str, default='../aed_data/' + DATASET + '/result/')
    parser.add_argument('-k', "--num_events", type=int, default=NUM_EVENTS)

    # Parameters below are changeable.
    parser.add_argument('-p', '--patience', type=int, default=100)
    parser.add_argument('-d', "--feature_dim", type=int, default=200)
    parser.add_argument('-z', "--latents_dim", type=int, default=15)
    parser.add_argument('-b', "--batch_size", type=int, default=100)
    parser.add_argument('-e', "--epoch", type=int, default=1)
    parser.add_argument('-l', "--learning_rate", type=int, default=0.0002)
    parser.add_argument('-n', '--num_samples', type=int, default=2000)
    parser.add_argument('-beta', "--beta", type=int, default=3)
    parser.add_argument('-lambda', "--lambda_", type=int, default=2)
    parser.add_argument('-gpu', "--gpu_device", type=str, default='0')
    parser.add_argument('-m', "--mix_data", type=int, default=0)  # if generate new sub-dataset using freesound
    parser.add_argument('-v', "--verbose", type=int, default=2)  # keras training verbose, only change the print format.
    return parser


def running(options):
    fold = 1
    dt = Dataset(options.dataset)
    # load data
    train_datas, train_labels, val_datas, val_labels, test_datas, test_labels \
        = dt.load_data(nevents=options.num_events, nsamples=options.num_samples, fold=fold)
    sequential_test_datas = dt.sequentialize_data(test_datas, timestep=options.time_step)
    random.shuffle(sequential_test_datas)

    # build model and load weights
    att_s_beta_vae = AttSBetaVAE(options)
    K.reset_uids()
    sbvae = att_s_beta_vae.build_model(options)
    sbvae.load_weights(options.result_path + sbvae.name + '/fold_' + str(fold) + '_last_weight.h5')

    # get the bottleneck features
    h_out = np.empty((1, 15))
    num_to_plot = 300
    for i in range(options.num_events):
        h_fnc = K.Function([sbvae.input], [sbvae.layers[15 + i].output])
        h = h_fnc([sequential_test_datas[:num_to_plot]])[0]
        h_out = np.concatenate([h_out, h])

    # visualization
    dt.visualization(datas=h_out[1:], name='Att_s_beta_VAE')


def clear_up():
    K.clear_session()


if __name__ == '__main__':
    args = setup_args()
    options = args.parse_args()
    running(options)
    clear_up()
