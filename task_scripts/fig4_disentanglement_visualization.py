import sys
sys.path.append('..')
print(sys.path)
import numpy as np
from MainClasses.AttSBetaVAE import AttSBetaVAE
from MainClasses.Dataset import Dataset
import keras.backend as K
import random
from argparse import ArgumentParser
import matplotlib.pylab as plt


def setup_args():
    parser = ArgumentParser(description='Fig. 4: disentanglement visualization')
    # NOTE: parameters below are not supposed to change.
    DATASET = 'freesound'
    TIMESTEP = 5
    NAME = 'att_s_beta_vae'
    NFOLDS = 4
    # NUM_EVENTS = 5
    parser.add_argument('-name', "--name", type=str, default=NAME)
    parser.add_argument('-nfolds', "--nfolds", type=int, default=NFOLDS)
    parser.add_argument("-dt", "--dataset", type=str, default=DATASET)
    parser.add_argument('-t', "--time_step", type=int, default=TIMESTEP)
    parser.add_argument('-o', "--result_path", type=str, default='../aed_data/' + DATASET + '/result/')
    parser.add_argument('-k', "--num_events", type=int, default=5)

    # Parameters below are changeable.
    parser.add_argument('-p', '--patience', type=int, default=100)
    parser.add_argument('-d', "--feature_dim", type=int, default=216)
    parser.add_argument('-z', "--latents_dim", type=int, default=15)
    parser.add_argument('-b', "--batch_size", type=int, default=100)
    parser.add_argument('-e', "--epoch", type=int, default=1)
    parser.add_argument('-l', "--learning_rate", type=int, default=0.0002)
    parser.add_argument('-n', '--num_samples', type=int, default=2000)
    parser.add_argument('-beta', "--beta", type=int, default=4)
    parser.add_argument('-lambda', "--lambda_", type=int, default=1)
    parser.add_argument('-gpu', "--gpu_device", type=str, default='0')
    parser.add_argument('-m', "--mix_data", type=int, default=0)  # if generate new sub-dataset using freesound
    parser.add_argument('-v', "--verbose", type=int, default=2)  # keras training verbose, only change the print format.
    return parser


def running(options):
    fold = 2
    num_to_generate = 250
    dt = Dataset(options.dataset)
    train_datas, train_labels, test_datas, test_labels \
        = dt.load_data(nevents=options.num_events, nsamples=options.num_samples, fold=fold)

    # setup the configure list, which decides the events and the latent factors to visualize:
    # [event index, latent factor index, min value, mid value, max value]
    # Note please: you are supposed to alter the event index and latent factor index to find the most clear samples.
    infs = []
    infs.append([1, 6, 0, 0, 0])
    infs.append([1, 14, 0, 0, 0])
    infs.append([2, 6, 0, 0, 0])
    infs.append([2, 14, 0, 0, 0])
    infs.append([4, 6, 0, 0, 0])
    infs.append([4, 14, 0, 0, 0])
    infs = np.asarray(infs, dtype='float')

    # load testing set
    sequential_test_datas = dt.sequentialize_data(test_datas, timestep=options.time_step)
    random.shuffle(sequential_test_datas)

    # build model and load weights
    att_s_beta_vae = AttSBetaVAE(options)
    K.reset_uids()
    sbvae = att_s_beta_vae.build_model(options)
    sbvae.load_weights(options.result_path + sbvae.name + '/fold_' + str(fold) + '_last_weight.h5')
    pic = []
    min_plot, mid_plot, max_plot = [], [], []
    plots = [min_plot, mid_plot, max_plot]
    for j in range(len(infs)):
        # get event index
        event_index = int(infs[j][0])
        gen_label = np.zeros((options.num_events,))
        gen_label[event_index] = 1
        indexs = list()
        for index in range(len(test_labels)):
            if test_labels[index][event_index] == gen_label[event_index]:
                indexs.append(index)

        # get z* and decode it to reconstruct MFCCs
        z_star_fnc = K.Function([sbvae.input], [sbvae.layers[14 + event_index].output])
        decoder_fnc = K.Function([sbvae.layers[6].output], [sbvae.output[0]])
        event_num = z_star_fnc([sequential_test_datas[indexs]])[0]

        dim = int(infs[j][1])
        max_z = max(event_num[:, dim])
        min_z = min(event_num[:, dim])
        mid_z = (max_z + min_z) / 2.0
        infs[j][2] = min_z
        infs[j][3] = mid_z
        infs[j][4] = max_z

        # Change the value of event-specific factor to visualize the disentanglement results
        idx = 0
        for value in [min_z, mid_z, max_z]:
            event_num[:, dim] = value
            generated_datas = decoder_fnc([event_num])[0]
            for i in range(0, num_to_generate):
                x = generated_datas[i][(options.num_events - 1) * options.feature_dim:]
                x = np.reshape(x, (options.feature_dim, 1))
                if i == 0:
                    plots[idx] = x
                else:
                    plots[idx] = np.concatenate([plots[idx], x], axis=1)
            idx += 1

        # delta(max, mid)
        pic.append(np.abs(plots[2] - plots[1]))
        # delta(mid, min)
        pic.append(np.abs(plots[1] - plots[0]))
    # plot delta maps
    plt.figure(figsize=(20, 15))
    for p in range(len(pic)):
        plt.subplot(3, 4, p + 1)
        plt.imshow(pic[p])

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.025, 0.8])
    plt.colorbar(cax=cax)
    plt.show()


def clear_up():
    K.clear_session()


if __name__ == '__main__':
    args = setup_args()
    options = args.parse_args()
    running(options)
    clear_up()
