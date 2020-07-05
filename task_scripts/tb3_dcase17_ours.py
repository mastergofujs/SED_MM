import sys
sys.path.append('..')
print(sys.path)
import numpy as np
from MainClasses.AttSBetaVAE import AttSBetaVAE
from MainClasses.Dataset import Dataset
import pandas as pd
import keras.backend as K
from argparse import ArgumentParser


def setup_args():
    parser = ArgumentParser(description='Table 3: F1 and ER comparison on the DCASE 2017 TUT dataset')
    # NOTE: parameters below are not supposed to change.
    DATASET = 'tut_data'
    TIMESTEP = 5
    NAME = 'att_s_beta_vae'
    NFOLDS = 4
    parser.add_argument('-name', "--name", type=str, default=NAME)
    parser.add_argument('-nfolds', "--nfolds", type=int, default=NFOLDS)
    parser.add_argument("-dt", "--dataset", type=str, default=DATASET)
    parser.add_argument('-t', "--time_step", type=int, default=TIMESTEP)
    parser.add_argument('-o', "--result_path", type=str, default='../aed_data/' + DATASET + '/result/')

    # Parameters below are changeable.
    parser.add_argument('-p', '--patience', type=int, default=100)
    parser.add_argument('-d', "--feature_dim", type=int, default=200)
    parser.add_argument('-z', "--latents_dim", type=int, default=15)
    parser.add_argument('-b', "--batch_size", type=int, default=100)
    parser.add_argument('-e', "--epoch", type=int, default=200)
    parser.add_argument('-l', "--learning_rate", type=int, default=0.0002)
    parser.add_argument('-k', "--num_events", type=int, default=6)
    parser.add_argument('-n', '--num_samples', type=int, default=2000)
    parser.add_argument('-beta', "--beta", type=int, default=3)
    parser.add_argument('-lambda', "--lambda_", type=int, default=2)
    parser.add_argument('-gpu', "--gpu_device", type=str, default='0')
    parser.add_argument('-m', "--mix_data", type=int, default=0)  # if generate new sub-dataset using freesound
    parser.add_argument('-v', "--verbose", type=int, default=2)  # keras training verbose, only change the print format.
    return parser


def running(options):
    dt = Dataset(options.dataset)
    folds = options.nfolds
    # 1.First construct polyphonic datasets by mixing single event sound, and extract MFCCs features;
    # here, TUT dataset do not need to construct datas.
    if options.mix_data:
        dt.mix_data(nevents=options.num_events, nsamples=options.num_samples)
    f1_list, er_list, fold_list = [], [], []

    att_s_beta_vae = AttSBetaVAE(options)
    for k in range(1, folds + 1):
        # 2.Load data.
        train_datas, train_labels, val_datas, val_labels, test_datas, test_labels \
            = dt.load_data(nevents=options.num_events, nsamples=options.num_samples, fold=k)
        sequential_train_datas = dt.sequentialize_data(train_datas, timestep=options.time_step)
        sequential_val_datas = dt.sequentialize_data(val_datas, timestep=options.time_step)
        sequential_test_datas = dt.sequentialize_data(test_datas, timestep=options.time_step)

        # 3.Create attention-based supervised beta-VAE model and train it.
        K.reset_uids()
        model = att_s_beta_vae.build_model(options)
        att_s_beta_vae.train_model(model, x_train=sequential_train_datas, y_train=train_labels,
                                   x_val=sequential_val_datas, y_val=val_labels, fold=k)

        # 4.Evaluate the performance on F1 and ER
        # Param: supervised is set to 'False' default. 'True' for supervised beta-VAE and 'False' for others.
        # This function evaluate the segment-based F1 score and ER.
        f1_score, error_rate = att_s_beta_vae.metric_model(model, sequential_test_datas, test_labels,
                                                           supervised=True, fold=k,
                                                           new_weight_path='/fold_' + str(k) + '_' + 'last_weight.h5'
                                                           )

        print('Fold {fold}, nevents {nevents}, nsamples {nsamples} ==> error_rate: {error_rate}, f1_score: {f1_score}'
              .format(fold=k, nevents=options.num_events, nsamples=options.num_samples,
                      error_rate=error_rate, f1_score=f1_score))
        f1_list.append(f1_score)
        er_list.append(error_rate)
        fold_list.append(k)
        del model
    f1_list.append(np.mean(f1_list))
    er_list.append(np.mean(er_list))
    fold_list.append('AVER')
    result_df = pd.DataFrame({'F1': f1_list, 'ER': er_list}, index=fold_list)
    result_df.to_csv(options.result_path + options.name + '_' + str(options.num_events) + '/K_Folds_results.csv')
    return result_df


def clear_up():
    K.get_session().close()


if __name__ == '__main__':
    args = setup_args()
    options = args.parse_args()
    evaluation_results = running(options)
    print(evaluation_results)

    clear_up()
