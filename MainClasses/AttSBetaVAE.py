from MainClasses.Models import *
from keras.metrics import binary_crossentropy
from keras.utils import plot_model
import keras.backend as K

'''
This class is inherited from the MODELS, which re-defines the architecture of our proposed attention-based 
supervised beta-VAE. Besides, our proposed model is also one of the generative model, so we also give the 
generative_data function to augment the unbanlced data.
'''


class AttSBetaVAE(MODELS):
    def __init__(self, options):
        MODELS.__init__(self, options)
        self.op = options

    def build_model(self, options):
        nstep = int(self.nlatents / self.nevents)

        # With the reparameterization trick, sample the latent z from z_mean and z_log_var
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.nlatents),
                                      mean=0.0, stddev=1.0, dtype=None, seed=None)
            z = z_mean + K.exp(z_log_var / 2) * epsilon
            return z

        # The proposed novel disentangling loss, mentioned at paper Eq.6.
        def loss_disent(y_true, y_pred):
            no = int(y_true.name.split('_')[0][1])
            z_star = vae.layers[8 + self.nevents + no].output
            var_square = K.square(K.var(z_star, axis=-1))
            mean_square = K.square(K.mean(z_star, axis=-1))
            loss_aed = binary_crossentropy(y_true, y_pred)
            loss_kl = 0.5 * K.sum(mean_square + var_square - 1 - K.log(var_square), axis=-1)
            return options.beta * options.latents_dim / options.feature_dim * loss_kl + loss_aed

        # The metrics for reconstruction.
        def R_square(y_true, y_pred):
            y_mean = K.mean(y_true)
            SSE = K.sum(K.square(y_true - y_pred))
            SST = K.sum(K.square(y_true - y_mean))
            R2 = 1 - SSE / SST
            return R2

        def encoder_layer(inputs):
            x = Reshape(target_shape=(self.timestep, self.input_dim))(inputs)
            x = LSTM(256, return_sequences=True)(x)
            x = LSTM(128, return_sequences=False)(x)
            # Latent Variables
            z_mean = Dense(self.nlatents)(x)
            z_log_var = Dense(self.nlatents)(x)
            z = Lambda(sampling, [self.nlatents, ])([z_mean, z_log_var])
            return z_mean, z_log_var, z

        def attention_layer(x):
            att_probs = Dense(self.nlatents, activation='softmax')(x)
            ev = Multiply()([x, att_probs])
            # ev = BatchNormalization()(ev)
            return ev

        def decoder_layer(z):
            x = RepeatVector(self.timestep)(z)
            x = LSTM(128, return_sequences=True)(x)
            x = LSTM(self.input_dim, return_sequences=True, activation='sigmoid')(x)
            output = Reshape((self.input_dim * self.timestep,), name='output')(x)
            return output

        def event_detector(ev):
            x = Dense(32, activation='relu')(ev)
            ev_out = Dense(1, activation='sigmoid', name='e' + str(int(ev.name.split('/')[0].split('_')[1]))
                                                         + '_out')(x)
            return ev_out

        event_names = locals()
        # Encoder
        input = Input(shape=[self.input_dim * self.timestep, ], name='input')
        z_mean, z_log_var, z = encoder_layer(input)

        # Disentangling_layer
        for n in range(self.nevents):
            event_names['e' + str(n + 1)] = attention_layer(z)

        # Decoder
        output = decoder_layer(z)

        # detector
        for n in range(self.nevents):
            event_names['e' + str(n + 1) + '_out'] = event_detector(event_names['e' + str(n + 1)])

        # outputs
        outputs = [output]
        for n in range(self.nevents):
            outputs.append(event_names['e' + str(n + 1) + '_out'])
        name = self.name + '_' + str(self.nevents)
        vae = Model(inputs=input, outputs=outputs, name=name)
        adam = Adam(lr=self.lr)
        vae.summary()
        plot_model(vae, to_file='../model_figures/' + vae.name + '.png', show_shapes=True)
        vae.compile(optimizer=adam,
                    loss=['binary_crossentropy'] + [loss_disent] * self.nevents,
                    loss_weights=[1] + [(options.lambda_ * self.nevents / self.nlatents)
                                        / self.nevents] * self.nevents,
                    metrics=[R_square] + ['binary_accuracy']
                    )
        return vae

    # This function can generate new data for specific event according to the 'event_index from the 'input_datas'.
    def generate_data(self, model, event_index, input_data):
        model = model
        model.load_weights(self.op.result_path + self.op.name + '_'
                           + str(self.op.nevents) + '/cp_weight.h5')
        # Get the output of hidden layers.
        # e_fnc: gets the z* of target event;
        # decoder_fnc gets the output of decoder, which reconstruct the given inputs.

        e_fnc = K.Function([model.input], [model.layers[8 + self.nevents + event_index + 1].output])
        decoder_fnc = K.Function([model.layers[6].output], [model.output[0]])

        event_num = e_fnc([input_data])[0]
        decoder_num = decoder_fnc([event_num])[0]

        gen_datas = decoder_num

        return gen_datas

