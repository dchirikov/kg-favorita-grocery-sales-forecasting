import tensorflow as tf
import datetime
import numpy as np
#import hyperdash

class RNNModel(object):

    def __init__(self, history=200, n_ts_attr=12, n_attr=397,
                batch_size=512,
                n_days_predict=16,
                #max_source_length=400,
                n_layers_rnn=2, rnn_size_encoder=200, rnn_size_decoder=200,
                starter_learning_rate=0.0005, clip_gradients=1.,
                path_to_store='log',
                output_droupouts_kp=[.7, .7, .9, 1., 1.],
                encoder_dropout_kp=0.7, decoder_dropout_kp=0.7):

        self.history = history
        self.n_ts_attr = n_ts_attr
        self.n_attr = n_attr
        self.n_days_predict = n_days_predict
        self.batch_size = batch_size

        #self.max_source_length = max_source_length

        self.clip_gradients = clip_gradients

        self.n_layers_rnn = n_layers_rnn
        self.rnn_size_encoder = rnn_size_encoder
        self.rnn_size_decoder = rnn_size_decoder
        self.encoder_dropout_kp = encoder_dropout_kp
        self.decoder_dropout_kp = decoder_dropout_kp
        self.output_droupouts_kp = output_droupouts_kp

        self.starter_learning_rate = starter_learning_rate

        self.path_to_store = path_to_store

        self.g_step = 0

        self.encoder_activation = tf.nn.tanh
        self.decoder_activation = tf.nn.tanh
        self.output_activation = tf.nn.selu


        now = datetime.datetime.now().strftime(format="%Y-%m-%d_%H-%M-%S")

        self.path = self.path_to_store + '/' + now

        #self.hd_exp = hyperdash.Experiment("RNN " + now)


    def init_inputs(self):

        self.t_X = tf.placeholder(
            tf.float32,
            [None, self.history, self.n_ts_attr],
            name="t_X"
        )

        self.t_y_day_attr = tf.placeholder(
            tf.int8,
            [None, self.n_days_predict, self.n_ts_attr-1],
            name="t_y_day_attr"
        )

        self.t_y = tf.placeholder(
            tf.float32,
            [None, self.n_days_predict],
            name="t_y"
        )

        self.t_feat_store_nbr = tf.placeholder(
            tf.int32,
            [None],
            name="t_feat_store_nbr"
        )

        self.t_feat_n_city = tf.placeholder(
            tf.int8,
            [None],
            name="t_feat_n_city"
        )

        self.t_feat_n_state = tf.placeholder(
            tf.int8,
            [None],
            name="t_feat_n_state"
        )

        self.t_feat_n_type = tf.placeholder(
            tf.int8,
            [None],
            name="t_feat_n_type"
        )

        self.t_feat_cluster = tf.placeholder(
            tf.int8,
            [None],
            name="t_feat_cluster"
        )

        self.t_feat_item_nbr = tf.placeholder(
            tf.int32,
            [None],
            name="t_feat_item_nbr"
        )

        self.t_feat_n_family = tf.placeholder(
            tf.int8,
            [None],
            name="t_feat_n_family"
        )

        self.t_feat_class = tf.placeholder(
            tf.int16,
            [None],
            name="t_feat_class"
        )

        #
        # validation inputs
        #

        self.v_t_X = tf.placeholder(
            tf.float32,
            [None, self.history, self.n_ts_attr],
            name="v_t_X"
        )

        self.v_t_y_day_attr = tf.placeholder(
            tf.int8,
            [None, self.n_days_predict, self.n_ts_attr-1],
            name="v_t_y_day_attr"
        )

        self.v_t_y = tf.placeholder(
            tf.float32,
            [None, self.n_days_predict],
            name="v_t_y"
        )

        self.v_t_feat_store_nbr = tf.placeholder(
            tf.int32,
            [None],
            name="v_t_feat_store_nbr"
        )

        self.v_t_feat_n_city = tf.placeholder(
            tf.int8,
            [None],
            name="v_t_feat_n_city"
        )

        self.v_t_feat_n_state = tf.placeholder(
            tf.int8,
            [None],
            name="v_t_feat_n_state"
        )

        self.v_t_feat_n_type = tf.placeholder(
            tf.int8,
            [None],
            name="v_t_feat_n_type"
        )

        self.v_t_feat_cluster = tf.placeholder(
            tf.int8,
            [None],
            name="v_t_feat_cluster"
        )

        self.v_t_feat_item_nbr = tf.placeholder(
            tf.int32,
            [None],
            name="v_t_feat_item_nbr"
        )

        self.v_t_feat_n_family = tf.placeholder(
            tf.int8,
            [None],
            name="v_t_feat_n_family"
        )

        self.v_t_feat_class = tf.placeholder(
            tf.int16,
            [None],
            name="v_t_feat_class"
        )

        # =====================

        #self.t_sequence_length = tf.placeholder_with_default(
        #   self.max_source_length, None, name='sequence_length')

        self.t_is_training =  tf.placeholder(
            tf.bool,  name="is_training")

        self.t_encoder_dropout_kp = tf.placeholder(
            tf.float32, name="enc_dropout")

        self.t_output_droupouts_kp = tf.placeholder(
            tf.float32, [5], name="output_droupouts")

        self.t_decoder_dropout_kp = tf.placeholder(
            tf.float32, name="dec_dropout")

        self.t_lr = tf.placeholder(
            tf.float32, [],  name="learning_rate")

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.NWRMSLE = tf.placeholder(tf.float32, name="NWRMSLE")
        tf.summary.scalar('NWRMSLE', self.NWRMSLE)

        self.saver = tf.train.Saver()

        return True
        #return t_ts_inputs, t_items_features, t_days_features, t_targets,

    def initiate_embeddings(self, d_items=20, d_stores=3):

        self.emb_feat_item_nbr_vars = tf.Variable(
            tf.random_uniform([2134244, d_items])
        )

        self.emb_feat_store_nbr_vars = tf.Variable(
            tf.random_uniform([54, d_stores])
        )

    def convert_inputs(self, t_ts_inputs, t_items_features):
        #t_inputs_plus_item_features = tf.concat(
        #    [
        #        t_ts_inputs,
        #        tf.reshape(
        #            tf.tile(
        #                t_items_features,
        #                [1, self.history]
        #            ),
        #            [-1, self.history, self.n_attr]
        #        )
        #    ],
        #    axis=2
        #)

        t_inputs_norm = tf.layers.batch_normalization(
            t_ts_inputs,
            #t_inputs_plus_item_features,
            axis=1,
            training=self.t_is_training
        )
        return t_inputs_norm

    def encoder(self, t_inputs):
        with tf.name_scope('encoder') as scope:
            stacked_encoder = []
            for _ in range(self.n_layers_rnn):
                t_enc_cell = tf.contrib.rnn.GRUCell(
                    self.rnn_size_encoder,
                    activation=self.encoder_activation,
                )
                t_enc_cell = tf.contrib.rnn.DropoutWrapper(
                    t_enc_cell, output_keep_prob=self.t_encoder_dropout_kp)
                stacked_encoder.append(t_enc_cell)

            t_enc_multicell = tf.contrib.rnn.MultiRNNCell(cells=stacked_encoder)

            _, t_encoder_state = tf.nn.dynamic_rnn(
                t_enc_multicell,
                t_inputs,
                #sequence_length=self.t_sequence_length,
                time_major=False,
                dtype=tf.float32)

        return t_encoder_state


    def output_layers(self, t_predictions,  t_y_day_attr, concatenated_emb):

        with tf.name_scope('output_layers'):

            t_y_day_attr = tf.to_float(
                tf.reshape(t_y_day_attr, shape=[-1, 11*self.n_days_predict])
            )

            t_concat = tf.concat(
                [
                    t_predictions,
                    t_y_day_attr,
                    concatenated_emb
                ],
                axis=1
            )

            t_norm = tf.contrib.layers.layer_norm(t_concat)

            t_output = tf.nn.dropout(t_norm, self.t_output_droupouts_kp[0])

            n_input = int(t_concat.shape[1]*2)
            t_output = tf.layers.dense(
                t_output,
                n_input,
                activation=self.output_activation,
                kernel_initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1 / n_input),
                )
            )

            t_output = tf.nn.dropout(t_output, self.t_output_droupouts_kp[1])

            n_input = 500
            t_output = tf.layers.dense(
                t_output,
                n_input,
                activation=self.output_activation,
                kernel_initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1 / n_input),
                )
            )

            t_output = tf.nn.dropout(t_output, self.t_output_droupouts_kp[2])

            n_input = 300
            t_output = tf.layers.dense(
                t_output,
                n_input,
                activation=self.output_activation,
                kernel_initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1 / n_input),
                )
            )

            t_output = tf.nn.dropout(t_output, self.t_output_droupouts_kp[3])

            n_input = self.n_days_predict
            t_output = tf.layers.dense(
                t_output,
                n_input,
                activation=self.output_activation,
                kernel_initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1 / n_input),
                )
            )

        return t_output

    def decoder(self, t_encoder_state, last_value):
        """
        :param encoder_state: shape [batch_size, encoder_rnn_depth]
        :param days_features: shape [batch_size, time_to_predict, n_features-1]
        :param last_value: shape [batch_size], last avaliable value
        :return: decoder rnn output
        """

        # decoder
        stacked_decoder = []
        for _ in range(self.n_layers_rnn):
            t_dec_cell = tf.contrib.rnn.GRUCell(
                self.rnn_size_decoder,
                activation=self.decoder_activation
            )

            t_dec_cell = tf.contrib.rnn.DropoutWrapper(
                t_dec_cell, output_keep_prob=self.t_decoder_dropout_kp)

            stacked_decoder.append(t_dec_cell)

        dec_multicell = tf.contrib.rnn.MultiRNNCell(cells=stacked_decoder)

        def dense_layer(t_output):
            return tf.layers.dense(t_output, 1)

        def loop_cond(time, prev_output, prev_decoder_state, array_y):
            return time < self.n_days_predict

        def loop_body(time, prev_output, prev_decoder_state, array_y):
            t_output, t_state = dec_multicell(prev_output, prev_decoder_state)
            t_output = dense_layer(t_output)
            #t_output = self.output_layers(
            #    t_output,
            #    t_days_features[:, time, :],
            #    t_items_features,
            #)
            array_y = array_y.write(time, t_output)
            return time + 1, t_output, t_state, array_y

        loop_vars = [
            tf.constant(0, dtype=tf.int32),
            last_value,
            t_encoder_state,
            tf.TensorArray(dtype=tf.float32, size=self.n_days_predict),
        ]
        y = tf.while_loop(loop_cond, loop_body, loop_vars)[-1]
        y = y.stack()
        y = tf.squeeze(y, axis=-1)
        y = tf.transpose(y)
        y = tf.reshape(y, [-1, self.n_days_predict])
        return y

    def predict(self, t_X, t_y_day_attr, t_feat_store_nbr, t_feat_item_nbr):

        t_encoder_state = self.encoder(t_X)

        with tf.name_scope('decoder') as scope:
            t_decoder_predictions = self.decoder(
                t_encoder_state,
                t_X[:, -1, :1], # last day
            )

        concatenated_emb = self.embeddings(
            t_feat_store_nbr, t_feat_item_nbr)

        t_predictions = self.output_layers(
            t_decoder_predictions,  t_y_day_attr,
            concatenated_emb
        )

        return t_predictions

    def embeddings(self, t_feat_store_nbr, t_feat_item_nbr):

        emb_feat_item_nbr = tf.nn.embedding_lookup(
            self.emb_feat_item_nbr_vars, t_feat_item_nbr)

        emb_feat_store_nbr = tf.nn.embedding_lookup(
            self.emb_feat_store_nbr_vars, t_feat_store_nbr)

        concatenated_emb = tf.concat(
            [
                emb_feat_item_nbr,
                emb_feat_store_nbr
            ],
            axis=1
        )

        return concatenated_emb

    def get_loss(self, t_targets, t_predictions ):
        loss = tf.losses.mean_squared_error(
            labels=t_targets, predictions=t_predictions)
        tf.summary.scalar('loss', loss)

        return loss

    def get_train_op(self, loss):

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.contrib.layers.optimize_loss(
                loss, self.global_step,
                self.starter_learning_rate,
                optimizer='Adam',
                #learning_rate_decay_fn=lr_decay,
                clip_gradients=self.clip_gradients,
                #summaries=["gradients"]
            )
            return train_op


    def get_input_data(self, train_batch_gen):
        def gen():
            return train_batch_gen
        train_dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=(
                tf.float32, # X
                tf.int8,    # y_day_attr
                tf.float32, # y
                tf.int32,   # store_nbr
                tf.int8,    # n_city
                tf.int8,    # n_state
                tf.int8,    # n_type
                tf.int8,    # cluster
                tf.int32,   # item_nbr
                tf.int8,    # n_family
                tf.int16,   # class
            ),
            output_shapes=(
                tf.TensorShape([self.history, self.n_ts_attr]),
                tf.TensorShape([self.n_days_predict, self.n_ts_attr-1]),
                tf.TensorShape([self.n_days_predict]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
                tf.TensorShape([]),
            )
        )

        train_dataset = train_dataset.prefetch(self.batch_size*10)
        train_dataset = train_dataset.batch(self.batch_size)

        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.v_t_X,
                self.v_t_y_day_attr,
                self.v_t_y,
                self.v_t_feat_store_nbr,
                self.v_t_feat_n_city,
                self.v_t_feat_n_state,
                self.v_t_feat_n_type,
                self.v_t_feat_cluster,
                self.v_t_feat_item_nbr,
                self.v_t_feat_n_family,
                self.v_t_feat_class,
            )

        )
        validation_dataset = validation_dataset.prefetch(512*420)
        validation_dataset = validation_dataset.batch(512*80)

        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes
        )

        next_element = iterator.get_next()

        self.training_iterator = iterator.make_initializer(train_dataset)
        self.validation_iterator = iterator.make_initializer(validation_dataset)

        #self.training_iterator = dataset.make_initializable_iterator()

        #next_element = self.training_iterator.get_next()

        return next_element


    def build_graph(self, train_batch_gen):
        tf.reset_default_graph()
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():

            self.init_inputs()
            self.initiate_embeddings()

            (
                self.t_X, self.t_y_day_attr, self.t_y,
                self.t_feat_store_nbr, self.t_feat_n_city,
                self.t_feat_n_state, self.t_feat_n_type,
                self.t_feat_cluster, self.t_feat_item_nbr,
                self.t_feat_n_family, self.t_feat_class
            ) = self.get_input_data(train_batch_gen)

            #t_inputs_norm = self.convert_inputs(
            #    self.t_ts_inputs, self.t_items_features)

            self.t_predictions = self.predict(
                self.t_X, self.t_y_day_attr,
                self.t_feat_store_nbr,
                self.t_feat_item_nbr
            )
            #    self.t_days_features, self.t_items_features)

            self.loss = self.get_loss(self.t_y, self.t_predictions)

            self.train_op = self.get_train_op(self.loss)

            self.summary_merged = tf.summary.merge_all()

            self.sess = tf.Session()

            self.train_writer = tf.summary.FileWriter(
                self.path, self.sess.graph, flush_secs=20)

            self.sess.run(tf.global_variables_initializer())

            #self.run_queue(train_batch_gen)

        return True

            #return t_ts_inputs, t_items_features, t_days_features, t_targets,
            #    self.t_encoder_dropout_kp, self.t_decoder_dropout_kp,
            #    self.t_is_training, self.NWRMSLE
            #    self.train_graph, sess, train_op, summary_merged, loss

    def get_validation_batch(self, validation_set, size):
        X1, X2, X3, y = validation_set
        num_items = y.shape[0]
        permutated_indx = np.arange(num_items)
        for i in range(num_items//size+1):
            s = size * i
            e = size * (i+1)
            idxes = permutated_indx[s:e]
            yield X1[idxes], X2[idxes], X3[idxes], y[idxes]

    def train(self, validation_set, coef, sum_W,
            report_every=100, validate_every=1000, hd_exp=None):


        NWRMSLE = .6 # just for convenience

        with self.train_graph.as_default():
            self.sess.run(self.training_iterator)
            losses = []
            g_step = 0
            while True:

                try:
                    g_step += 1

                    _, summary, loss_value = self.sess.run(
                        [self.train_op, self.summary_merged,  self.loss],
                        feed_dict={
                            self.t_is_training: True,
                            self.NWRMSLE: NWRMSLE,
                            self.t_encoder_dropout_kp: self.encoder_dropout_kp,
                            self.t_decoder_dropout_kp: self.decoder_dropout_kp,
                            self.t_output_droupouts_kp: self.output_droupouts_kp,
                        }
                    )

                    self.train_writer.add_summary(summary, g_step)

                    losses.append(loss_value)
                except tf.errors.OutOfRangeError:
                    print('End of traininng set')
                    break

                except KeyboardInterrupt:
                    print("Ctrl+C")
                    break

                if g_step % report_every == 0:

                    losses = np.array(losses)
                    print("g_step: {} loss std/mean: {} {}".format(
                            g_step, losses.std(), losses.mean()))
                    if hd_exp is not None:
                        hd_exp.metric("Loss std", losses.std())
                        hd_exp.metric("Loss mean", losses.mean())
                    losses = []


                if g_step % validate_every == 0:
                    print("\tValidation", end='\r', flush=True)
                    #self.sess.run(self.validation_iterator)

                    predicted_val = None
                    real_unit_sales= None

                    (
                        v_X, y_day_attr, v_y,
                        v_store_nbr, v_n_city, v_n_state, n_type,
                        v_cluster, v_item_nbr, v_n_family, v_class
                    ) = validation_set

                    real_unit_sales = v_y

                    feed_dict = {

                        self.v_t_X: v_X,
                        self.v_t_y_day_attr: y_day_attr,
                        self.v_t_y: v_y,
                        self.v_t_feat_store_nbr: v_store_nbr,
                        self.v_t_feat_n_city: v_n_city,
                        self.v_t_feat_n_state: v_n_state,
                        self.v_t_feat_n_type: n_type,
                        self.v_t_feat_cluster: v_cluster,
                        self.v_t_feat_item_nbr: v_item_nbr,
                        self.v_t_feat_n_family: v_n_family,
                        self.v_t_feat_class: v_class,
                        self.t_encoder_dropout_kp: 1.,
                        self.t_decoder_dropout_kp: 1.,
                        self.t_output_droupouts_kp: [1., 1., 1., 1., 1.],
                        self.t_is_training: False,
                    }
                    self.sess.run(self.validation_iterator, feed_dict=feed_dict)

                    predictions = []

                    while True:
                        try:
                            predictions.append(
                                self.sess.run(
                                    self.t_predictions,
                                    feed_dict=feed_dict
                                )
                            )
                        except tf.errors.OutOfRangeError:
                            break

                        except KeyboardInterrupt:
                            print("Ctrl+C")
                            break

                    predicted_val = np.vstack(predictions)

                    NWRMSLE = (
                        np.sqrt(
                            np.sum(
                                np.square(
                                    coef*(predicted_val.reshape(-1,1) - real_unit_sales.reshape(-1,1))
                                )
                            )/(sum_W*self.n_days_predict)
                        )
                    )

                    print("\tValidation NWRMSLE  :", NWRMSLE)
                    if hd_exp is not None:
                        hd_exp.metric("Validation NWRMSLE", NWRMSLE)

                    if self.n_days_predict > 5:
                        NWRMSLE_5 = (
                            np.sqrt(
                                np.sum(
                                    np.square(
                                        coef*(predicted_val[:,:5].reshape(-1,1) - real_unit_sales[:,:5].reshape(-1,1))
                                    )
                                )/((sum_W*5.))
                            )
                        )
                        print("\tValidation NWRMSLE_5:", NWRMSLE_5)

                        if hd_exp is not None:
                            hd_exp.metric("Validation NWRMSLE_5", NWRMSLE_5)

                    self.sess.run(self.training_iterator)

            if hd_exp is not None:
                hd_exp.end()

    #def __del__(self):
    #    pass
    #    #self.hd_exp.end()
