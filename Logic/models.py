import tensorflow as tf
import keras
import time
import warnings
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.topology import Layer
#from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import concatenate
from keras.layers import CuDNNGRU
from keras.layers import Dense
from keras.layers import Dropout
#from keras.layers import GRU
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Multiply
from keras.layers import Reshape
from keras.layers import TimeDistributed

from keras.models import load_model


# Feature MLP encoder
def input_encoder(x, hyperpars):
  if hyperpars['encoding_input_dropout'] > 0:
    x = Dropout(hyperpars['encoding_input_dropout'], name='input_dropout')(x)
  
  num_enc_layers = len(hyperpars['encoding_layers'])
  for (i, layer_size) in enumerate(hyperpars['encoding_layers']):
    if i == num_enc_layers-1 and not hyperpars.get(
        'relu_last_encoding_layer', False):
      activation = 'linear'
    else:
      activation = 'relu'
    x = Dense(layer_size, activation=activation, name='enc_layer' + str(i))(x)
    if hyperpars.get('encoding_dropout', 0) > 0:
      x = Dropout(hyperpars['encoding_dropout'])(x)

  return x


# Autoregressive part of the baseline recurrent sub-chunk model
def network_autoregressive(x, hyperpars):
  x = CuDNNGRU(units=hyperpars['num_recurrent_cells'], return_sequences=True,
               name='ar_emb')(x)
  
  for (i, layer_size) in enumerate(hyperpars['prediction_layers']):
    x = TimeDistributed(Dense(layer_size, activation='relu',
                              name='pred_layer' + str(i)))(x)
  
  x = TimeDistributed(Dense(1))(x)
  x = Lambda(lambda x: K.squeeze(x, axis=2))(x)

  return x


# Baseline recurrent sub-chunk model
def initial_gru(sub_steps, sub_dims, hyperpars):
  # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
  K.set_learning_phase(1)
  
  # Define encoder model - encoding is fit to the autoregressive encoder
  encoder_input = Input((sub_dims,))
  encoder_output = input_encoder(encoder_input, hyperpars)
  encoder_model = keras.models.Model(encoder_input, encoder_output,
                                     name='encoder')
  
  # Define the rest of the model - autoregressive model predicting the target
  inputs = Input((sub_steps, sub_dims))
  x_encoded = TimeDistributed(encoder_model)(inputs)
  outputs = network_autoregressive(x_encoded, hyperpars)
  
  return (inputs, outputs)


###############################################################################


# Autoregressive part of the sequential recurrent sub-chunk model
def network_autoregressive_seq_gru(x, hyperpars):
#  x = CuDNNGRU(units=hyperpars['num_recurrent_cells'], return_sequences=True,
#               name='ar_emb')(x)
  for (i, layer_size) in enumerate(hyperpars['recurrent_cells']):
    x = Bidirectional(CuDNNGRU(layer_size, return_sequences=True),
                      merge_mode='sum')(x)
    x = Dropout(hyperpars['gru_dropout'])(x)
  
  for (i, layer_size) in enumerate(hyperpars['prediction_layers']):
    x = TimeDistributed(Dense(layer_size, activation='relu',
                              name='pred_layer' + str(i)))(x)
  
  x = TimeDistributed(Dense(1))(x)
  x = Lambda(lambda x: K.squeeze(x, axis=2))(x)

  return x


# Sequential recurrent sub-chunk model - part of Final submission
def sequential_gru(hyperpars):
  # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
  K.set_learning_phase(1)
  
  num_features = 0
  if hyperpars['include_freq_features']:
    num_features += hyperpars['input_dimension_freq']
  if hyperpars['include_cpc_features']:
    num_features += hyperpars['input_dimension_cpc']
  sub_steps = hyperpars['chunk_blocks']
  
  # Define FF encoder model
  encoder_input = Input((num_features,))
  encoder_output = input_encoder(encoder_input, hyperpars)
  encoder_model = keras.models.Model(encoder_input, encoder_output,
                                     name='encoder')
  
  # Define the rest of the model - autoregressive model predicting the target
  inputs = Input((sub_steps, num_features))
  x_encoded = TimeDistributed(encoder_model)(inputs)
  outputs = network_autoregressive_seq_gru(x_encoded, hyperpars)
  
  if hyperpars['clip_preds_zero_one']:
    outputs = Lambda(lambda x: K.clip(x, 0, 1))(outputs)
  
  return (inputs, outputs)


###############################################################################


# Recurrent core of the gap prediction model
def stacked_gru(x, hyperpars):
  for (i, layer_size) in enumerate(hyperpars['recurrent_cells']):
    x = Bidirectional(CuDNNGRU(layer_size, return_sequences=True),
                      merge_mode='sum')(x)
    x = Dropout(hyperpars['gru_dropout'])(x)
  
  return x


# Prediction head for the gap prediction model
def prediction_mlp_head(x, hyperpars):
  for (i, layer_size) in enumerate(hyperpars['prediction_layers']):
    x = TimeDistributed(Dense(layer_size, activation='relu',
                              name='pred_layer' + str(i)))(x)
    x = Dropout(hyperpars['prediction_dropout'])(x)
  
  x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
  x = Lambda(lambda x: K.squeeze(x, axis=2))(x)

  return x


# Baseline gap prediction model
def initial_gap(hyperpars):
  # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
  K.set_learning_phase(1)
  
  # Encode the input with a stacked bidirectional RNN
  inputs = Input((hyperpars['block_steps'], 1))
  x_encoded = stacked_gru(inputs, hyperpars)
  
  # Predict the gap probability
  outputs = prediction_mlp_head(x_encoded, hyperpars)
  
  return (inputs, outputs)


###############################################################################
  

# Attention layer - taken from https://www.kaggle.com/takuok/bidirectional-lstm-and-attention-lb-0-043/notebook
class Attention(Layer):
  def __init__(self, step_dim=-1,
               W_regularizer=None, b_regularizer=None,
               W_constraint=None, b_constraint=None,
               bias=True, **kwargs):
    self.supports_masking = True
    self.init = initializers.get('glorot_uniform')

    self.W_regularizer = regularizers.get(W_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)

    self.W_constraint = constraints.get(W_constraint)
    self.b_constraint = constraints.get(b_constraint)

    self.bias = bias
    self.step_dim = step_dim
    self.features_dim = 0
    super(Attention, self).__init__(**kwargs)

  def build(self, input_shape):
    assert len(input_shape) == 3

    self.W = self.add_weight((input_shape[-1],),
                             initializer=self.init,
                             name='{}_W'.format(self.name),
                             regularizer=self.W_regularizer,
                             constraint=self.W_constraint)
    self.features_dim = input_shape[-1]

    if self.bias:
      self.b = self.add_weight((input_shape[1],),
                               initializer='zero',
                               name='{}_b'.format(self.name),
                               regularizer=self.b_regularizer,
                               constraint=self.b_constraint)
    else:
      self.b = None

    self.built = True

  def compute_mask(self, input, input_mask=None):
    return None

  def call(self, x, mask=None):
    features_dim = self.features_dim
    step_dim = self.step_dim

    eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                    K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
    if self.bias:
      eij += self.b
    eij = K.tanh(eij)
    a = K.exp(eij)

    if mask is not None:
      a *= K.cast(mask, K.floatx())

    a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
    a = K.expand_dims(a)
    weighted_input = x * a
    return K.sum(weighted_input, axis=1)

  def compute_output_shape(self, input_shape):
    return input_shape[0], self.features_dim
  
  def get_config(self):
    config = {'step_dim': self.step_dim}
    base_config = super(Attention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  

# Gradient reversal logic - taken from https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
def reverse_gradient(X, hp_lambda):
  '''Flips the sign of the incoming gradient during training.'''
  grad_name = 'GradientReversal%d' % int(round(time.time() * 1000))
  time.sleep(0.001)

  @tf.RegisterGradient(grad_name)
  def _flip_gradients(op, grad):
    return [tf.negative(grad) * hp_lambda]

  g = K.get_session().graph
  with g.gradient_override_map({'Identity': grad_name}):
    y = tf.identity(X)

  return y


class GradientReversal(Layer):
  '''Flip the sign of gradient during training.'''
  def __init__(self, hp_lambda, **kwargs):
    super(GradientReversal, self).__init__(**kwargs)
    self.supports_masking = False
    self.hp_lambda = hp_lambda

  def build(self, input_shape):
    self.trainable_weights = []

  def call(self, x, mask=None):
    return reverse_gradient(x, self.hp_lambda)

#  def get_output_shape_for(self, input_shape):
#    return input_shape
  
  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'hp_lambda': self.hp_lambda}
    base_config = super(GradientReversal, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
      
      
# CPC encoder: stacked bidirectional RNN followed by MLP and aggregation.
# The aggregation is either an attention layer or an averaging layer.
def cpc_encoder(x, hyperpars):
  # Optionally, drop the gap predictions from the inputs
  if hyperpars['drop_gap_prediction_from_inputs']:
    x = Lambda(lambda x: x[:, :, :1])(x)
  
  for (i, layer_size) in enumerate(hyperpars['encoder_recurrent_cells']):
    x = Bidirectional(CuDNNGRU(layer_size, return_sequences=True),
                      merge_mode='sum')(x)
    x = Dropout(hyperpars['gru_dropout'])(x)
    
  mlp_layers = hyperpars['encoder_embedding_layers']
  for (i, layer_size) in enumerate(mlp_layers):
    act = 'linear' if i == (len(mlp_layers)-1) else 'relu'
    x = Dense(layer_size, activation=act, name= 'enc_mlp_layer' + str(i))(x)
    x = Dropout(hyperpars['encoder_mlp_dropout'])(x)
  
  if hyperpars['use_attention_encoder']:
    x = Attention(hyperpars['block_steps'])(x)
  else:
    x = Lambda(lambda x: K.mean(x, axis=1))(x)
  
  return x


# CPC MAE prediction head - TimeDistributed MLP
def cpc_mae_head(x, hyperpars):
  for (i, layer_size) in enumerate(hyperpars['mae_prediction_layers']):
    x = TimeDistributed(Dense(
        layer_size, activation='relu', name='mae_pred_layer' + str(i)))(x)
    x = Dropout(hyperpars['prediction_dropout'])(x)
  
  x = TimeDistributed(Dense(1))(x)
  x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
  x = Lambda(lambda x: x, name='train_mae_prediction')(x)

  return x


# Domain prediction MLP - trained with reversed gradients to make the learned
# embedding agnostic to the domain
def domain_prediction(x, hyperpars):
#  x = GradientReversal(hp_lambda=hyperpars['grad_rev_lambda'])(x)
  x = Lambda(lambda x, _lambda=hyperpars['grad_rev_lambda']: K.stop_gradient(
      x*K.cast(1 + _lambda, 'float32')) - x*(
      K.cast(_lambda, 'float32')))(x)
  
  for (i, layer_size) in enumerate(hyperpars['domain_prediction_layers']):
    x = Dense(
        layer_size, activation='relu', name='domain_pred_layer' + str(i))(x)
    x = Dropout(hyperpars['prediction_dropout'])(x)
  
  x = Dense(1, activation='sigmoid')(x)
  x = Lambda(lambda x: K.squeeze(x, axis=1))(x)

  return x


# Domain prediction MLP - trained with reversed gradients to make the learned
# embedding agnostic to the domain. TODO: don't duplicate def domain_prediction
def additional_domain_prediction(x, hyperpars):
#  x = GradientReversal(hp_lambda=hyperpars['grad_rev_lambda'])(x)
#  x = Lambda(lambda x: K.stop_gradient(x))(x)
  x = Lambda(lambda x, _lambda=hyperpars['grad_rev_lambda']: K.stop_gradient(
      x*K.cast(1 + _lambda, 'float32')) - x*(
      K.cast(_lambda, 'float32')))(x)
  
  for (i, layer_size) in enumerate(
      hyperpars['additional_domain_prediction_layers']):
    x = Dense(layer_size, activation='relu',
        name='additional_domain_pred_layer' + str(i))(x)
    x = Dropout(hyperpars['prediction_dropout'])(x)
  
  x = Dense(1, activation='sigmoid')(x)
  x = Lambda(lambda x: K.squeeze(x, axis=1))(x)

  return x


# Helper function to extract and stack the relevant first and candidate
# encodings.
def subset_stack_encodings(encodings, comp_ids):
  first_encodings = Lambda(lambda x: tf.gather(
      x[0], K.cast(x[1][:, :, 0], 'int32'), axis=1)[:, 0])(
      [encodings, comp_ids])
  candidate_encodings = []
  num_candidates = int(comp_ids.shape[-1])-1
  for i in range(num_candidates):
    candidate_encodings.append(Lambda(lambda x, i=i: tf.gather(
        x[0], K.cast(x[1][:, :, i+1], 'int32'), axis=1)[:, 0])(
        [encodings, comp_ids]))
  candidate_encodings = Lambda(lambda x: K.stack(x, axis=2))(
      candidate_encodings)
  first_encodings = Lambda(lambda x, num_candidates=num_candidates: K.tile(
      K.expand_dims(x, 2), [1, 1, num_candidates, 1]))(first_encodings)
  
  return first_encodings, candidate_encodings, num_candidates


# Compute the dot product between all first and subsequent candidate encodings.
def dot_product_pairs(encodings, comp_ids, name):
  first_encodings, candidate_encodings, num_candidates = (
      subset_stack_encodings(encodings, comp_ids))
      
  x = Multiply()([candidate_encodings, first_encodings])
  x = Lambda(lambda x, num_candidates=num_candidates: K.softmax(
      num_candidates*K.sum(x, axis=-1)))(x)
  x = Lambda(lambda x: x, name=name)(x)
  
  return x


# Apply the model to the stacked encodings.
def pair_comparison_from_ids(model, encodings, comp_ids, name):
  first_encodings, candidate_encodings, num_candidates = (
      subset_stack_encodings(encodings, comp_ids))
  x = Lambda(lambda x: K.concatenate(x, axis=-1))(
      [first_encodings, candidate_encodings])
  batch_dim = x.shape.as_list()[1]
  encoding_size = x.shape.as_list()[3]
  x = Reshape((batch_dim*num_candidates, encoding_size))(x)
  x = TimeDistributed(model)(x)
  x = Reshape((batch_dim, num_candidates))(x)
  x = Lambda(lambda x: x, name=name)(x)
  
  return x


# Custom MLP that uses the layers of a layer prefix and has no final layer
# activation.
def custom_mlp(x, prefix, hyperpars, final_sigmoid_layer=False):
  mlp_layers = hyperpars[prefix + '_prediction_layers']
  for (i, layer_size) in enumerate(mlp_layers):
    final_layer = not final_sigmoid_layer and (i == (len(mlp_layers) - 1))
    act = 'linear' if final_layer else 'relu'
    x = Dense(
        layer_size, activation=act, name=prefix + '_pred_layer' + str(i))(x)
    if not final_layer:
      x = Dropout(hyperpars['prediction_dropout'])(x)
      
  if final_sigmoid_layer:
    x = Dense(1, activation='sigmoid')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
  else:
    x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)
  
  return x

# Baseline CPC model. The model outputs four predictions:
# 1) MAE prediction (train only)
# 2) Domain (train+test) prediction - trained adversarially
# 3) Consecutive sub-chunk prediction (train+test)
# 4) Same earthquake prediction (train only)
def initial_cpc(hyperpars):
  # Define the dimensions of the different cpc inputs
  train_inputs = Input((hyperpars['batch_size'], hyperpars['block_steps'], 2),
                       name='train_inputs')
  test_inputs = Input((hyperpars['batch_size'], hyperpars['block_steps'], 2),
                      name='test_inputs')
  num_embeddings = 2 + hyperpars['num_decoys']
  train_sub_comp_ids = Input((hyperpars['batch_size']/2, num_embeddings),
                             name='train_sub_comp', dtype='int32')
  train_chunk_comp_ids = Input((hyperpars['batch_size']/2, num_embeddings),
                               name='train_chunk_comp', dtype='int32')
  train_eq_comp_ids = Input((hyperpars['batch_size'], num_embeddings),
                            name='train_eq_comp', dtype='int32')
  test_sub_comp_ids = Input((hyperpars['batch_size']/2, num_embeddings),
                             name='test_sub_comp', dtype='int32')
  test_chunk_comp_ids = Input((hyperpars['batch_size']/2, num_embeddings),
                              name='test_chunk_comp', dtype='int32')
  all_inputs = [train_inputs, test_inputs, train_sub_comp_ids,
                train_chunk_comp_ids, train_eq_comp_ids, test_sub_comp_ids,
                test_chunk_comp_ids]
  
  # Define the models
  # Model 1) Encoder model
  encoder_input = Input((hyperpars['block_steps'], 2))
  encoder_output = cpc_encoder(encoder_input, hyperpars)
  encoder_model = keras.models.Model(encoder_input, encoder_output,
                                     name='encoder')
  
  # Model 2) Domain prediction model
  domain_pred_input = Input((encoder_output.shape.as_list()[-1],))
  double_domain_pred_input = Input((2*encoder_output.shape.as_list()[-1],))
  domain_pred_output = domain_prediction(domain_pred_input, hyperpars)
  domain_pred_model = keras.models.Model(domain_pred_input, domain_pred_output,
                                         name='domain_pred_model')
  add_domain_pred_output = additional_domain_prediction(
      domain_pred_input, hyperpars)
  add_domain_pred_model = keras.models.Model(
      domain_pred_input, add_domain_pred_output,
      name='add_domain_pred_model')
  
  # Models 3) Subchunk, chunk and earthquake encoding models
  subchunk_pred_output = custom_mlp(double_domain_pred_input, 'subchunk',
                                    hyperpars, final_sigmoid_layer=True)
  subchunk_pred_model = keras.models.Model(
      double_domain_pred_input, subchunk_pred_output,
      name='subchunk_pred_model')
  chunk_pred_output = custom_mlp(domain_pred_input, 'chunk', hyperpars)
  chunk_pred_model = keras.models.Model(
      domain_pred_input, chunk_pred_output, name='chunk_pred_model')
  eq_pred_output = custom_mlp(domain_pred_input, 'eq', hyperpars)
  eq_pred_model = keras.models.Model(
      domain_pred_input, eq_pred_output, name='eq_pred_model')
  
  # Encode the input with a shared model
  train_encoded = TimeDistributed(encoder_model)(train_inputs)
  test_encoded = TimeDistributed(encoder_model)(test_inputs)
  
  # Encode the CPC embeddings from the shared embedding
  train_chunk_encoded = TimeDistributed(chunk_pred_model)(train_encoded)
  train_eq_encoded = TimeDistributed(eq_pred_model)(train_encoded)
  test_chunk_encoded = TimeDistributed(chunk_pred_model)(test_encoded)
  
  # Generate the outputs
  train_mae_preds = cpc_mae_head(train_encoded, hyperpars)
  train_dom_preds = TimeDistributed(domain_pred_model)(train_encoded)
  test_dom_preds = TimeDistributed(domain_pred_model)(test_encoded)
  dom_preds = concatenate([train_dom_preds, test_dom_preds])
  dom_preds = Lambda(lambda x: x, name='domain_prediction')(dom_preds)
  
  add_train_dom_preds = TimeDistributed(add_domain_pred_model)(train_encoded)
  add_test_dom_preds = TimeDistributed(add_domain_pred_model)(test_encoded)
  add_dom_preds = concatenate([add_train_dom_preds, add_test_dom_preds])
  add_dom_preds = Lambda(lambda x: x, name='additional_domain_prediction')(
      add_dom_preds)
  
  train_subchunk_preds = pair_comparison_from_ids(
      subchunk_pred_model, train_encoded, train_sub_comp_ids,
      'train_subchunk_predictions')
  train_chunk_preds = dot_product_pairs(
      train_chunk_encoded, train_chunk_comp_ids, 'train_chunk_predictions')
  train_eq_preds = dot_product_pairs(
      train_eq_encoded, train_eq_comp_ids, 'train_eq_predictions')
  test_subchunk_preds = pair_comparison_from_ids(
      subchunk_pred_model, test_encoded, test_sub_comp_ids,
      'test_subchunk_predictions')
  test_chunk_preds = dot_product_pairs(
      test_chunk_encoded, test_chunk_comp_ids, 'test_chunk_predictions')
  
  all_outputs = [train_mae_preds, dom_preds, add_dom_preds,
                 train_subchunk_preds, train_chunk_preds, train_eq_preds,
                 test_subchunk_preds, test_chunk_preds]
  
  return (all_inputs, all_outputs, encoder_model)


###############################################################################
  

# CPC main encoder: MLP - stacked bidirectional RNN and aggregation
# The aggregation is either an attention layer or an averaging layer.
def cpc_main_encoder(x, hyperpars):
  # Input dropout
  x = Dropout(hyperpars['input_dropout'])(x)
  
  # Input MLP
  mlp_layers = hyperpars['encoder_input_embedding_layers']
  for (i, layer_size) in enumerate(mlp_layers):
    x = Dense(layer_size, activation='relu',
              name= 'main_enc_input_mlp_layer' + str(i))(x)
    x = Dropout(hyperpars['encoder_mlp_dropout'])(x)
    
  # Recurrent encoding
  for (i, layer_size) in enumerate(hyperpars['encoder_recurrent_cells']):
    x = Bidirectional(CuDNNGRU(layer_size, return_sequences=True),
                      merge_mode='sum')(x)
    x = Dropout(hyperpars['gru_dropout'])(x)
    
  # Output MLP
  mlp_layers = hyperpars['encoder_output_embedding_layers']
  for (i, layer_size) in enumerate(mlp_layers):
    act = 'linear' if i == (len(mlp_layers)-1) else 'relu'
    x = Dense(layer_size, activation=act,
              name='main_enc_output_mlp_layer' + str(i))(x)
    x = Dropout(hyperpars['encoder_mlp_dropout'])(x)
  
  if hyperpars['use_attention_encoder']:
    x = Attention(hyperpars['chunk_blocks'], name='main_attention')(x)
  else:
    x = Lambda(lambda x: K.mean(x, axis=1))(x)
  
  return x


# Baseline CPC model. The model outputs three predictions:
# 1) MAE prediction (train only)
# 2) Domain (train+test) prediction - trained adversarially
# 3) Consecutive chunk prediction (train only)
def main_cpc(hyperpars, encoder_path):
  # Read the encoder and freeze the weights
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    pretrained_encoder = load_model(encoder_path,
                                    custom_objects={'Attention': Attention})
  pretrained_encoder._make_predict_function()
  pretrained_encoder.trainable = False
  for layer in pretrained_encoder.layers:
    layer.trainable = False
  
  # Define the dimensions of the different cpc inputs
  train_inputs = Input((hyperpars['batch_size'], hyperpars['chunk_blocks'],
                        hyperpars['input_dimension']), name='train_inputs')
  test_inputs = Input((hyperpars['batch_size'], hyperpars['chunk_blocks'],
                       hyperpars['input_dimension']), name='test_inputs')
  num_embeddings = 2 + hyperpars['num_decoys']
  train_chunk_comp_ids = Input((hyperpars['batch_size']/2, num_embeddings),
                               name='train_chunk_comp', dtype='int32')
  all_inputs = [train_inputs, test_inputs, train_chunk_comp_ids]
  
  # Define the models
  # Model 1) Encoder model
  encoder_input = Input((hyperpars['chunk_blocks'],
                         hyperpars['input_dimension']))
  encoder_output = cpc_main_encoder(encoder_input, hyperpars)
  encoder_model = keras.models.Model(encoder_input, encoder_output,
                                     name='main_encoder')
  
  # Model 2) Domain prediction model
  domain_pred_input = Input((encoder_output.shape.as_list()[-1],))
  double_domain_pred_input = Input((2*encoder_output.shape.as_list()[-1],))
  domain_pred_output = domain_prediction(domain_pred_input, hyperpars)
  domain_pred_model = keras.models.Model(domain_pred_input, domain_pred_output,
                                         name='domain_pred_model')
  add_domain_pred_output = additional_domain_prediction(
      domain_pred_input, hyperpars)
  add_domain_pred_model = keras.models.Model(
      domain_pred_input, add_domain_pred_output,
      name='add_domain_pred_model')
  
  # Models 3) Chunk encoding model
  chunk_pred_output = custom_mlp(double_domain_pred_input, 'chunk',
                                 hyperpars, final_sigmoid_layer=True)
  chunk_pred_model = keras.models.Model(
      double_domain_pred_input, chunk_pred_output, name='chunk_pred_model')
  
  # Encode the input with a shared model
  train_encoded = TimeDistributed(encoder_model)(train_inputs)
  test_encoded = TimeDistributed(encoder_model)(test_inputs)
  
  # Generate the outputs
  train_mae_preds = cpc_mae_head(train_encoded, hyperpars)
  train_dom_preds = TimeDistributed(domain_pred_model)(train_encoded)
  test_dom_preds = TimeDistributed(domain_pred_model)(test_encoded)
  dom_preds = concatenate([train_dom_preds, test_dom_preds])
  dom_preds = Lambda(lambda x: x, name='domain_prediction')(dom_preds)
  
  add_train_dom_preds = TimeDistributed(add_domain_pred_model)(train_encoded)
  add_test_dom_preds = TimeDistributed(add_domain_pred_model)(test_encoded)
  add_dom_preds = concatenate([add_train_dom_preds, add_test_dom_preds])
  add_dom_preds = Lambda(lambda x: x, name='additional_domain_prediction')(
      add_dom_preds)
  
  train_chunk_preds = pair_comparison_from_ids(
      chunk_pred_model, train_encoded, train_chunk_comp_ids,
      'train_chunk_predictions')
  
  all_outputs = [train_mae_preds, dom_preds, add_dom_preds, train_chunk_preds]
#  import pdb; pdb.set_trace()
  
  return (all_inputs, all_outputs, pretrained_encoder)