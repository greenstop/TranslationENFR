
import sys;
print(sys.executable, sys.version)

import helper

english_sentences = helper.load_data('data/small_vocab_en')
french_sentences = helper.load_data('data/small_vocab_fr')

print('Dataset Loaded')

for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, french_sentences[sample_i]))

import collections

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

import project_tests as tests
from keras.preprocessing.text import Tokenizer

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    tokenizer = Tokenizer();
    tokenizer.fit_on_texts(x);
    texts = tokenizer.texts_to_sequences(x);
    return texts, tokenizer
tests.test_tokenize(tokenize)

text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))

import numpy as np
from keras.preprocessing.sequence import pad_sequences

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    return pad_sequences(x,maxlen=length,padding='post',truncating='post')
tests.test_pad(pad)

test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =    preprocess(english_sentences, french_sentences)

print('Data Preprocessed')

print(preproc_english_sentences.shape)
print(preproc_french_sentences.shape)

text_tokenizer.word_index.items()

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')

print(preproc_english_sentences.shape);
print(pad(preproc_english_sentences, preproc_french_sentences.shape[1]).shape);
print(preproc_french_sentences.shape)
print(preproc_french_sentences.shape[-2])
print(preproc_french_sentences.shape[1])

tmp_x1 = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
print(tmp_x1.shape)
tmp_x1 = tmp_x1.reshape((-1, preproc_french_sentences.shape[-2], 1))
print(tmp_x1.shape[1:])
print(len(english_tokenizer.word_index))

print(tmp_x1[:1])

from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    learning_rate=0.001
    # Build the layers
    inputs_seq = Input(shape=input_shape[1:]);
    x = GRU(64,return_sequences=True,dropout=0.1,recurrent_dropout=0.1)(inputs_seq);
    x = TimeDistributed(Dense(french_vocab_size))(x);
    outputs = Activation('softmax')(x);
    
    model = Model(inputs=inputs_seq,outputs=outputs);
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
tests.test_simple_model(simple_model)

tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

simple_rnn_model = simple_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index),
    len(french_tokenizer.word_index))
simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=40, validation_split=0.2)

print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

print("english in shape",preproc_english_sentences.shape);
print("french in shape",preproc_french_sentences.shape);
y = preproc_french_sentences;
X = pad(preproc_english_sentences,y.shape[1]);
print("X shape", X.shape);

from keras.layers.embeddings import Embedding

def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    inputs = Input(shape=input_shape[1:]); 
    print("inputs",inputs.shape);
    x = Embedding(english_vocab_size,128,input_length=input_shape[1])(inputs);
    print("Embedding",x.shape);
    x = GRU(256,return_sequences=True,dropout=0.2,recurrent_dropout=0.2)(x);
    print("GRU",x.shape);
    x = TimeDistributed(Dense(french_vocab_size))(x);
    print("TimeD(D)",x.shape);
    outputs = Activation('softmax')(x);
    print("outputs",outputs.shape)
    
    model = Model(inputs=inputs,outputs=outputs);
    print("model",model.output_shape)
    model.compile(loss=sparse_categorical_crossentropy,optimizer='RMSprop',metrics=['accuracy']);
    return model;
tests.test_embed_model(embed_model)

model = embed_model(X.shape,y.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index));
model.fit(X,y,batch_size=(1024+1024//2),epochs=40,verbose=1,validation_split=0.2);

print("Prediction\n",logits_to_text(model.predict(X[:1])[0],french_tokenizer))

Y = preproc_french_sentences;
print("Y shape", Y.shape);
X = pad(preproc_english_sentences,Y.shape[1]).reshape(-1,Y.shape[1],1);
print("X shape", X.shape);

from keras.layers import Bidirectional

def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Implement
    inputs = Input(shape=input_shape[1:]);
    print("inputs",inputs.shape);
    x = Bidirectional(GRU(128,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))(inputs);
    print("forward",x.shape);
    x = Bidirectional(GRU(128,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))(x);
    print("backward",x.shape);
    x = TimeDistributed(Dense(french_vocab_size))(x);
    print("TimeD",x.shape);
    outputs = Activation('softmax')(x);
    
    model = Model(inputs=inputs,outputs=outputs);
    model.compile(loss=sparse_categorical_crossentropy,optimizer='rmsprop',metrics=['accuracy']);
    return model;
tests.test_bd_model(bd_model)

model = bd_model(X.shape,Y.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index));
model.summary();
model.fit(X,Y,batch_size=1024,epochs=20,validation_split=0.2);

P = logits_to_text(model.predict(X[:1])[0],french_tokenizer);
print(P)

Y = preproc_french_sentences;
X = preproc_english_sentences.reshape(-1,X.shape[1],1);
print(X.shape);

from keras.layers import RepeatVector

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    inputs = Input(shape=input_shape[1:]);
    print("inputs",inputs.shape);
    x = GRU(128,dropout=0.2,recurrent_dropout=0.2)(inputs);
    print("GRU",x.shape);
    x = RepeatVector(output_sequence_length)(x);
    print("Repeat",x.shape);
    x = GRU(128,return_sequences=True,dropout=0.2,recurrent_dropout=0.2)(x);
    print("GRU",x.shape);
    x = TimeDistributed(Dense(french_vocab_size))(x);
    print("TimeD",x.shape);
    outputs = Activation('softmax')(x);
    print("outputs",outputs.shape);
    
    model = Model(inputs=inputs,outputs=outputs);
    model.compile(loss=sparse_categorical_crossentropy,optimizer='rmsprop',metrics=['accuracy']);
    
    return model;
tests.test_encdec_model(encdec_model)

model = encdec_model(X.shape,Y.shape[1],len(english_tokenizer.word_index),len(french_tokenizer.word_index));
model.summary();
model.fit(X,Y,batch_size=1024,epochs=20,validation_split=0.2);

P = logits_to_text(model.predict(X[:1])[0],french_tokenizer);
print("Prediction\n",P);

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Implement
    inputs = Input(shape=input_shape[1:]);
    print("inputs",inputs.shape);
    
    #Embeddding
    x = Embedding(english_vocab_size,256,input_length=input_shape[1])(inputs);
    print("Embedding",x.shape);
    
    #Bidirectional Encoder
    x = Bidirectional(GRU(128,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))(x);
    print("Forward",x.shape);
    x = Bidirectional(GRU(128,return_sequences=False,dropout=0.2,recurrent_dropout=0.2))(x);
    print("Backward",x.shape);
    
    #Decoder
    x = RepeatVector(output_sequence_length)(x);
    print("Repeat",x.shape);
    x = GRU(256,return_sequences=True,dropout=0.2,recurrent_dropout=0.2)(x);
    print("GRU",x.shape);
    
    #Output
    x = TimeDistributed(Dense(french_vocab_size))(x);
    print("TimeD",x.shape);
    outputs = Activation('softmax')(x);
    print("outputs",x.shape);
    
    model = Model(inputs=inputs,outputs=outputs);
    model.compile(loss=sparse_categorical_crossentropy,optimizer='rmsprop',metrics=['accuracy']);
    return model;
tests.test_model_final(model_final)

print('Final Model Loaded')

import numpy as np
from keras.preprocessing.sequence import pad_sequences

def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    # Train neural network using model_final
    model = model_final(x.shape,y.shape[1],len(x_tk.word_index),len(y_tk.word_index));
    model.summary();
    model.fit(x,y,batch_size=1024,epochs=40,validation_split=0.2);

    
    ## DON'T EDIT ANYTHING BELOW THIS LINE
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.argmax(x)] for x in y[0]]))

final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)


