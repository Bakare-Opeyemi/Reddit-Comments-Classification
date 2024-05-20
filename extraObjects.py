import tensorflow as tf
import keras
import statistics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer 
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Embedding, Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from keras.layers import LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


@keras.saving.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, heads, neurons):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(neurons, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.5)
        self.dropout2 = layers.Dropout(0.5)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    



#Function to preprocess texts for single prediction
def preprocessComment(comment):
  MAX_SEQUENCE_LENGTH = 500
  tokenizer = RegexpTokenizer(r'[A-Za-z]+')
  tokenized_text = tokenizer.tokenize(comment)

  stemmer = SnowballStemmer("english")
  stemmed_words = [stemmer.stem(word) for word in tokenized_text]

  stemmed_sentence = ' '.join(stemmed_words)

  tok = Tokenizer(oov_token='<UNK>')
  # fit the tokenizer on the documents
  tok.fit_on_texts([stemmed_sentence])
  tok.word_index['<PAD>'] = 0
  max([(k, v) for k, v in tok.word_index.items()], key = lambda x:x[1]), min([(k, v) for k, v in tok.word_index.items()], key = lambda x:x[1]), tok.word_index['<UNK>']
  text_sequence = tok.texts_to_sequences([stemmed_sentence])
  print(len(text_sequence[0]))


  input_data = sequence.pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH)
  return input_data


def ensemblePrediction(predictionList):
  predictionList.sort()
  indecisiveVotes = [[0,1,2]]
  if predictionList in indecisiveVotes:
    ensembleDecision = 1

  else:
    ensembleDecision = statistics.mode(predictionList)

  return ensembleDecision


#Function to preprocess texts for batch prediction
def preprocessBatch(comments):
  MAX_SEQUENCE_LENGTH = 500
  tokenizer = RegexpTokenizer(r'[A-Za-z]+')
  stemmed_sentences = []
  for comment in comments:
    tokenized_text = tokenizer.tokenize(comment)

    stemmer = SnowballStemmer("english")
    stemmed_words = [stemmer.stem(word) for word in tokenized_text]

    stemmed_sentence = ' '.join(stemmed_words)
    stemmed_sentences.append(stemmed_sentence)

  tok = Tokenizer(oov_token='<UNK>')
  # fit the tokenizer on the documents
  tok.fit_on_texts(stemmed_sentences)
  tok.word_index['<PAD>'] = 0
  max([(k, v) for k, v in tok.word_index.items()], key = lambda x:x[1]), min([(k, v) for k, v in tok.word_index.items()], key = lambda x:x[1]), tok.word_index['<UNK>']
  
  text_sequence = tok.texts_to_sequences(stemmed_sentences)
  #print(len(text_sequence[0]))


  input_data = sequence.pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH)
  return input_data