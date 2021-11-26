# Import module
import tensorflow as tf
import pickle

# # Load vocab
# file_to_read = open("vocab.pickle", "rb")
# vocab = pickle.load(file_to_read)
# file_to_read.close()

# # Load model
# model = tf.keras.models.load_model('my_sa_model')

# def predict_sentiment(net, vocab, sentence):
#     sentence = tf.constant(vocab[sentence.split()])
#     label = tf.argmax(net(tf.reshape(sentence, (1, -1))), axis=1)
#     if label == 1:
#       return 'positive'
#     elif label == 2:
#       return 'negative'
#     else:
#       return 'neutral'

# print(predict_sentiment(model, vocab, 'this movie is great')) # positive
# print(predict_sentiment(model, vocab, 'this movie is so bad')) # negative

class Model:
  def __init__(self):
    self.model = None
    self.vocab = None

  def load_vocab(self, filename="vocab.pickle"):
    print("Loading vocabulary files.....")
    file_to_read = open(filename, "rb")
    self.vocab = pickle.load(file_to_read)
    file_to_read.close()

  def load_the_model(self, modelname='my_sa_model'):
    print("Loading model weights.....")
    self.model = tf.keras.models.load_model(modelname)

  def predict(self, sentence):
    sentence = tf.constant(self.vocab[sentence.split()])
    label = tf.argmax(self.model(tf.reshape(sentence, (1, -1))), axis=1)
    if label == 1:
      return 'positive'
    elif label == 2:
      return 'negative'
    else:
      return 'neutral'

  def stupid_infer(self, sentence):
    if (len(sentence.split()) > 1):
      return 'positive'
    else:
      return 'negative'