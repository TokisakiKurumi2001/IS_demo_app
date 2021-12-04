# Import module
import tensorflow as tf
import pickle
import re


class Model:
    def __init__(self):
        self.model = None
        self.vocab = None
        self.appos = {
            "aren't": "are not",
            "can't": "cannot",
            "cant": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "I would",
            "i'd": "I had",
            "i'll": "I will",
            "i'm": "I am",
            "im": "I am",
            "isn't": "is not",
            "it's": "it is",
            "it'll": "it will",
            "i've": "I have",
            "let's": "let us",
            "mightn't": "might not",
            "mustn't": "must not",
            "shan't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "we'd": "we would",
            "we're": "we are",
            "weren't": "were not",
            "we've": "we have",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where's": "where is",
            "who'd": "who would",
            "who'll": "who will",
            "who're": "who are",
            "who's": "who is",
            "who've": "who have",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have",
            "'re": " are",
            "wasn't": "was not",
            "we'll": " will",
            "didn't": "did not",
            "gg": "going"
        }

    def preprocess_text(self, sentence):
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',
                      '', sentence)
        text = re.sub('@[^\s]+', '', text)
        text = text.lower().split()
        reformed = [self.appos[word]
                    if word in self.appos else word for word in text]
        reformed = " ".join(reformed)
        text = re.sub('&[^\s]+;', '', reformed)
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def load_vocab(self, filename="vocab.pickle"):
        print("Loading vocabulary files.....")
        file_to_read = open(filename, "rb")
        self.vocab = pickle.load(file_to_read)
        file_to_read.close()

    def load_the_model(self, modelname='my_sa_model'):
        print("Loading model weights.....")
        self.model = tf.keras.models.load_model(modelname)

    def predict(self, sentence):
        sentence = self.preprocess_text(sentence)
        sentence = tf.constant(self.vocab[sentence.split()])
        result = tf.squeeze(self.model(tf.reshape(sentence, (1, -1))))
        return_dict = [('Neutral', result[0].numpy()), ('Positive',
                                                        result[1].numpy()), ('Negative', result[2].numpy())]
        return_dict.sort(key=lambda x: x[1], reverse=True)
        return return_dict

    def stupid_infer(self, sentence):
        if (len(sentence.split()) > 1):
            return 'positive'
        else:
            return 'negative'
