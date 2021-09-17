import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def getwordgoing(word):
	maxlen = 100
	tokenizer = Tokenizer(num_words = len(word.split()), filters="""!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',""")
	tokenizer.fit_on_texts(word)
	word_index = tokenizer.word_index
	X_blurb = tokenizer.texts_to_sequences(word)
	X_blurb = pad_sequences(X_blurb, maxlen=maxlen)

	return X_blurb	


# with open("converTextToWord.pkl", "rb") as f:
#     convertWordToNumber = pickle.load(f)

# print(convertWordToNumber("my name is Dotun"))


print(getwordgoing("my name is Dotun"))