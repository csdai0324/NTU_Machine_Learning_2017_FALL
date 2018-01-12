import json
import gensim
import jieba
import matplotlib.pyplot as plt
import matplotlib as  mpl
from matplotlib.font_manager import FontManager, FontProperties  
from gensim.models import word2vec
from sklearn.manifold import TSNE
from adjustText import adjust_text

def cut_sentence():
	jieba.set_dictionary('./dict.txt.big')
	all_sents = []
	with open('./all_sents.txt', 'r') as data:
	    for line in data:
	        sent = line.strip()
	        all_sents.append(sent)
	all_sents_cut = []
	for sent in all_sents:
	    s = []
	    words = jieba.cut(sent, cut_all=False)
	    for word in words:
	        s.append(word.strip())
	    all_sents_cut.append(s)

	return all_sents_cut

def train_word2vec(all_sents_cut):
	model = word2vec.Word2Vec(all_sents_cut, size=256, min_count=15, window=5)
	model.save('./myword2vecmodel')
	#model = word2vec.Word2Vec.load('./myword2vecmodel')
	return model

def tsne(model):
	vocab = list(model.wv.vocab)
	X = model[vocab]
	tsne = TSNE(n_components=2)
	X_tsne = tsne.fit_transform(X)
	return X_tsne, vocab

def plot(Xs, Ys, Texts):
    plt.plot(Xs, Ys, 'o')
    texts = [plt.text(X, Y, Text) for X, Y, Text in zip(Xs, Ys, Texts)]
    plt.figure(figsize=(800, 600))
    plt.title(str(adjust_text(texts, Xs, Ys, arrowprops=dict(arrowstyle='->', color='b'))))

def main():
	X_tsne, vocab = tsne(train_word2vec(cut_sentence()))
	texts = ['聊天', '吃飯', '你', '我']
	indexes = []
	for text in texts:
	    indexes.append(vocab.index(text))
	Xs = []
	Ys = []
	for index in indexes:
	    x, y = X_tsne[index]
	    Xs.append(x)
	    Ys.append(y)
	plot(Xs, Ys, texts)

if __name__ == '__main__':
	main()

