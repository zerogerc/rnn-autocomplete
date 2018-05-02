import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from zerogercrnn.lib.data.preprocess import read_jsons
from zerogercrnn.lib.embedding import Embeddings


def tsne_plot(emb: Embeddings, vocab):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for cur in range(emb.embeddings_tensor.size()[0]):
        if cur == -1:
            labels.append('SPACE')
        else:
            labels.append(vocab[cur])
        tokens.append(emb.embeddings_tensor[cur].numpy())

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    nearest(x, y, vocab, 'TryStatement10')

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def nearest(x, y, vocab, word):
    w_id = -1
    for i in range(len(vocab)):
        if vocab[i] == word:
            w_id = i
    if w_id == -1:
        raise Exception('No such word in vocabulary: {}'.format(word))

    px = x[w_id]
    py = y[w_id]
    p = np.array([px, py])

    points_with_distance = []
    for i in range(len(x)):
        points_with_distance.append((i, np.linalg.norm(np.array([x[i], y[i]]) - p)))

    print('Nearest to {}:'.format(vocab[w_id]))
    for c_p in sorted(points_with_distance, key=lambda x: x[1])[:10]:
        print(vocab[c_p[0]])


if __name__ == '__main__':
    emb = Embeddings(vector_file='/Users/zerogerc/Documents/diploma/GloVe/vectors.txt', embeddings_size=5)
    vocab = list(read_jsons('data/ast/non_terminals.json'))[0]
    vocab.append('EOF')
    tsne_plot(emb, vocab)
