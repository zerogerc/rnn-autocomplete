import torch

ENCODING = 'ISO-8859-1'


class Embeddings:

    def __init__(self, embeddings_size, vector_file):
        self.embedding_size = embeddings_size
        self.vector_file = vector_file
        self._read_embeddings(vector_file)

    def get_embedding(self, id):
        if id in self.embeddings.keys():
            return self.embeddings[id]
        else:
            print('Return <unk> for id: {}'.format(id))
            return self.unk_embedding

    def index_select(self, index, out=None):
        return torch.index_select(self.embeddings_tensor, dim=0, index=index, out=out)

    def cuda(self):
        self.embeddings_tensor = self.embeddings_tensor.cuda()

        self.unk_embedding = self.unk_embedding.cuda()
        for k in self.embeddings.keys():
            self.embeddings[k] = self.embeddings[k].cuda()

    def _read_embeddings(self, vector_file):
        self.embeddings = {}
        max_emb_id = 0
        for l in open(vector_file, mode='r', encoding=ENCODING):
            numbers = l.split(' ')
            assert len(numbers) == self.embedding_size + 1

            id = numbers[0]
            cur_emb = torch.FloatTensor([float(x) for x in numbers[1:]])

            if id == '<unk>':
                self.unk_embedding = cur_emb
            else:
                assert int(id) not in self.embeddings.keys()
                max_emb_id = max(int(id), max_emb_id)
                self.embeddings[int(id)] = cur_emb


        self.embeddings_tensor = torch.FloatTensor(max_emb_id + 1, self.embedding_size)
        for k, v in self.embeddings.items():
            self.embeddings_tensor[k] = v
