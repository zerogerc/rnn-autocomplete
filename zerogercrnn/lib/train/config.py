import json


class Config:
    def read_from_file(self, file):
        with open(file=file) as json_file:
            data = json.load(json_file)
            for attr in data:
                setattr(self, attr, data[attr])

    def write_to_file(self, file):
        with open(file=file, mode='w') as json_file:
            data = {}
            for attr in list(filter(lambda x: not x.startswith('__'), dir(self))):
                if attr != 'read_from_file' and attr != 'write_to_file':
                    data[attr] = getattr(self, attr)
            json.dump(data, json_file)


if __name__ == '__main__':
    attributes = [
        'learning_rate',
        'weight_decay',
        'seq_len',
        'batch_size',
        'embedding_size',
        'hidden_size',
        'num_layers',
        'dropout',
        'weight_decay',
        'epochs',
        'decay_after_epoch',
        'loss_function',
        'optimizer'
    ]
    # parameters = Config()
    # parameters.write_to_file(os.path.join(os.getcwd(), 'params_c.json'))
