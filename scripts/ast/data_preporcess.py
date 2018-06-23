from zerogercrnn.lib.constants import ENCODING
from zerogercrnn.lib.log import tqdm_lim


def create_smaller_file(file, new_file, lim):
    in_file = open(file, mode='r', encoding=ENCODING)
    out_file = open(new_file, mode='w', encoding=ENCODING)

    for line in tqdm_lim(in_file, lim=lim):
        out_file.write(line)

    in_file.close()
    out_file.close()


if __name__ == '__main__':
    create_smaller_file('data/ast/programs_eval.json', 'data/ast/programs_eval_5k.json', lim=5000)
