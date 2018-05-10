import json
from abc import abstractmethod

from zerogercrnn.lib.constants import ENCODING
from zerogercrnn.lib.log import tqdm_lim


def write_json(file, raw_json):
    """Writes json as is to file."""

    with open(file, mode='w', encoding=ENCODING) as f:
        f.write(json.dumps(raw_json))


def read_lines(file, total=None, lim=None):
    """Returns generator of lines from file.

        :param file: path to file
        :param total: total number of lines in file
        :param lim: limit on number of read lines
    """
    with open(file, mode='r', encoding=ENCODING) as f:
        for line in tqdm_lim(f, total=total, lim=lim):
            yield line


def read_jsons(*files, lim=None):
    """Reads jsons from passed files. Suppose files to contain json lines separated by newlines.

        :param files: files to read jsons from
        :param lim: limit number of read jsons for all files
    """
    for file in files:
        for line in read_lines(file, lim=lim):
            yield json.loads(line)


def read_json(file):
    """Reads single json from file.

        :param file: file to read jsons from
    """
    return list(read_jsons(file))[0]


class JsonExtractor:
    """Extracts some info from passed json. See specific implementations for more info."""

    @abstractmethod
    def extract(self, raw_json):
        pass


class JsonListKeyExtractor(JsonExtractor):
    """Extracts values by specified key if it present. Suppose json to be a list of jsons."""

    def __init__(self, key):
        self.key = key

    def extract(self, raw_json):
        for node in raw_json:
            if node == 0:
                break

            if self.key in node:
                yield node[self.key]


def extract_jsons_info(extractor: JsonExtractor, *files, lim=None):
    """Read jsons from files and run extractor on them."""
    for raw_json in read_jsons(*files, lim=lim):
        yield extractor.extract(raw_json)


def test():
    nt_extractor = JsonListKeyExtractor(key='type')

    for info_gen in extract_jsons_info(nt_extractor, 'data/programs_eval.json', lim=10):
        for val in info_gen:
            print(val)


if __name__ == '__main__':
    test()
