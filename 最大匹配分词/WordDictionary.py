import json


class WordDictionary:
    def __init__(self, path, encoding="UTF-8"):
        fd = open(path, 'r', encoding=encoding)
        lines = fd.readlines()
        json_str = ''.join(lines)
        word_dict = json.loads(json_str)
        self.word_set = {word['word'] for word in word_dict}
        fd.close()

    def __contains__(self, item):
        return item in self.word_set
