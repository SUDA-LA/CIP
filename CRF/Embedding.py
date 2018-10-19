class Embedding:
    def __init__(self, path, encoding='UTF-8'):
        fi = open(path, "r", encoding=encoding)
        embedding_map = {}

        while True:
            try:
                line = next(fi)
            except StopIteration:
                break
            line_split = line.split(' ')
            word = line_split[0]
            vector = line_split[1:]
            embedding_map[word] = vector

        self.data = embedding_map
        fi.close()

    def __call__(self, s: list):
        return [self.data.get(w, None) for w in s]
