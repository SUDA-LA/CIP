import torch

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
            vector = [float(v) for v in line_split[1:] if self._is_float(v)]
            embedding_map[word] = vector
            self.embedding_dim = len(vector)

        self.data = embedding_map
        self.none_v = [0 for i in range(self.embedding_dim)]
        fi.close()

    def __call__(self, s: list, type_torch=False):
        if type_torch:
            return torch.tensor([self.data.get(w, self.none_v) for w in s], requires_grad=False)
        else:
            return [self.data.get(w, None) for w in s]

    @staticmethod
    def _is_float(string):
        try:
            float(string)
            return True
        except Exception:
            return False

    def get_embedding_dim(self):
        return self.embedding_dim
