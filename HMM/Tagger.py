import pickle
from DataReader import DataReader


class Tagger:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def train(self, data_path):
        dr = DataReader(data_path)
        data = dr.get_seg_data()
        pos = dr.get_pos_data()
        data_count = len(data)
        self.model = {'p': {}, 'w': {}, 'pc': 0, 'v': None}
        pc = 0
        pos_set = {'START': 0, 'STOP': 1}
        for i in range(data_count):
            word_count = len(data[i])
            pc += word_count

            for j in range(word_count):
                if j == 0:
                    self.model['p'][('START', pos[i][j])] = self.model['p'].setdefault(('START', pos[i][j]), 0) + 1
                else:
                    self.model['p'][(pos[i][j - 1], pos[i][j])] = \
                        self.model['p'].setdefault((pos[i][j - 1], pos[i][j]), 0) + 1

                self.model['w'][(data[i][j], pos[i][j])] = self.model['w'].setdefault((data[i][j], pos[i][j]), 0) + 1
                self.model['p'][(pos[i][j])] = self.model['p'].setdefault((pos[i][j]), 0) + 1
                if pos[i][j] not in pos_set:
                    pos_set.update({pos[i][j]: len(pos_set)})

            self.model['p'][(pos[i][word_count - 1], 'STOP')] = \
                self.model['p'].setdefault((pos[i][word_count - 1], 'STOP'), 0) + 1

        self.model['pc'] = pc
        self.model['v'] = pos_set

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

    def tag(self, s):
        assert self.model
        v_count = len(self.model['v'])
        pi = [[0 for i in range(v_count)]]
        pi[0][0] = 1
        pos_name = [key for (key, index) in self.model['v'].items()]
        word_count = len(s)
        pos = []
        for i in range(word_count):
            pk = [max([pi[i][w_index] *
                ((self.model['p'].get((w, v), 0) + 1) / (self.model['p'].get(w, 0) + v_count)) *
                ((self.model['w'].get((s[i], v), 0) + 1) / v_count)
                for (w, w_index) in self.model['v'].items()])
                for (v, v_index) in self.model['v'].items()]
            pi.append(pk)
            pos.append(pos_name[pk.index(max(pk))])

        return pos

if __name__ == '__main__':
    import os
    tagger = Tagger()
    if not os.path.exists('.\\model'):
        os.mkdir('.\\model')
    tagger.train('.\\data\\train.conll')
    tagger.save_model('.\\model\\model.pickle')
    tagger.load_model('.\\model\\model.pickle')