class DataReader:
    def __init__(self, path, encoding='UTF-8'):
        fi = open(path, "r", encoding=encoding)
        self.data = []
        sentence = []
        tags = []
        self.SOS_token = 0
        self.EOS_token = 1
        self.tags_reverse = {self.SOS_token: 'SOS', self.EOS_token: 'EOS'}
        self.tags = {'SOS': self.SOS_token, 'EOS': self.EOS_token}
        self.chars_reverse = {}
        self.chars = {}
        self.max_length = 0
        while True:
            try:
                line = next(fi)
            except StopIteration:
                break
            line_split = line.split('\t')
            if len(line_split) > 1:
                word = line_split[1]
                chars = [w for w in word]
                tag = line_split[3]
                sentence += chars
                tags.append(tag)
                for c in chars:
                    if c not in self.chars:
                        c_id = len(self.chars)
                        self.chars[c] = c_id
                        self.chars_reverse[c_id] = c
                if tag not in self.tags:
                    t_id = len(self.tags)
                    self.tags[tag] = t_id
                    self.tags_reverse[t_id] = tag
            else:
                self.data.append([sentence, tags])
                if len(tags) > self.max_length:
                    self.max_length = len(tags)
                sentence = []
                tags = []

        fi.close()

    def cid2chars(self, s):
        return [self.chars_reverse[c_id] for c_id in s]

    def chars2cid(self, s):
        return [self.chars[t] for t in s]

    def tid2name(self, s):
        return [self.tags_reverse[t_id] for t_id in s]

    def name2tid(self, s):
        return [self.tags[t] for t in s]

    def get_data(self):
        return self.data

    def get_tag_size(self):
        return len(self.tags)

    def get_word_size(self):
        return len(self.chars)

    def get_max_length(self):
        return self.max_length
