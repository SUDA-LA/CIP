import numpy as np


class DataReader:
    UNK = '<UNK>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    PAD = '<PAD>'

    def __init__(self, train_path, test_path, dev_path, embed_path, encoding='UTF-8', device="cpu", char_level=False):
        self.device = device
        train_data = self.parse(train_path, encoding=encoding)
        test_data = self.parse(test_path, encoding=encoding)
        dev_data = self.parse(dev_path, encoding=encoding)

        self.word_set, self.char_set, self.tag_set = self.extract_sets(train_data[0], train_data[1])
        self.word_set = [self.PAD, self.UNK, self.SOS, self.EOS] + self.word_set
        self.char_set = [self.PAD, self.UNK] + self.char_set

        self.word_dict = {w: i for i, w in enumerate(self.word_set)}
        self.char_dict = {c: i for i, c in enumerate(self.char_set)}
        self.tag_dict = {t: i for i, t in enumerate(self.tag_set)}

        self.word_reverse_dict = {i: w for i, w in enumerate(self.word_set)}
        self.char_reverse_dict = {i: c for i, c in enumerate(self.char_set)}
        self.tag_reverse_dict = {i: t for i, t in enumerate(self.tag_set)}

        self.unk_wi = self.word_dict[self.UNK]
        self.sos_wi = self.word_dict[self.SOS]
        self.eos_wi = self.word_dict[self.EOS]
        self.pad_wi = self.word_dict[self.PAD]

        self.pad_ci = self.char_dict[self.PAD]
        self.unk_ci = self.char_dict[self.UNK]

        self.tarin_sentence_size = len(train_data[0])
        self.word_size = len(self.word_set)
        self.tag_size = len(self.tag_set)
        self.char_size = len(self.char_set)

        self.embed = self._load_embed(embed_path, encoding=encoding)
        self.embed_size = self.embed.size(1)

        self.train_dataset = self.to_dataset(train_data, char_level)
        self.test_dataset = self.to_dataset(test_data, char_level)
        self.dev_dataset = self.to_dataset(dev_data, char_level)

    def _load_embed(self, embed_path, encoding='UTF-8'):
        with open(embed_path, "r", encoding=encoding) as fe:
            lines = [line for line in fe]
        splits = [line.split() for line in lines]

        embed_words, embed_val = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])

        self._extend(embed_words)

        part_embed_tensor = np.array(embed_val)
        embed_indices = [self.word_dict[embed_word] for embed_word in embed_words]
        embed_tensor = np.zeros((self.word_size, part_embed_tensor.size(1)))
        self._init_embed(embed_tensor)
        embed_tensor[embed_indices] = part_embed_tensor
        return embed_tensor

    def _extend(self, embed_words):
        extend_word = [embed_word for embed_word in embed_words if embed_word not in self.word_dict]
        extend_char = [char for char in ''.join(extend_word) if char not in self.char_dict]

        self.word_set = sorted(set(self.word_set + extend_word) - {self.PAD})
        self.word_set = [self.PAD] + self.word_set
        self.word_dict = {w: i for i, w in enumerate(self.word_set)}
        self.word_reverse_dict = {i: w for i, w in enumerate(self.word_set)}

        self.char_set = sorted(set(self.char_set + extend_char) - {self.PAD})
        self.char_set = [self.PAD] + self.char_set
        self.char_dict = {c: i for i, c in enumerate(self.char_set)}
        self.char_reverse_dict = {i: c for i, c in enumerate(self.char_set)}

        self.unk_wi = self.word_dict[self.UNK]
        self.sos_wi = self.word_dict[self.SOS]
        self.eos_wi = self.word_dict[self.EOS]
        self.pad_wi = self.word_dict[self.PAD]

        self.unk_ci = self.char_dict[self.UNK]
        self.pad_ci = self.char_dict[self.PAD]

        self.word_size = len(self.word_set)
        self.char_size = len(self.char_set)

    @staticmethod
    def _init_embed(tensor):
        pass

    @staticmethod
    def parse(path, encoding):
        sentences = []
        labels = []
        with open(path, "r", encoding=encoding) as fi:
            sentence = []
            tags = []
            for line in fi:
                line_split = line.split('\t')
                if len(line_split) > 1:
                    word = line_split[1]
                    tag = line_split[3]
                    sentence.append(word)
                    tags.append(tag)
                else:
                    sentences.append(sentence)
                    sentence = []
                    labels.append(tags)
                    tags = []
        return sentences, labels

    @staticmethod
    def extract_sets(sentences, tags):
        word_set = sorted(set(word for sentence in sentences for word in sentence))
        char_set = sorted(set(char for word in word_set for char in word))
        tag_set = sorted(set(tag for tag_sequence in tags for tag in tag_sequence))
        return word_set, char_set, tag_set

    def to_dataset(self, data, n_gram=2):
        x, y, chars, lens = [], [], [], []
        for sentence, label in zip(*data):
            words_id = [self.word_dict.get(word, self.unk_wi) for word in sentence]
            n_words_id = [
                [self.sos_wi] * -offset +
                (words_id[:offset] if offset < 0 else words_id[offset:]) +
                [self.eos_wi] * offset
                for offset in range(-n_gram, n_gram + 1)
            ]
            tags_id = [self.tag_dict[tag] for tag in label]
            x.append(np.array(n_words_id).transpose())
            y.append(np.array(tags_id))
            lens.append(len(tags_id))

        # TODO fix return

    @staticmethod
    def padding_data(x, y):


    def wid2word(self, s):
        return [self.word_reverse_dict.get(wid, self.UNK) for wid in s]

    def word2wid(self, s):
        return [self.word_dict.get(word, self.unk_wi) for word in s]

    def tag2tid(self, s):
        return [self.tag_dict[tag] for tag in s]

    def tid2tag(self, s):
        return [self.tag_reverse_dict[tid] for tid in s]
