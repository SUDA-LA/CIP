from WordDictionary import WordDictionary


class Segmenter:
    def __init__(self, dict_path=".\\data\\word.dict", forward=True):
        self.word_dict = WordDictionary(dict_path)
        self.forward = forward

    def segment(self, s, max_length):
        assert isinstance(s, str)
        length = len(s)
        result = []
        if self.forward:
            p1 = 0
            while p1 < length:
                i = max_length
                while not s[p1:p1+i] in self.word_dict and i != 1:
                    i = i - 1
                result.append(s[p1:p1+i])
                p1 = p1 + i
            return result
        else:
            p1 = length
            while p1 > 0:
                i = min(max_length, p1)
                while not s[p1 - i:p1] in self.word_dict and i != 1:
                    i = i - 1
                result.append(s[p1 - i:p1])
                p1 = p1 - i
            result.reverse()
            return result
