import torch
import visdom
from torch import nn
import torch.nn.functional as functional
from torch import optim
from DataReader import DataReader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.to(device=device)
        self.hidden_dim = config.hidden_dim
        self.lstm = nn.LSTM(config.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True)
        self.lstm.to(device=device)
        self.hidden2tag = nn.Linear(config.hidden_dim, config.tag_size)
        self.hidden2tag.to(device=device)
        self.hidden = self.init_hidden()

        self.tags_reverse = None
        self.words_reverse = None
        self.words = None
        self.tags = None

    def init_hidden(self):
        # (h0, c0)
        return (torch.zeros(2, 1, self.hidden_dim // 2, device=device),
                torch.zeros(2, 1, self.hidden_dim // 2, device=device))

    def forward(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = functional.log_softmax(tag_space, dim=1)
        return tag_scores

    def tag(self, sentence):
        return [self.tags_reverse.get(int(t_id), 'ERROR') for t_id in torch.argmax(self(sentence), 1)]

    class Config:
        def __init__(self, hidden_dim, embedding_dim, tag_size=None, vocab_size=None):
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.tag_size = tag_size
            self.vocab_size = vocab_size

    class ModelTrainer:
        def __init__(self, dataset_path, config):
            self.reader = DataReader(dataset_path)
            if config.vocab_size is None:
                config.vocab_size = self.reader.get_word_size()
            if config.tag_size is None:
                config.tag_size = self.reader.get_tag_size()
            self.model = Model(config)

        def train(self):
            model = self.model
            loss_function = nn.NLLLoss()
            optimizer = optim.SGD(self.model.parameters(), lr=0.1)
            reader = self.reader
            training_data = reader.get_seg_data(name=False)
            training_tags = reader.get_pos_data(name=False)
            model.tags_reverse = reader.tags_reverse
            model.words_reverse = reader.words_reverse
            model.words = reader.words
            model.tags = reader.tags
            vis = visdom.Visdom(env=u'Tagger')
            step = 0
            loss_i = 0
            for epoch in range(300):
                for sentence, tag in zip(training_data, training_tags):

                    model.zero_grad()
                    sentence_tensor = torch.tensor(sentence, dtype=torch.long, device=device)
                    tag_scores = model(sentence_tensor)
                    loss = loss_function(tag_scores, torch.tensor(tag, device=device))
                    loss.backward()
                    optimizer.step()
                    loss_i += int(loss)

                    if step % 1000 == 0:
                        vis.line(X=torch.tensor([step]), Y=torch.tensor([(loss_i / 1000) if step > 0 else loss_i]),
                                 win='loss', update='append' if step > 0 else None)
                        loss_i = 0
                        gt = reader.tid2name(tag)
                        pred = model.tag(sentence_tensor)
                        mark = [True if gt_t == pred_t else False for gt_t, pred_t in zip(gt, pred)]

                        vis.text('<h5>输入</h5><p>' +
                                 ' '.join([
                                     w if m else '<font color="red">' + w + '</font>'
                                     for m, w in zip(mark, reader.wid2name(sentence))
                                 ]) +
                                 '</p><h5>目标</h5><p>' +
                                 ' '.join([
                                     t if m else '<font color="green">' + t + '</font>'
                                     for m, t in zip(mark, gt)
                                 ]) +
                                 '</p><h5>输出</h5><p>' +
                                 ' '.join([
                                     t if m else '<font color="red">' + t + '</font>'
                                     for m, t in zip(mark, pred)
                                 ]) +
                                 '</p>', win='tag')
                    step += 1