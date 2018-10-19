import random

import torch
import visdom
from torch import nn
import torch.nn.functional as functional
from torch import optim
from Char2seqReader import DataReader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 300


class Model:
    def __init__(self, config):
        self.encoder = self.Encoder(config.char_size, config.embedding_dim, config.hidden_dim)
        self.decoder = self.Decoder(config.tag_size, config.hidden_dim, config.max_length)
        self.chars = None
        self.tags_reverse = None
        self.max_length = config.max_length
        self.SOS_token = 0
        self.EOS_token = 1

    def tag(self, sentence):
        assert self.chars is not None and self.tags_reverse is not None
        input_tensor = torch.tensor([self.chars[c] for c in sentence], dtype=torch.long, device=device)
        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)
        encoder_h_c = self.encoder.init_h_c()

        encoder_output_part, encoder_h_c = self.encoder(input_tensor, encoder_h_c)
        for ei in range(input_length):
            encoder_outputs[ei] = encoder_output_part[ei][0]

        decoder_h_c = (
            torch.cat((encoder_h_c[0][0], encoder_h_c[0][1]), dim=1).unsqueeze(0),
            torch.cat((encoder_h_c[1][0], encoder_h_c[1][1]), dim=1).unsqueeze(0)
        )

        decoder_input = torch.tensor([[self.SOS_token]], device=device)
        decoder_attentions = torch.zeros(self.max_length, self.max_length, device=device)

        tags = []

        for di in range(self.max_length):
            decoder_output, decoder_h_c, decoder_attention = self.decoder(
                decoder_input, decoder_h_c, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_attentions[di] = decoder_attention.data

            if topi.item() == self.EOS_token:
                break
            else:
                tags += [self.tags_reverse[topi.item()]]

            decoder_input = topi.squeeze().detach()

        return tags, decoder_attentions

    class Encoder(nn.Module):
        def __init__(self, char_size, embedding_size, hidden_size):
            super(Model.Encoder, self).__init__()
            self.hidden_size = hidden_size
            self.char_size = char_size
            self.embedding_size = embedding_size
            self.embedding = nn.Embedding(char_size, embedding_size)
            self.embedding.to(device=device)
            self.bi_lstm = nn.LSTM(embedding_size, hidden_size // 2, num_layers=1, bidirectional=True)
            self.bi_lstm.to(device=device)

        def init_h_c(self):
            return (torch.zeros(2, 1, self.hidden_size // 2, device=device),
                    torch.zeros(2, 1, self.hidden_size // 2, device=device))

        def forward(self, layer_input, h_c):
            embedded = self.embedding(layer_input).view(len(layer_input), 1, -1)
            output = embedded
            output, h_c = self.bi_lstm(output, h_c)
            return output, h_c

    class Decoder(nn.Module):
        def __init__(self, tag_size, hidden_size, max_length=MAX_LENGTH):
            super(Model.Decoder, self).__init__()
            self.hidden_size = hidden_size
            self.tag_size = tag_size
            self.max_length = max_length

            self.embedding = nn.Embedding(self.tag_size, self.hidden_size)
            self.embedding.to(device=device)
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn.to(device=device)
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.attn_combine.to(device=device)
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
            self.lstm.to(device=device)
            self.out = nn.Linear(self.hidden_size, self.tag_size)
            self.out.to(device=device)

        def forward(self, layer_input, h_c, encoder_outputs):
            embedded = self.embedding(layer_input).view(1, 1, -1)

            attn_weights = functional.softmax(
                self.attn(torch.cat((embedded[0], h_c[0][0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)

            output = functional.relu(output)
            output, h_c = self.lstm(output, h_c)

            output = functional.log_softmax(self.out(output[0]), dim=1)
            return output, h_c, attn_weights

        def init_h_c(self):
            return (torch.zeros(1, 1, self.hidden_size, device=device),
                    torch.zeros(1, 1, self.hidden_size, device=device))

    class Config:
        def __init__(self, hidden_dim, embedding_dim, tag_size=None, char_size=None, max_length=None):
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            self.tag_size = tag_size
            self.char_size = char_size
            self.max_length = max_length

    class ModelTrainer:
        def __init__(self, dataset_path, config):
            self.reader = DataReader(dataset_path)
            self.config = config
            if config.char_size is None:
                config.char_size = self.reader.get_word_size()
            if config.tag_size is None:
                config.tag_size = self.reader.get_tag_size()
            if config.max_length is None:
                config.max_length = self.reader.get_max_length()
            self.model = Model(config)
            self.model.chars = self.reader.chars
            self.model.tags_reverse = self.reader.tags_reverse
            self.model.SOS_token = self.reader.SOS_token
            self.model.EOS_token = self.reader.EOS_token

        def _tensors_from_data(self, data):
            return [[
                torch.tensor(self.reader.chars2cid(pair[0]), dtype=torch.long, device=device),
                torch.tensor(self.reader.name2tid(pair[1]), dtype=torch.long, device=device)
            ] for pair in data]

        def train_step(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor,
                       encoder: nn.Module, decoder: nn.Module,
                       encoder_optimizer: optim.Optimizer, decoder_optimizer: optim.Optimizer,
                       criterion: nn.Module, max_length=MAX_LENGTH):
            encoder_h_c = encoder.init_h_c()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            loss = 0

            encoder_output_part, encoder_h_c = encoder(input_tensor, encoder_h_c)
            for ei in range(input_length):
                encoder_outputs[ei] = encoder_output_part[ei][0]

            decoder_h_c = (
                torch.cat((encoder_h_c[0][0], encoder_h_c[0][1]), dim=1).unsqueeze(0),
                torch.cat((encoder_h_c[1][0], encoder_h_c[1][1]), dim=1).unsqueeze(0)
            )

            decoder_input = torch.tensor([[self.reader.SOS_token]], device=device)

            EOS_tensor = torch.tensor([self.reader.EOS_token], dtype=torch.long, device=device)

            use_teacher_forcing = True if random.random() < 0.5 else False
            if use_teacher_forcing:
                for di in range(max_length):
                    decoder_output, decoder_h_c, decoder_attention = decoder(
                        decoder_input, decoder_h_c, encoder_outputs
                    )
                    expect_tensor = target_tensor[di].unsqueeze(0) if di < target_length else EOS_tensor
                    loss += criterion(decoder_output, expect_tensor)
                    decoder_input = expect_tensor
                    if di > target_length:
                        break
            else:
                for di in range(max_length):
                    decoder_output, decoder_h_c, decoder_attention = decoder(
                        decoder_input, decoder_h_c, encoder_outputs
                    )
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()

                    expect_tensor = target_tensor[di].unsqueeze(0) if di < target_length else EOS_tensor

                    loss += criterion(decoder_output, expect_tensor)
                    if decoder_input.item() == self.reader.EOS_token or di > target_length:
                        break

            loss.backward()

            encoder_optimizer.step(None)
            decoder_optimizer.step(None)

            return loss.item() / target_length

        def train(self, learning_rate=0.01, max_epoch=300):
            data = self._tensors_from_data(self.reader.get_data())
            encoder_optimizer = optim.SGD(self.model.encoder.parameters(), lr=learning_rate)
            decoder_optimizer = optim.SGD(self.model.decoder.parameters(), lr=learning_rate)

            criterion = nn.NLLLoss()

            data_size = len(data)
            loss_visdom = 0
            vis = visdom.Visdom(env='C2S')
            step = 0
            show_step = 500

            for epoch in range(max_epoch):
                for train_iter in range(data_size):
                    pair = random.choice(data)

                    input_tensor = pair[0]
                    target_tensor = pair[1]

                    loss = self.train_step(input_tensor, target_tensor,
                                                       self.model.encoder, self.model.decoder,
                                                       encoder_optimizer, decoder_optimizer,
                                                       criterion, self.config.max_length)

                    loss_visdom += loss

                    if step % show_step == 0:
                        vis.line(X=torch.Tensor([step]), Y=torch.Tensor([loss_visdom / show_step]), win='loss',
                                 update='append' if step > 0 else None)
                        loss_visdom = 0

                        sentence = [self.reader.chars_reverse[c_id.item()] for c_id in input_tensor]
                        gt = [self.reader.tags_reverse[t_id.item()] for t_id in target_tensor]
                        pred, attentions_show = self.model.tag(sentence)
                        vis.text('<h5>输入</h5><p>' +
                                 ' '.join(sentence) +
                                 '</p><h5>目标</h5><p>' +
                                 ' '.join(gt) +
                                 '</p><h5>输出</h5><p>' +
                                 ' '.join(pred) +
                                 '</p>', win='tag')

                        vis.heatmap(
                            attentions_show.cpu().data,
                            win='attention',
                            opts=dict(
                                colormap='Electric',
                            )
                        )

                    step += 1
