import torch
from torch import nn


class CRF(nn.Module):
    def __init__(self, tags_size, hidden_dim, device="cpu"):
        super(CRF, self).__init__()
        self.device = device
        self.tags_size = tags_size
        # (i_1 -> i)
        self.trans = None
        self.trans_linear = nn.Linear(hidden_dim, tags_size * tags_size).to(device=device)
        self.trans_from_start = None
        self.trans_from_start_linear = nn.Linear(hidden_dim, tags_size).to(device=device)
        self.trans_to_end = None
        self.trans_to_end_linear = nn.Linear(hidden_dim, tags_size).to(device=device)
        # self.init_parameter()

    def init_parameter(self):
        std = (1 / self.tags_size) ** 0.5  # copy from zysite
        nn.init.normal_(self.trans, mean=0, std=std)
        nn.init.normal_(self.trans_from_start, mean=0, std=std)
        nn.init.normal_(self.trans_to_end, mean=0, std=std)

    def forward(self, emit, hidden, target, mask):
        # input: (L, B, C)
        _, b, _ = emit.shape
        self.trans = self.trans_linear(hidden).view(-1, self.tags_size, self.tags_size)
        self.trans_from_start = self.trans_from_start_linear(hidden)
        self.trans_to_end = self.trans_to_end_linear(hidden)
        log_z = self.get_log_z(emit, mask)
        score = self.get_score(emit, target, mask)
        return (log_z - score) / b

    def get_log_z(self, emit, mask):
        l, _, c = emit.shape
        log_alpha = emit[0] + self.trans_from_start  # (B, C)

        for i in range(1, l):
            trans_i = self.trans                                # (B, C, C)
            emit_i = emit[i].unsqueeze(1)                       # (B, 1, C)
            mask_i = mask[i].unsqueeze(1).expand_as(log_alpha)  # (B, C)
            scores = trans_i + emit_i + log_alpha.unsqueeze(2)  # (B, C, C)
            scores = torch.logsumexp(scores, dim=1)             # (B, C)
            log_alpha[mask_i] = scores[mask_i]                  # (B, C)

        return torch.logsumexp(log_alpha + self.trans_to_end, dim=1).sum()

    def get_score(self, emit, target, mask):
        l, b, c = emit.shape
        scores = torch.zeros(l, b, device=self.device)
        idx = [i for i in range(b)]

        scores[1:] = self.trans[idx, target[:-1], target[1:]]
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        score = scores.masked_select(mask).sum()

        ends = mask.sum(dim=0).view(1, -1) - 1

        score += self.trans_from_start[idx, target[0]].sum()
        score += self.trans_to_end[idx, target.gather(dim=0, index=ends)].sum()

        return score

    def viterbi(self, emit, hidden, mask):
        l, b, c = emit.shape
        lens = mask.sum(dim=0)
        delta = torch.zeros(l, b, c, device=self.device)
        path = torch.zeros(l, b, c, dtype=torch.long, device=self.device)

        self.trans = self.trans_linear(hidden).view(-1, self.tags_size, self.tags_size)
        self.trans_from_start = self.trans_from_start_linear(hidden)
        self.trans_to_end = self.trans_to_end_linear(hidden)

        delta[0] = self.trans_from_start + emit[0]                  # (B, C)

        for i in range(1, l):
            trans_i = self.trans                                    # (B, C, C)
            emit_i = emit[i].unsqueeze(1)                           # (B, 1, C)
            scores = trans_i + emit_i + delta[i - 1].unsqueeze(2)   # (B, C, C) + (B, 1, C) + (B, C, 1) -> (B, C, C)
            delta[i], path[i] = torch.max(scores, dim=1)

        predicts = []

        for n, length in enumerate(lens):
            prev = torch.argmax(delta[length - 1, n] + self.trans_to_end[n])
            predict = [prev]

            for i in reversed(range(1, length)):
                prev = path[i, n, prev]
                predict.append(prev)

            predicts.append(torch.tensor(predict, device=self.device).flip(0))

        return torch.cat(predicts)

    def get_trans(self, hidden):
        return self.trans_from_start_linear(hidden), \
               self.trans_linear(hidden).view(-1, self.tags_size, self.tags_size), \
               self.trans_to_end_linear(hidden)
