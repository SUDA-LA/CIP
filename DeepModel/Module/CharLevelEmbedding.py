import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class CharLevelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, char_embedding_dim: int, char_level_embedding_dim: int,
                 padding_idx=0, device="cpu", use_gru=True):
        super(CharLevelEmbedding, self).__init__()
        self.device = device
        self.num_embeddings = num_embeddings
        self.embedding_dim = char_embedding_dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=char_embedding_dim,
                                      padding_idx=padding_idx).to(device=device)
        self.embed_norm = nn.BatchNorm1d(char_embedding_dim).to(device=self.device)
        self.rnn = nn.GRU(
            char_embedding_dim,
            char_level_embedding_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        ).to(device=self.device) if use_gru else nn.LSTM(
            char_embedding_dim,
            char_level_embedding_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        ).to(device=self.device)
        self.use_gru = use_gru
        self.dropout = nn.Dropout(0.5).to(device=self.device)
        self.feat_norm = nn.BatchNorm1d(char_level_embedding_dim).to(device=self.device)

    def forward(self, chars):
        mask = chars.ne(self.padding_idx)                               # (WordCount, MaxCharCount)
        lens, indices = torch.sort(mask.sum(dim=1), descending=True)
        _, inverse_indices = indices.sort()
        max_len = lens[0]
        chars = chars[indices, :max_len]

        embed = self.embedding(chars)                                   # (WordCount, CharCount, C)
        # embed = self.embed_norm(embed.transpose(1, 2)).transpose(1, 2)  # (WordCount, CharCount, C)
        # embed = self.dropout(embed)
        pack_embed = pack_padded_sequence(embed, lens, True)
        if self.use_gru:
            _, hidden = self.rnn(pack_embed)      # (2, WordCount, EmbedDim // 2)
        else:
            _, (_, hidden) = self.rnn(pack_embed)
        char_level_feat = torch.cat(torch.unbind(hidden), dim=1)       # (WordCount, EmbedDim)
        # char_level_feat = self.feat_norm(char_level_feat)
        char_level_feat = char_level_feat[inverse_indices]
        return char_level_feat

