import torch
import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.encoding = torch.zeros(seq_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, seq_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class self_attention(nn.Module):
    def __init__(self, head, d_model):
        super().__init__()
        assert d_model % head == 0, "head must divide d_model"
        self.attention_head = d_model // head
        self.head = head
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        N = q.shape[0]
        key_len, query_len, value_len = k.shape[1], q.shape[1], v.shape[1]
        #print(N, key_len, query_len, value_len)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = q.reshape(N, query_len, self.head, self.attention_head)
        k = k.reshape(N, key_len, self.head, self.attention_head)
        v = v.reshape(N, value_len, self.head, self.attention_head)

        # q: (N, query_len, head, attention_head)
        # k: (N, key_len, head, attention_head)
        # -> (N, head, qurey_len, key_len)
        temp = torch.einsum("nqha,nkha->nhqk", [q, k])

        if mask is not None:
            temp = temp.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(temp / math.sqrt(self.attention_head), dim=-1)

        # temp: (N, head, query_len, key_len)
        # v: (N, value_len, head, attention_head)
        # -> (N, query_len, head, attention_head)
        output = torch.einsum(
            "nhqk,nkha->nqha", [attention, v]
        )  # multiply across key_len==value_len

        output = output.reshape(N, query_len, self.head * self.attention_head)
        output = self.output_layer(output)

        return output


class fc_layer(nn.Module):
    def __init__(self, d_model, hidden_size, dropout):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        return self.output(self.dropout(self.relu(self.fc(x))))


class transformer_block(nn.Module):
    def __init__(self, d_model, head, dropout, forward_expansion):
        super().__init__()
        self.mha = self_attention(head, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.ReLU(),
            nn.Linear(forward_expansion * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        temp = self.mha(q, k, v, mask)
        temp = self.dropout(self.norm1(temp + q))
        x = self.fc(temp)
        out = self.dropout(self.norm2(x + temp))

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        d_model,
        num_layers,
        head,
        dropout,
        forward_expansion,
        device,
        max_length,
    ):
        super().__init__()
        self.device = device
        self.word_embedding = Embeddings(d_model, src_vocab_size)
        self.postion_embedding = PositionalEmbedding(d_model, max_length)
        self.layers = nn.ModuleList(
            [
                transformer_block(d_model, head, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape

        out = self.dropout(self.word_embedding(x) + self.postion_embedding(x))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, d_model, head, forward_expansion, dropout, device):
        super().__init__()
        self.device = device
        self.mha = self_attention(head, d_model)
        self.tranformer_block = transformer_block(
            d_model, head, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.mha(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.tranformer_block(query, key, value, src_mask)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        d_model,
        num_layers,
        head,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super().__init__()
        self.device = device
        self.word_embedding = Embeddings(d_model, trg_vocab_size)
        self.postion_embedding = PositionalEmbedding(d_model, max_length)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, head, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        out = self.dropout(self.word_embedding(x) + self.postion_embedding(x))

        for layer in self.layers:
            out = layer(out, encoder_out, encoder_out, src_mask, trg_mask)

        #print(out.shape)
        out = self.fc(out)
        out = nn.Softmax(dim=-1)(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        d_model=512,
        num_layers=6,
        head=8,
        forward_expansion=4,
        dropout=0,
        device="cuda",
        max_length=100,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_layers,
            head,
            dropout,
            forward_expansion,
            device,
            max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            d_model,
            num_layers,
            head,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, tgr_len = trg.shape
        trg_mask = torch.tril(torch.ones((tgr_len, tgr_len))).expand(
            N, 1, tgr_len, tgr_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoder_out = self.encoder(src, src_mask)
        out = self.decoder(trg, encoder_out, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(
        src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device
    ).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)
