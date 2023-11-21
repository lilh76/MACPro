import torch as th
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, args, input_shape):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.token_dim = input_shape
        self.transformer = Transformer(args, self.token_dim, args.z_dim, args.enc_heads, args.enc_depth, args.z_dim)
        self.fc = nn.Sequential(
            nn.Linear(args.z_dim, args.z_dim),
            nn.ReLU(),
            nn.Linear(args.z_dim, args.z_dim * 2),
        )
        self.z_dim = args.z_dim
        
    def forward(self, batch_trajectory, trajectory_mask):
        outputs, _ = self.transformer.forward(batch_trajectory, trajectory_mask)
        outputs = self.fc(outputs)

        bs = outputs.shape[0]
        p_mu     = th.zeros((bs, 1, self.z_dim)).to(self.args.device)
        p_lgvar = th.zeros((bs, 1, self.z_dim)).to(self.args.device)
        mus     = outputs[:, :, : self.z_dim]
        lgvars = outputs[:, :, self.z_dim :]
        mu     = th.cat((p_mu,     mus), dim=1)
        lgvar = th.cat((p_lgvar, lgvars), dim=1)
        EPS = 1e-7
        var = th.exp(lgvar) + EPS
        T = 1. / (var + EPS)
        poe_mu = th.sum(mu * T, dim=1) / th.sum(T, dim=1)
        poe_var = 1. / th.sum(T, dim=1)
        poe_std = th.sqrt(poe_var)
        eps = th.randn_like(poe_std)
        encoded_z = eps * poe_std + poe_mu
        mu_lgvar = th.cat((poe_mu, th.log(poe_var)), dim = -1)
        return encoded_z, mu_lgvar

class Transformer(nn.Module):

    def __init__(self, args, input_dim, emb,         heads,     depth, output_dim):
        super().__init__()

        self.args = args

        self.num_tokens = output_dim

        self.token_embedding = nn.Linear(input_dim, emb).to(args.device)

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(args, emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks).to(args.device)

        self.toprobs = nn.Linear(emb, output_dim).to(args.device)

    def forward(self, x, mask):

        tokens = self.token_embedding(x)

        b, t, e = tokens.size()

        y, mask = self.tblocks((tokens, mask))

        y = self.toprobs(y.view(b * t, e)).view(b, t, self.num_tokens)

        return y, tokens

class TransformerBlock(nn.Module):

    def __init__(self, args, emb, heads, mask, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.args = args

        self.attention = SelfAttention(args, emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb).to(args.device)
        self.norm2 = nn.LayerNorm(emb).to(args.device)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        ).to(args.device)

        self.do = nn.Dropout(dropout).to(args.device)

    def forward(self, x_mask):
        x, mask = x_mask

        attended = self.attention(x, mask)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask

class SelfAttention(nn.Module):
    def __init__(self, args, emb, heads=8, mask=False):
        super().__init__()

        self.args = args

        self.emb = emb
        self.heads = heads
        self.mask = mask
        assert not mask, "We do not consider mask in this project"

        self.tokeys = nn.Linear(emb, emb * heads, bias=False).to(args.device)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False).to(args.device)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False).to(args.device)

        self.unifyheads = nn.Linear(heads * emb, emb).to(args.device)

    def forward(self, x, mask):

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))
        dot = th.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.squeeze(-1)
            repeated_mask = mask.unsqueeze(1).repeat(1, h * t, 1)
            repeated_mask = repeated_mask.reshape(b, h, t, t).reshape(b * h, t, t)

            dot = dot.masked_fill(repeated_mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        out = th.bmm(dot, values).view(b, h, t, e)

        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = th.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval
