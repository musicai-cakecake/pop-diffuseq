from transformers import AutoConfig
# from transformers import BertEncoder
# from transformers.models.bert.modeling_bert import BertEncoder, BertModel
# from transformers.models.bert.modeling_bert import BertModel
from .new_bert_model import BertEncoder
import torch

import numpy as np
import torch as th
import torch.nn as nn
# import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


def get_output_head_mask(output_head_mask, expose_range_min, expose_range_max):
    for i in range(len(output_head_mask)):
        if expose_range_min <= i <= expose_range_max:
            output_head_mask[i] = 1.0
    return output_head_mask


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
            self,
            input_dims,  # 128
            output_dims,  # 128
            hidden_t_dim,  # 128
            dropout=0,  # 0.1
            config=None,  # None
            config_name='bert-base-uncased',
            vocab_size=None,
            init_pretrained='no',  # 'no'
            logits_mode=1,
            learned_mean_embed=False,  # True
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims  # 128
        self.hidden_t_dim = hidden_t_dim  # 128
        self.output_dims = output_dims  # 128
        self.dropout = dropout  # 0.1
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size  # dim of embedding, 768

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                               nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))

        # if init_pretrained == 'bert':
        #     print('initializing from pretrained bert...')
        #     print(config)
        #     temp_bert = BertModel.from_pretrained(config_name, config=config)
        #
        #     self.word_embedding = temp_bert.embeddings.word_embeddings
        #     with th.no_grad():
        #         self.lm_head.weight = self.word_embedding.weight
        #     # self.lm_head.weight.requires_grad = False
        #     # self.word_embedding.weight.requires_grad = False
        #
        #     self.input_transformers = temp_bert.encoder
        #     self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        #     self.position_embeddings = temp_bert.embeddings.position_embeddings
        #     self.LayerNorm = temp_bert.embeddings.LayerNorm
        #
        #     del temp_bert.embeddings
        #     del temp_bert.pooler

        if init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        else:
            assert False, "invalid type of init_pretrained"

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                  nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

        if learned_mean_embed:
            self.mean_embed = nn.Parameter(th.randn(input_dims), requires_grad=True)
            nn.init.normal_(self.mean_embed, mean=0, std=input_dims ** -0.5)
        else:
            self.mean_embed = None

        self.lm_head = nn.Linear(self.input_dims, vocab_size)

        # self.lm_head_1 = nn.Linear(self.input_dims, vocab_size)  # 128 to 307
        # self.lm_head_2 = nn.Linear(self.input_dims, vocab_size)
        # self.lm_head_3 = nn.Linear(self.input_dims, vocab_size)
        # self.lm_head_4 = nn.Linear(self.input_dims, vocab_size)

        self.output_head_mask_1 = torch.zeros(
            size=(vocab_size,), requires_grad=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_head_mask_1 = get_output_head_mask(
            self.output_head_mask_1, 0, 32
        )  # ins label category

        self.output_head_mask_2 = torch.zeros(
            size=(vocab_size,), requires_grad=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_head_mask_2 = get_output_head_mask(
            self.output_head_mask_2, 33, 129
        )  # onset time category

        self.output_head_mask_3 = torch.zeros(
            size=(vocab_size,), requires_grad=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_head_mask_3 = get_output_head_mask(
            self.output_head_mask_3, 130, 298
        )  # pitch and chord category

        self.output_head_mask_4 = torch.zeros(
            size=(vocab_size,), requires_grad=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_head_mask_4 = get_output_head_mask(
            self.output_head_mask_4, 299, 306
        )  # duration category

        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight
            # self.lm_head_1.weight = self.word_embedding.weight
            # self.lm_head_2.weight = self.word_embedding.weight
            # self.lm_head_3.weight = self.word_embedding.weight
            # self.lm_head_4.weight = self.word_embedding.weight

        # Layer Normalization on Embedding
        # if learned_normalize_embed:
        #     self.emb_ln_weight = nn.parameter.Parameter(data=th.randn(size=(self.hidden_t_dim,)), requires_grad=True)
        #     self.emb_ln_bias = nn.parameter.Parameter(data=th.randn(size=(self.hidden_t_dim,)), requires_grad=True)

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    # def get_logits(self, hidden_repr):
    #     if self.logits_mode == 1:
    #         return self.lm_head(hidden_repr)
    # elif self.logits_mode == 2: # standard cosine similarity
    #     text_emb = hidden_repr
    #     emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
    #     text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
    #     arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
    #     dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
    #                                                              text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
    #     scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
    #                                                        hidden_repr.size(1)) # vocab, bsz*seqlen
    #     scores = -scores.permute(1, 2, 0).contiguous()
    #     return scores
    # else:
    #     raise NotImplementedError

    # def get_logits(self, hidden_repr):
    #
    #     lm_src = self.lm_head_1(hidden_repr[:, 0:11, :])
    #
    #     # Multiply by zero on the irrelevant categories, Multiply one for the target category.
    #     lm_5 = self.lm_head_1(hidden_repr[:, 11:508:4, :]) * self.output_head_mask_1  # (40,125,307)
    #     lm_6 = self.lm_head_2(hidden_repr[:, 12:509:4, :]) * self.output_head_mask_2  # (40,125,307)
    #     lm_7 = self.lm_head_3(hidden_repr[:, 13:510:4, :]) * self.output_head_mask_3  # (40,125,307)
    #     lm_8 = self.lm_head_4(hidden_repr[:, 14:511:4, :]) * self.output_head_mask_4  # (40,125,307)
    #
    #     # print('lm_8[-1,-1]:', lm_8[-1, -1])
    #     # print('lm_8[-1,0]:', lm_8[-1, 0])
    #
    #     lm_end = self.lm_head_1(hidden_repr[:, 511, :]).unsqueeze(1)
    #
    #     lm_stack_trg = torch.stack((lm_5, lm_6, lm_7, lm_8), dim=2)
    #     lm_trg = lm_stack_trg.view([lm_stack_trg.shape[0], -1, lm_stack_trg.shape[-1]])
    #
    #     lm_out = torch.cat((lm_src, lm_trg, lm_end), dim=1)
    #
    #     return lm_out

    def get_logits(self, hidden_repr):

        lm_src = self.lm_head(hidden_repr[:, 0:11, :])

        # Multiply by zero on the irrelevant categories, Multiply one for the target category.
        lm_5 = self.lm_head(hidden_repr[:, 11:508:4, :]) * self.output_head_mask_1  # (40,125,307)
        lm_6 = self.lm_head(hidden_repr[:, 12:509:4, :]) * self.output_head_mask_2  # (40,125,307)
        lm_7 = self.lm_head(hidden_repr[:, 13:510:4, :]) * self.output_head_mask_3  # (40,125,307)
        lm_8 = self.lm_head(hidden_repr[:, 14:511:4, :]) * self.output_head_mask_4  # (40,125,307)

        # print('lm_8[-1,-1]:', lm_8[-1, -1])
        # print('lm_8[-1,0]:', lm_8[-1, 0])

        lm_end = self.lm_head(hidden_repr[:, 511, :]).unsqueeze(1)

        lm_stack_trg = torch.stack((lm_5, lm_6, lm_7, lm_8), dim=2)
        lm_trg = lm_stack_trg.view([lm_stack_trg.shape[0], -1, lm_stack_trg.shape[-1]])

        lm_out = torch.cat((lm_src, lm_trg, lm_end), dim=1)

        return lm_out

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h

    # def layer_normalization_on_embeddings(self, in_emb):
    #     # layer normalization on emb
    #     emb_m = th.mean(in_emb, dim=2, keepdim=True)  # （64，200，1）
    #     emb_v = th.var(in_emb, dim=2, keepdim=True)  # （64，200，1）
    #     emb_weight = self.emb_ln_weight.to(in_emb.device)  # (256)
    #     emb_bias = self.emb_ln_bias.to(in_emb.device)  # (256)
    #     emb_ln = (th.mul(
    #         (in_emb - emb_m) / th.sqrt(emb_v + th.tensor(0.00001)), emb_weight
    #     ) + emb_bias).to(in_emb.device)
    #     return emb_ln
