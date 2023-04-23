import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn.modules import Module
from typing import Optional, Tuple, Union

from onmt.decoders.activation import Hopfield
from onmt.decoders.decoder import DecoderBase
from onmt.utils.misc import sequence_mask
from onmt.modules.rmsnorm import RMSNorm





class HopfieldDecoderLayer(Module):
    """
    Module with underlying Hopfield associations to be used as a decoder in transformer-like architectures.
    """

    def __init__(self,
                 hopfield_association_self: Hopfield,
                 hopfield_association_cross: Hopfield,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = r'relu',
                 ):
        """
        Initialise a new instance of a Hopfield association-based encoder module.
        :param hopfield_association_self: instance of Hopfield self-association module
        :param hopfield_association_cross: instance of Hopfield cross-association module
        :param dim_feedforward: depth of the linear projections applied internally
        :param dropout: dropout probability to be applied internally
        :param activation: activation to be applied on the result of the internal linear projections
        """
        super(HopfieldDecoderLayer, self).__init__()
        self.hopfield_association_self = deepcopy(hopfield_association_self)
        self.hopfield_association_cross = deepcopy(hopfield_association_cross)

        self.linear_residual = nn.Linear(self.hopfield_association_self.state_pattern_dim, dim_feedforward)
        self.dropout_residual = nn.Dropout(dropout)
        self.linear_output = nn.Linear(dim_feedforward, self.hopfield_association_self.state_pattern_dim)

        self.norm_residual_self = nn.LayerNorm(self.hopfield_association_self.state_pattern_dim)
        self.norm_residual_cross = nn.LayerNorm(self.hopfield_association_self.state_pattern_dim)
        self.norm_output = nn.LayerNorm(self.hopfield_association_self.state_pattern_dim)
        self.dropout_hopfield_association_self = nn.Dropout(dropout)
        self.dropout_hopfield_association_cross = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

        self.activation_residual = getattr(torch, activation, None)
        assert self.activation_residual is not None, r'invalid activation function supplied.'
        self.reset_parameters()

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if True:  # apply future_mask, result mask in (B, T, T)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        # else:  # only mask padding, result mask in (B, 1, T)
            # dec_mask = tgt_pad_mask
        return dec_mask # (B*heads, T, S)

    def reset_parameters(self) -> None:
        """
        Reset parameters, including Hopfield association.
        :return: None
        """
        for module in (self.hopfield_association_self, self.hopfield_association_cross,
                       self.linear_residual, self.linear_output, self.norm_residual_self,
                       self.norm_residual_cross, self.norm_output):
            if hasattr(module, r'reset_parameters'):
                module.reset_parameters()

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Hopfield decoding on specified data.
        :param tgt: data to be processed by Hopfield decoder module (self-association)
        :param memory: data to be processed by Hopfield encoder module (cross-association)
        :param tgt_mask: mask to be applied on self-association matrix
        :param memory_mask: mask to be applied on cross-association matrix
        :param tgt_key_padding_mask: mask to be applied on stored patterns
        :param memory_key_padding_mask: mask to be applied on state patterns as well as pattern projection
        :return: Hopfield-decoded input
        """


        # print(tgt_key_padding_mask.shape, tgt_key_padding_mask[0][0])
        tgt_mask = self._compute_dec_mask(tgt_key_padding_mask, future=False)
        if tgt_key_padding_mask.dim() == 3 and tgt_key_padding_mask.size(1) == 1:
            tgt_key_padding_mask = tgt_key_padding_mask.squeeze(1)

        head_num = self.hopfield_association_self.num_heads
        if tgt_mask.size(0) == tgt.size(0):
            tgt_mask = tgt_mask.repeat(head_num, 1, 1)
        
        # print(tgt_mask[0, 0])
        # raise Exception
        data_associated, self_attn_weight = self.hopfield_association_self(
            input=tgt, stored_pattern_padding_mask=tgt_key_padding_mask,
            association_mask=tgt_mask)

        tgt = tgt + self.dropout_hopfield_association_self(input=data_associated)
        tgt = self.norm_residual_self(input=tgt)

        data_associated, cross_attn_weight = self.hopfield_association_cross(
            input=(memory, tgt, memory), stored_pattern_padding_mask=memory_key_padding_mask,
            association_mask=memory_mask)
        tgt = tgt + self.dropout_hopfield_association_cross(input=data_associated)
        tgt = self.norm_residual_cross(input=tgt)

        result_residual_inner = self.activation_residual(input=self.linear_residual(input=tgt))
        data_associated = self.linear_output(input=self.dropout_residual(input=result_residual_inner))
        tgt = tgt + self.dropout_output(input=data_associated)
        return self.norm_output(input=tgt), cross_attn_weight

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout

    def get_association_matrix_self(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        """
        Fetch Hopfield self-association matrix gathered by passing through the specified data.
        :param input: data to be passed through the Hopfield association
        :return: association matrix as computed by the Hopfield core module
        """
        return self.hopfield_association_self.get_association_matrix(input=input)

    def get_association_matrix_cross(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        """
        Fetch Hopfield cross-association matrix gathered by passing through the specified data.
        :param input: data to be passed through the Hopfield association
        :return: association matrix as computed by the Hopfield core module
        """
        return self.hopfield_association_cross.get_association_matrix(input=input)

    @property
    def batch_first(self) -> int:
        return self.hopfield_association_self.batch_first

    @property
    def input_size(self) -> int:
        return self.hopfield_association_self.input_size

    @property
    def output_size(self) -> int:
        return self.linear_output_self.out_features
    



class HopfieldTransformerDecoderBase(DecoderBase):
    def __init__(self, d_model, copy_attn, embeddings, alignment_layer,
                 layer_norm='standard'):
        super(HopfieldTransformerDecoderBase, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        if layer_norm == 'standard':
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        elif layer_norm == 'rms':
            self.layer_norm = RMSNorm(d_model, eps=1e-6)
        else:
            raise ValueError(f'{layer_norm} layer norm type is not supported')

        self.alignment_layer = alignment_layer

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_hid_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0]
            if type(opt.attention_dropout) is list
            else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads,
            layer_norm=opt.layer_norm,
            sparse=opt.sparse
        )

    def init_state(self, src, enc_out, enc_final_hs):
        """Initialize decoder state."""
        self.state["src"] = src

    def map_state(self, fn):

        if self.state["src"] is not None:
            self.state["src"] = fn(self.state["src"], 0)
        for layer in self.transformer_layers:
            if hasattr(layer, 'context_attn'):
                if layer.context_attn.layer_cache[1]['keys'].numel() != 0:
                    x = fn(layer.context_attn.layer_cache[1]['keys'], 0)
                    y = fn(layer.context_attn.layer_cache[1]['values'], 0)
                    layer.context_attn.layer_cache = True, {'keys': x,
                                                            'values': y}
            if isinstance(layer.self_attn, AverageAttention):
                if layer.self_attn.layer_cache[1]['prev_g'].numel() != 0:
                    x = fn(layer.self_attn.layer_cache[1]['prev_g'], 0)
                    layer.self_attn.layer_cache = True, {'prev_g': x}
            else:
                if layer.self_attn.layer_cache[1]['keys'].numel() != 0:
                    x = fn(layer.self_attn.layer_cache[1]['keys'], 0)
                    y = fn(layer.self_attn.layer_cache[1]['values'], 0)
                    layer.self_attn.layer_cache = True, {'keys': x,
                                                         'values': y}

    def detach_state(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)




class HopfieldTransformerDecoder(HopfieldTransformerDecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
        add_qkvbias (bool): whether to add bias to the Key/Value nn.Linear
        layer_norm (string): type of layer normalization standard/rms
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        aan_useffn,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
        layer_norm='standard',
        scaling=0.04,
        sparse='softmax'
    ):
        super(HopfieldTransformerDecoder, self).__init__(
            d_model, copy_attn, embeddings, alignment_layer
        )

        layer_list = [HopfieldDecoderLayer(
            Hopfield(input_size=embeddings.embedding_size,
                     hidden_size=d_model,
                     output_size=d_model,
                     scaling=scaling,
                     num_heads=heads,
                     sparse=sparse,
                     update_steps_max=1),
            Hopfield(input_size=embeddings.embedding_size,
                     hidden_size=d_model,
                     output_size=d_model,
                     scaling=scaling,
                     num_heads=heads,
                     sparse=sparse,
                     update_steps_max=1),
            dropout=dropout,
        )]

        for i in range(num_layers-1):
            layer_list.append(
                HopfieldDecoderLayer(
                    Hopfield(input_size=d_model,
                            hidden_size=d_model,
                            output_size=d_model,
                            scaling=scaling,
                            dropout=dropout,
                            num_heads=heads,
                            sparse=sparse,
                            update_steps_max=5),
                    Hopfield(input_size=d_model,
                            hidden_size=d_model,
                            output_size=d_model,
                            scaling=scaling,
                            dropout=dropout,
                            num_heads=heads,
                            sparse=sparse,
                            update_steps_max=5),
            ))

        self.transformer_layers = nn.ModuleList(layer_list)

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, enc_out=None, step=None, **kwargs):
        """
        Decode, possibly stepwise.
        when training step is always None, when decoding, step increases
        tgt (Tensor): batch x tlen x feats
        enc_out (Tensor): encoder output (batch x slen x model_dim)
        """
        if enc_out is None:
            enc_out = self.embeddings(tgt)
        # if step == 0:
        #     self._init_cache(enc_out)
        # elif step is None:
        #     for layer in self.transformer_layers:
        #         layer.self_attn.layer_cache = (
        #             False, {'keys': torch.tensor([]),
        #                     'values': torch.tensor([])})
        #         layer.context_attn.layer_cache = (
        #             False, {'keys': torch.tensor([]),
        #                     'values': torch.tensor([])})

        tgt_words = tgt[:, :, 0]

        emb = self.embeddings(tgt, step=step)
        dec_out = emb
        assert emb.dim() == 3  # len x batch x embedding_dim

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["src_len"]
        src_max_len = self.state["src"].shape[1]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len)  # [B x slen]
        src_pad_mask = src_pad_mask  # [B x slen]
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop("with_align", False)
        attn_aligns = []

        for layer in self.transformer_layers:
            dec_out, attn = layer(
                tgt=dec_out,
                memory=enc_out,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            # if attn_align is not None:
            #     attn_aligns.append(attn_align)
        if torch.isnan(dec_out).any():
            raise Exception('decoder nan problem')
        dec_out = self.layer_norm(dec_out)

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_out, attns

    def _init_cache(self, enc_out):

        batch_size = enc_out.size(0)
        depth = enc_out.size(-1)

        for layer in self.transformer_layers:
            # first value set to True triggered by the beginning of decoding
            # layer_cache becomes active in the MultiHeadedAttention fwd
            layer.context_attn.layer_cache = (
                True,
                {'keys': torch.tensor([], device=enc_out.device),
                 'values': torch.tensor([], device=enc_out.device)}
                )
            if isinstance(layer.self_attn, AverageAttention):
                layer.self_attn.layer_cache = True, {'prev_g': torch.zeros(
                     (batch_size, 1, depth), device=enc_out.device
                ).to(enc_out.dtype)}
            else:
                layer.self_attn.layer_cache = (
                    True,
                    {'keys': torch.tensor([], device=enc_out.device),
                     'values': torch.tensor([], device=enc_out.device)}
                    )
