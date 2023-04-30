"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.utils.misc import sequence_mask

import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn.modules import Module
from typing import Optional, Tuple, Union

from onmt.encoders.activation import Hopfield


class HopfieldEncoderLayer(Module):
    """
    Module with underlying Hopfield association to be used as an encoder in transformer-like architectures.
    """

    def __init__(self,
                 hopfield_association: Hopfield,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = r'relu',
                 ):
        """
        Initialise a new instance of a Hopfield association-based encoder module.
        :param hopfield_association: instance of Hopfield association module
        :param dim_feedforward: depth of the linear projections applied internally
        :param activation: activation to be applied on the result of the internal linear projections
        :param dropout: dropout probability to be applied internally
        """
        super(HopfieldEncoderLayer, self).__init__()
        self.hopfield_association = deepcopy(hopfield_association)

        self.linear_residual = nn.Linear(self.hopfield_association.state_pattern_dim, dim_feedforward)
        self.dropout_residual = nn.Dropout(dropout)
        self.linear_output = nn.Linear(dim_feedforward, self.hopfield_association.state_pattern_dim)

        self.norm_residual = nn.LayerNorm(self.hopfield_association.state_pattern_dim)
        self.norm_output = nn.LayerNorm(self.hopfield_association.state_pattern_dim)
        self.dropout_hopfield_association = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

        self.activation_residual = getattr(torch, activation, None)
        assert self.activation_residual is not None, r'invalid activation function supplied.'
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset parameters, including Hopfield association.
        :return: None
        """
        for module in (self.hopfield_association, self.linear_residual,
                       self.linear_output, self.norm_residual, self.norm_output):
            if hasattr(module, r'reset_parameters'):
                module.reset_parameters()

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Hopfield encoding on specified data.
        :param src: data to be processed by Hopfield encoder module
        :param src_mask: mask to be applied on association matrix
        :param src_key_padding_mask: mask to be applied on stored patterns
        :return: Hopfield-encoded input data
        """

        # print('src', src.size(), src_key_padding_mask.size())
        if src_key_padding_mask.dim() == 3 and src_key_padding_mask.size(1) == 1:
            src_key_padding_mask = src_key_padding_mask.squeeze(1)

        data_associated = self.hopfield_association(
            input=src, stored_pattern_padding_mask=src_key_padding_mask, association_mask=src_mask)
        src = src + self.dropout_hopfield_association(input=data_associated)
        src = self.norm_residual(input=src)

        result_residual_inner = self.activation_residual(input=self.linear_residual(input=src))
        data_associated = self.linear_output(input=self.dropout_residual(input=result_residual_inner))
        src = src + self.dropout_output(input=data_associated)

        return self.norm_output(input=src)

    def get_association_matrix(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]]) -> Tensor:
        """
        Fetch Hopfield association matrix gathered by passing through the specified data.
        :param input: data to be passed through the Hopfield association
        :return: association matrix as computed by the Hopfield core module
        """
        return self.hopfield_association.get_association_matrix(input=input)

    @property
    def batch_first(self) -> int:
        return self.hopfield_association.batch_first

    @property
    def input_size(self) -> int:
        return self.hopfield_association.input_size

    @property
    def output_size(self) -> int:
        return self.linear_output.out_features




class HopfieldTransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * enc_out ``(batch_size, src_len, model_dim)``
        * encoder final state: None in the case of Transformer
        * src_len ``(batch_size)``
    """

    def __init__(self, num_layers, d_model, heads, dropout,
                 attention_dropout, embeddings, scaling=0.04,sparse='softmax'
                 ):
        super(HopfieldTransformerEncoder, self).__init__()

        self.embeddings = embeddings
        layer_list = [HopfieldEncoderLayer(
            Hopfield(input_size=embeddings.embedding_size,
                     hidden_size=d_model//heads,
                     output_size=d_model,
                     scaling=scaling,
                     num_heads=heads,
                     sparse=sparse,
                     update_steps_max=1,
                    normalize_hopfield_space=False,
                    normalize_hopfield_space_affine=False,
                    normalize_pattern_projection=False,
                    normalize_pattern_projection_affine=False, 
                    normalize_state_pattern=False, 
                    normalize_state_pattern_affine=False, 
                    normalize_stored_pattern=False, 
                    normalize_stored_pattern_affine=False,
                     ), 
            dropout=dropout
        )]
        
        for i in range(num_layers-1):
            layer_list.append(HopfieldEncoderLayer(
                hopfield_association=Hopfield(
                    input_size=d_model,
                    scaling=0.1, 
                    num_heads=heads,
                    sparse=sparse,
                    update_steps_max=1,
                    normalize_hopfield_space=False,
                    normalize_hopfield_space_affine=False,
                    normalize_pattern_projection=False,
                    normalize_pattern_projection_affine=False, 
                    normalize_state_pattern=False, 
                    normalize_state_pattern_affine=False, 
                    normalize_stored_pattern=False, 
                    normalize_stored_pattern_affine=False,
                    ), 
                dropout=dropout
            ))

        self.transformer = nn.ModuleList(layer_list)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_hid_size,
            opt.heads,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.scaling,
            sparse=opt.sparse
            )

    def forward(self, src, src_len=None):
        """See :func:`EncoderBase.forward()`"""
        enc_out = self.embeddings(src)
        mask = ~sequence_mask(src_len)
        # mask = mask.unsqueeze(1)
        # mask = mask.expand(-1, -1, mask.size(3), -1)
        # mask is now (batch x 1 x slen x slen)
        # 1 to be expanded to number of heads in MHA
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            enc_out = layer(src=enc_out, src_key_padding_mask=mask)
        enc_out = self.layer_norm(enc_out)

        if torch.isnan(enc_out).any():
            print('encode')
            raise Exception

        return enc_out, None, src_len

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
