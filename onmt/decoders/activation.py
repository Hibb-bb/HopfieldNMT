
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Linear, Module, Parameter
from typing import Optional

from onmt.decoders.functional import hopfield_core_forward
from typing import Optional, Tuple, Union

try:
    from torch.nn.modules.linear import _LinearWithBias
except ImportError:
    _LinearWithBias = None

class Hopfield(Module):
    """
    Module with underlying Hopfield association.
    """

    def __init__(self,
                 input_size: Optional[int] = None,
                 hidden_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 pattern_size: Optional[int] = None,
                 num_heads: int = 1,
                 scaling: Optional[Union[float, Tensor]] = None,
                 update_steps_max: Optional[Union[int, Tensor]] = 3,
                 update_steps_eps: Union[float, Tensor] = 1e-4,

                 normalize_stored_pattern: bool = True,
                 normalize_stored_pattern_affine: bool = True,
                 normalize_stored_pattern_eps: float = 1e-5,
                 normalize_state_pattern: bool = False,
                 normalize_state_pattern_affine: bool = False,
                 normalize_state_pattern_eps: float = 1e-5,
                 normalize_pattern_projection: bool = True,
                 normalize_pattern_projection_affine: bool = True,
                 normalize_pattern_projection_eps: float = 1e-5,
                 normalize_hopfield_space: bool = False,
                 normalize_hopfield_space_affine: bool = False,
                 normalize_hopfield_space_eps: float = 1e-5,
                 stored_pattern_as_static: bool = False,
                 state_pattern_as_static: bool = False,
                 pattern_projection_as_static: bool = False,
                 pattern_projection_as_connected: bool = False,
                 stored_pattern_size: Optional[int] = None,
                 pattern_projection_size: Optional[int] = None,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 association_activation: Optional[str] = None,
                 c: float = 0.0,
                 input_bias: bool = True,
                 concat_bias_pattern: bool = False,
                 add_zero_association: bool = False,
                 disable_out_projection: bool = False,
                 sparse: str = 'softmax'
                 ):
        """
        Initialise new instance of a Hopfield module.
        :param input_size: depth of the input (state pattern)
        :param hidden_size: depth of the association space
        :param output_size: depth of the output projection
        :param pattern_size: depth of patterns to be selected
        :param num_heads: amount of parallel association heads
        :param scaling: scaling of association heads, often represented as beta (one entry per head)
        :param update_steps_max: maximum count of association update steps (None equals to infinity)
        :param update_steps_eps: minimum difference threshold between two consecutive association update steps
        :param normalize_stored_pattern: apply normalization on stored patterns
        :param normalize_stored_pattern_affine: additionally enable affine normalization of stored patterns
        :param normalize_stored_pattern_eps: offset of the denominator for numerical stability
        :param normalize_state_pattern: apply normalization on state patterns
        :param normalize_state_pattern_affine: additionally enable affine normalization of state patterns
        :param normalize_state_pattern_eps: offset of the denominator for numerical stability
        :param normalize_pattern_projection: apply normalization on the pattern projection
        :param normalize_pattern_projection_affine: additionally enable affine normalization of pattern projection
        :param normalize_pattern_projection_eps: offset of the denominator for numerical stability
        :param normalize_hopfield_space: enable normalization of patterns in the Hopfield space
        :param normalize_hopfield_space_affine: additionally enable affine normalization of patterns in Hopfield space
        :param normalize_hopfield_space_eps: offset of the denominator for numerical stability
        :param stored_pattern_as_static: interpret specified stored patterns as being static
        :param state_pattern_as_static: interpret specified state patterns as being static
        :param pattern_projection_as_static: interpret specified pattern projections as being static
        :param pattern_projection_as_connected: connect pattern projection with stored pattern
        :param stored_pattern_size: depth of input (stored pattern)
        :param pattern_projection_size: depth of input (pattern projection)
        :param batch_first: flag for specifying if the first dimension of data fed to "forward" reflects the batch size
        :param association_activation: additional activation to be applied on the result of the Hopfield association
        :param dropout: dropout probability applied on the association matrix
        :param input_bias: bias to be added to input (state and stored pattern as well as pattern projection)
        :param concat_bias_pattern: bias to be concatenated to stored pattern as well as pattern projection
        :param add_zero_association: add a new batch of zeros to stored pattern as well as pattern projection
        :param disable_out_projection: disable output projection
        """
        super(Hopfield, self).__init__()
        assert type(batch_first) == bool, f'"batch_first" needs to be a boolean, not {type(batch_first)}.'
        assert (association_activation is None) or (type(association_activation) == str)

        self.sparse = sparse

        self.num_heads = num_heads
        # Initialise Hopfield association module.
        self.association_core = HopfieldCore(
            embed_dim=input_size, num_heads=num_heads, dropout=dropout, bias=input_bias,
            add_bias_kv=concat_bias_pattern, add_zero_attn=add_zero_association, kdim=stored_pattern_size,
            vdim=pattern_projection_size, head_dim=hidden_size, pattern_dim=pattern_size, out_dim=output_size,
            disable_out_projection=disable_out_projection, key_as_static=stored_pattern_as_static,
            query_as_static=state_pattern_as_static, value_as_static=pattern_projection_as_static,
            value_as_connected=pattern_projection_as_connected, normalize_pattern=normalize_hopfield_space,
            normalize_pattern_affine=normalize_hopfield_space_affine,
            normalize_pattern_eps=normalize_hopfield_space_eps, sparse=sparse)
        self.association_activation = None
        if association_activation is not None:
            self.association_activation = getattr(torch, association_activation, None)

        # Initialise stored pattern normalization.
        self.norm_stored_pattern = None
        if normalize_stored_pattern_affine:
            assert normalize_stored_pattern, "affine normalization without normalization has no effect."
        if normalize_stored_pattern:
            normalized_shape = input_size if stored_pattern_size is None else stored_pattern_size
            assert normalized_shape is not None, "stored pattern size required for setting up normalisation"
            self.norm_stored_pattern = nn.LayerNorm(
                normalized_shape=normalized_shape, elementwise_affine=normalize_stored_pattern_affine,
                eps=normalize_stored_pattern_eps)

        # Initialise state pattern normalization.
        self.norm_state_pattern = None
        if normalize_state_pattern_affine:
            assert normalize_state_pattern, "affine normalization without normalization has no effect."
        if normalize_state_pattern:
            assert input_size is not None, "input size required for setting up normalisation"
            self.norm_state_pattern = nn.LayerNorm(
                normalized_shape=input_size, elementwise_affine=normalize_state_pattern_affine,
                eps=normalize_state_pattern_eps)

        # Initialise pattern projection normalization.
        self.norm_pattern_projection = None
        if normalize_pattern_projection_affine:
            assert normalize_pattern_projection, "affine normalization without normalization has no effect."
        if normalize_pattern_projection:
            normalized_shape = input_size if pattern_projection_size is None else pattern_projection_size
            assert normalized_shape is not None, "pattern projection size required for setting up normalisation"
            self.norm_pattern_projection = nn.LayerNorm(
                normalized_shape=normalized_shape, elementwise_affine=normalize_pattern_projection_affine,
                eps=normalize_pattern_projection_eps)

        # Initialise remaining auxiliary properties.
        if self.association_core.static_execution:
            self.__scaling = 1.0 if scaling is None else scaling
        else:
            assert self.association_core.head_dim > 0, f'invalid hidden dimension encountered.'
            self.__scaling = (1.0 / sqrt(self.association_core.head_dim)) if scaling is None else scaling
        self.__batch_first = batch_first
        self.__update_steps_max = update_steps_max
        self.__update_steps_eps = update_steps_eps
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset Hopfield association.
        :return: None
        """
        for module in (self.association_core, self.norm_stored_pattern,
                       self.norm_state_pattern, self.norm_pattern_projection):
            if hasattr(module, r'reset_parameters'):
                module.reset_parameters()

    def _maybe_transpose(self, *args: Tuple[Tensor, ...]) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Eventually transpose specified data.
        :param args: tensors to eventually transpose (dependent on the state of "batch_first")
        :return: eventually transposed tensors
        """

        transposed_result = tuple(_.transpose(0, 1) for _ in args) if self.__batch_first else args
        return transposed_result[0] if len(transposed_result) == 1 else transposed_result

    def _associate(self, data: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
                   return_raw_associations: bool = False, return_projected_patterns: bool = False,
                   stored_pattern_padding_mask: Optional[Tensor] = None,
                   association_mask: Optional[Tensor] = None) -> Tuple[Optional[Tensor], ...]:
        """
        Apply Hopfield association module on specified data.
        :param data: data to be processed by Hopfield core module
        :param return_raw_associations: return raw association (softmax) values, unmodified
        :param return_projected_patterns: return pattern projection values, unmodified
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: Hopfield-processed input data
        """
        assert (type(data) == Tensor) or ((type(data) == tuple) and (len(data) == 3)), \
            r'either one tensor to be used as "stored pattern", "state pattern" and' \
            r' "pattern_projection" must be provided, or three separate ones.'
        if type(data) == Tensor:
            stored_pattern, state_pattern, pattern_projection = data, data, data
        else:
            stored_pattern, state_pattern, pattern_projection = data

        # Optionally transpose data.
        stored_pattern, state_pattern, pattern_projection = self._maybe_transpose(
            stored_pattern, state_pattern, pattern_projection)

        # Optionally apply stored pattern normalization.
        if self.norm_stored_pattern is not None:
            stored_pattern = self.norm_stored_pattern(input=stored_pattern.reshape(
                shape=(-1, stored_pattern.shape[2]))).reshape(shape=stored_pattern.shape)

        # Optionally apply state pattern normalization.
        if self.norm_state_pattern is not None:
            state_pattern = self.norm_state_pattern(input=state_pattern.reshape(
                shape=(-1, state_pattern.shape[2]))).reshape(shape=state_pattern.shape)

        # Optionally apply pattern projection normalization.
        if self.norm_pattern_projection is not None:
            pattern_projection = self.norm_pattern_projection(input=pattern_projection.reshape(
                shape=(-1, pattern_projection.shape[2]))).reshape(shape=pattern_projection.shape)

        # Apply Hopfield association and optional activation function.
        return self.association_core(
            query=state_pattern, key=stored_pattern, value=pattern_projection,
            key_padding_mask=stored_pattern_padding_mask, need_weights=True, attn_mask=association_mask,
            scaling=self.__scaling, update_steps_max=self.__update_steps_max, update_steps_eps=self.__update_steps_eps,
            return_raw_associations=return_raw_associations, return_pattern_projections=return_projected_patterns)

    def forward(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
                stored_pattern_padding_mask: Optional[Tensor] = None,
                association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply Hopfield association on specified data.
        :param input: data to be processed by Hopfield association module
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: Hopfield-processed input data
        """

        output, attn_weight, a, b = self._associate(
            data=input, return_raw_associations=False,
            stored_pattern_padding_mask=stored_pattern_padding_mask,
            association_mask=association_mask)

        association_output = self._maybe_transpose(output)
        if self.association_activation is not None:

            association_output = self.association_activation(association_output)
        return association_output, attn_weight

    def get_association_matrix(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
                               stored_pattern_padding_mask: Optional[Tensor] = None,
                               association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Fetch Hopfield association matrix gathered by passing through the specified data.
        :param input: data to be passed through the Hopfield association
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: association matrix as computed by the Hopfield core module
        """
        with torch.no_grad():
            return self._associate(
                data=input, return_raw_associations=True,
                stored_pattern_padding_mask=stored_pattern_padding_mask,
                association_mask=association_mask)[2]

    def get_projected_pattern_matrix(self, input: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
                                     stored_pattern_padding_mask: Optional[Tensor] = None,
                                     association_mask: Optional[Tensor] = None) -> Tensor:
        """
        Fetch Hopfield projected pattern matrix gathered by passing through the specified data.
        :param input: data to be passed through the Hopfield association
        :param stored_pattern_padding_mask: mask to be applied on stored patterns
        :param association_mask: mask to be applied on inner association matrix
        :return: pattern projection matrix as computed by the Hopfield core module
        """
        with torch.no_grad():
            return self._associate(
                data=input, return_projected_patterns=True,
                stored_pattern_padding_mask=stored_pattern_padding_mask,
                association_mask=association_mask)[3]

    @property
    def batch_first(self) -> bool:
        return self.__batch_first

    @property
    def scaling(self) -> Union[float, Tensor]:
        return self.__scaling.clone() if type(self.__scaling) == Tensor else self.__scaling

    @property
    def stored_pattern_dim(self) -> Optional[int]:
        return self.association_core.kdim

    @property
    def state_pattern_dim(self) -> Optional[int]:
        return self.association_core.embed_dim

    @property
    def pattern_projection_dim(self) -> Optional[int]:
        return self.association_core.vdim

    @property
    def input_size(self) -> Optional[int]:
        return self.state_pattern_dim

    @property
    def hidden_size(self) -> Optional[int]:
        return self.association_core.head_dim

    @property
    def output_size(self) -> Optional[int]:
        return self.association_core.out_dim

    @property
    def pattern_size(self) -> Optional[int]:
        return self.association_core.pattern_dim

    @property
    def update_steps_max(self) -> Optional[Union[int, Tensor]]:
        return self.__update_steps_max.clone() if type(self.__update_steps_max) == Tensor else self.__update_steps_max

    @property
    def update_steps_eps(self) -> Optional[Union[float, Tensor]]:
        return self.__update_steps_eps.clone() if type(self.__update_steps_eps) == Tensor else self.__update_steps_eps

    @property
    def stored_pattern_as_static(self) -> bool:
        return self.association_core.key_as_static

    @property
    def state_pattern_as_static(self) -> bool:
        return self.association_core.query_as_static

    @property
    def pattern_projection_as_static(self) -> bool:
        return self.association_core.value_as_static

    @property
    def normalize_stored_pattern(self) -> bool:
        return self.norm_stored_pattern is not None

    @property
    def normalize_stored_pattern_affine(self) -> bool:
        return self.normalize_stored_pattern and self.norm_stored_pattern.elementwise_affine

    @property
    def normalize_state_pattern(self) -> bool:
        return self.norm_state_pattern is not None

    @property
    def normalize_state_pattern_affine(self) -> bool:
        return self.normalize_state_pattern and self.norm_state_pattern.elementwise_affine

    @property
    def normalize_pattern_projection(self) -> bool:
        return self.norm_pattern_projection is not None

    @property
    def normalize_pattern_projection_affine(self) -> bool:
        return self.normalize_pattern_projection and self.norm_pattern_projection.elementwise_affine

    @property
    def normalize_hopfield_space(self) -> bool:
        return self.hopfield.normalize_hopfield_space

    @property
    def normalize_hopfield_space_affine(self) -> bool:
        return self.hopfield.normalize_hopfield_space_affine



class HopfieldCore(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See references: "Hopfield Networks is All You Need" and
                    "Attention Is All You Need" (on which this implementation is partly based on).
    .. math::
        \text{HopfieldHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> hopfield_attn = HopfieldCore(embed_dim, num_heads)
        >>> attn_output, attn_output_weights, attn_matrix = hopfield_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }

    def __init__(self,
                 embed_dim=None,                  # type: Optional[int]
                 num_heads=1,                     # type: int
                 dropout=0.0,                     # type: float
                 bias=True,                       # type: bool
                 add_bias_kv=False,               # type: bool
                 add_zero_attn=False,             # type: bool
                 kdim=None,                       # type: Optional[int]
                 vdim=None,                       # type: Optional[int]

                 head_dim=None,                   # type: Optional[int]
                 pattern_dim=None,                # type: Optional[int]
                 out_dim=None,                    # type: Optional[int]
                 disable_out_projection=False,    # type: bool
                 key_as_static=True,             # type: bool
                 query_as_static=True,           # type: bool
                 value_as_static=True,           # type: bool
                 value_as_connected=False,        # type: bool
                 normalize_pattern=False,         # type: bool
                 normalize_pattern_affine=False,  # type: bool
                 normalize_pattern_eps=1e-5,      # type: float
                 sparse='softmax'
                 ):
        super(HopfieldCore, self).__init__()

        self.sparse = sparse
        assert (type(key_as_static) == bool) and (type(query_as_static) == bool) and (type(value_as_static) == bool)
        self.key_as_static, self.query_as_static, self.value_as_static = key_as_static, query_as_static, value_as_static
        num_non_static = 3 - (self.key_as_static + self.query_as_static + self.value_as_static)
        assert 0 <= num_non_static < 4

        self.value_as_connected = value_as_connected
        self.normalize_pattern, self.normalize_pattern_affine = normalize_pattern, normalize_pattern_affine
        self.normalize_pattern_eps = normalize_pattern_eps
        self.disable_out_projection = disable_out_projection

        # In case of a static-only executions, check corresponding projections and normalizations.
        self.static_execution = self._check_execution_mode()
        if self.static_execution:
            embed_dim, kdim, vdim = None, None, None
        if embed_dim is None:
            assert self.static_execution, r'static-only execution requires all projections to be deactivated.'

        # Check and set all other properties, conditioned on <static_execution>.
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = all((
            self.kdim == embed_dim, self.vdim == embed_dim, pattern_dim is None, not self.value_as_connected))
        assert (not self.value_as_connected) or (self.kdim == self.vdim), r'key and value need to be of same dimension.'

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = None
        self.pattern_dim = pattern_dim
        self.virtual_hopfield_dim = None
        self.virtual_pattern_dim = None
        if not self.static_execution:
            if head_dim is None:
                self.head_dim = embed_dim // num_heads
                assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads."
            else:
                assert head_dim > 0, "dimension of the association space has to be positive."
                self.head_dim = head_dim
            if self.pattern_dim is None:
                self.pattern_dim = self.head_dim
            self.virtual_hopfield_dim = self.num_heads * self.head_dim
            self.virtual_pattern_dim = self.num_heads * self.pattern_dim

        self.out_dim = embed_dim if out_dim is None else out_dim
        assert disable_out_projection or (self.out_dim > 0), "output projection dimension has to be positive."

        if normalize_pattern_affine:
            assert normalize_pattern, "affine pattern normalization without pattern normalization has no effect."
            self.p_norm_weight = Parameter(torch.Tensor(head_dim))
            self.p_norm_bias = Parameter(torch.Tensor(head_dim))
        else:
            self.register_parameter('p_norm_weight', None)
            self.register_parameter('p_norm_bias', None)

        if self._qkv_same_embed_dim is False:
            if query_as_static:
                self.register_parameter('q_proj_weight', None)
            else:
                self.q_proj_weight = Parameter(torch.Tensor(self.virtual_hopfield_dim, embed_dim))
            if key_as_static:
                self.register_parameter('k_proj_weight', None)
            else:
                self.k_proj_weight = Parameter(torch.Tensor(self.virtual_hopfield_dim, self.kdim))
            if value_as_static:
                self.register_parameter('v_proj_weight', None)
            else:
                self.v_proj_weight = Parameter(torch.Tensor(
                    self.virtual_pattern_dim,
                    self.virtual_hopfield_dim if (value_as_connected and not key_as_static) else self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            if num_non_static > 0:
                self.in_proj_weight = Parameter(torch.empty(
                    (not query_as_static) * self.virtual_hopfield_dim +
                    (not key_as_static) * self.virtual_hopfield_dim +
                    (not value_as_static) * self.virtual_pattern_dim, embed_dim))
            else:
                self.register_parameter('in_proj_weight', None)
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias and (num_non_static > 0):
            self.in_proj_bias = Parameter(torch.empty(
                (not query_as_static) * self.virtual_hopfield_dim +
                (not key_as_static) * self.virtual_hopfield_dim + self.virtual_pattern_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        if disable_out_projection:
            self.register_parameter('out_proj', None)
        else:
            if bias and _LinearWithBias is not None:
                self.out_proj = _LinearWithBias(self.virtual_pattern_dim, self.out_dim)
            else:
                self.out_proj = Linear(self.virtual_pattern_dim, self.out_dim, bias=bias)

        self.bias_k, self.bias_v = None, None
        if add_bias_kv:
            if not key_as_static:
                self.bias_k = Parameter(torch.empty(1, 1, self.virtual_hopfield_dim))
            if not value_as_static:
                self.bias_v = Parameter(torch.empty(1, 1, self.virtual_hopfield_dim))
            assert not (self.bias_k is None and self.bias_v is None), r'cannot set key/value bias if both are static.'

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    def _check_execution_mode(self) -> bool:
        return all((
            self.key_as_static, self.query_as_static, self.value_as_static, not self.value_as_connected,
            not self.normalize_pattern, not self.normalize_pattern_affine, self.disable_out_projection
        ))

    def reset_parameters(self):
        if self.p_norm_weight is not None:
            nn.init.ones_(self.p_norm_weight)
            nn.init.zeros_(self.p_norm_bias)

        if self._qkv_same_embed_dim and (self.in_proj_weight is not None):
            nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
        else:
            if self.q_proj_weight is not None:
                nn.init.normal_(self.q_proj_weight, mean=0.0, std=0.02)
            if self.k_proj_weight is not None:
                nn.init.normal_(self.k_proj_weight, mean=0.0, std=0.02)
            if self.v_proj_weight is not None:
                nn.init.normal_(self.v_proj_weight, mean=0.0, std=0.02)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
        if not self.disable_out_projection:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
            if self.out_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.normal_(self.bias_k, mean=0.0, std=0.02)
        if self.bias_v is not None:
            nn.init.normal_(self.bias_v, mean=0.0, std=0.02)

    def __setstate__(self, state):
        super(HopfieldCore, self).__setstate__(state)

    def forward(self,
                query,                            # type: Tensor
                key,                              # type: Tensor
                value,                            # type: Tensor
                key_padding_mask=None,            # type: Optional[Tensor]
                need_weights=True,                # type: bool
                attn_mask=None,                   # type: Optional[Tensor]

                scaling=None,                     # type: Optional[Tensor]
                update_steps_max=5,               # type: Optional[int]
                update_steps_eps=1e-4,            # type: float
                return_raw_associations=False,    # type: bool
                return_pattern_projections=False  # type: bool
                ):
        # type: (...) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
                See "Hopfield Networks is All You Need" for more details in the setting of Hopfield networks.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            scaling: scaling of association heads, often represented as beta (one entry per head).
            update_steps_max: maximum count of association update steps (None equals to infinity).
            update_steps_eps: minimum difference threshold between two consecutive association update steps.
            return_raw_associations: return raw association (softmax) values, unmodified.
            return_pattern_projections: return pattern projection values, unmodified.
        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - scaling: :math:`(num_heads,)`, where num_heads is the amount of heads.
            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
            - attn_raw: :math:``(N, num_heads, L, S)`, where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if self.query_as_static and self.key_as_static:
            assert query.shape[2] == key.shape[2], \
                f'query shape[2] of {query.shape[2]} and key shape[2] of {key.shape[2]} need to be equal'
            head_dim, embed_dim_to_check = query.shape[2], query.shape[2]
        else:
            assert self.query_as_static or (query.shape[2] == self.embed_dim), \
                f'query shape[2] of {query.shape[2]} invalid, needs to be {self.embed_dim}.'
            assert (not self.query_as_static) or (self.query_as_static and query.shape[2] == self.head_dim), \
                f'query shape[2] of {query.shape[2]} invalid, needs to be {self.head_dim}'

            assert self.key_as_static or (key.shape[2] == self.kdim), \
                f'key shape[2] of {key.shape[2]} invalid, needs to be {self.kdim}.'
            assert (not self.key_as_static) or (self.key_as_static and key.shape[2] == self.head_dim), \
                f'key shape[2] of {key.shape[2]} invalid, needs to be {self.head_dim}'
            head_dim, embed_dim_to_check = self.head_dim, self.head_dim if self.query_as_static else self.embed_dim

        assert self.value_as_static or (value.shape[2] == self.vdim), \
            f'value shape[2] of {value.shape[2]} invalid, needs to be {self.vdim}.'
        assert any((
            not self.value_as_static, self.value_as_static and value.shape[2] == self.pattern_dim,
            self.disable_out_projection)
        ), f'value shape[2] of {value.shape[2]} invalid, needs to be {self.pattern_dim}'

        out_weights, out_bias = None, None
        if not self.disable_out_projection:
            out_weights, out_bias = self.out_proj.weight, self.out_proj.bias

        if not self._qkv_same_embed_dim:
            return hopfield_core_forward(
                query=query, key=key, value=value, embed_dim_to_check=embed_dim_to_check, num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, bias_k=self.bias_k,
                bias_v=self.bias_v, add_zero_attn=self.add_zero_attn, dropout_p=self.dropout,
                out_proj_weight=out_weights, out_proj_bias=out_bias, training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,

                key_as_static=self.key_as_static, query_as_static=self.query_as_static,
                value_as_static=self.value_as_static, value_as_connected=self.value_as_connected,
                normalize_pattern=self.normalize_pattern, normalize_pattern_eps=self.normalize_pattern_eps,
                p_norm_weight=self.p_norm_weight, p_norm_bias=self.p_norm_bias,
                head_dim=head_dim, pattern_dim=self.pattern_dim, scaling=scaling,
                update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
                return_raw_associations=return_raw_associations, return_projected_patterns=return_pattern_projections, sparse=self.sparse)
        else:
            return hopfield_core_forward(
                query=query, key=key, value=value, embed_dim_to_check=embed_dim_to_check, num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, bias_k=self.bias_k,
                bias_v=self.bias_v, add_zero_attn=self.add_zero_attn, dropout_p=self.dropout,
                out_proj_weight=out_weights, out_proj_bias=out_bias, training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,

                key_as_static=self.key_as_static, query_as_static=self.query_as_static,
                value_as_static=self.value_as_static, value_as_connected=self.value_as_connected,
                normalize_pattern=self.normalize_pattern, normalize_pattern_eps=self.normalize_pattern_eps,
                p_norm_weight=self.p_norm_weight, p_norm_bias=self.p_norm_bias,
                head_dim=head_dim, pattern_dim=self.pattern_dim, scaling=scaling,
                update_steps_max=update_steps_max, update_steps_eps=update_steps_eps,
                return_raw_associations=return_raw_associations, return_projected_patterns=return_pattern_projections, sparse=self.sparse)
