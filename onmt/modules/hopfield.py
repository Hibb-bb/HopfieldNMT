""" Multi-Head Attention module """
import math
import torch
from torch import Tensor
from typing import Optional, Tuple
import torch.nn as nn


import torch
import torch.nn as nn
from torch.autograd import Function


def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for sparsemax: optimal threshold and support size.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply sparsemax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


def _entmax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for 1.5-entmax: optimal threshold and support size.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


class Entmax15Function(Function):
    @classmethod
    def forward(cls, ctx, X, dim=0, k=None):
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


def sparsemax(X, dim=-1, k=None):
    """sparsemax: normalizing sparse transform (a la softmax).
    Solves the projection:
        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    dim : int
        The dimension along which to apply sparsemax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """

    return SparsemaxFunction.apply(X, dim, k)


def entmax15(X, dim=-1, k=None):
    """1.5-entmax: normalizing sparse transform (a la softmax).
    Solves the optimization problem:
        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.
    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """

    return Entmax15Function.apply(X, dim, k)


# Help functions for Rotary Embeddings
# https://arxiv.org/pdf/2104.09864.pdf
# too convoluted to make maxseqlen a parameter.
# we suppose src_seq_len at training and max_length at inference
# are both < 2048 tokens.

def rotaryembeddings(dim: int, maxseqlen=4096, base=10000):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    tmax = torch.arange(maxseqlen, device=inv_freq.device)
    rope = torch.outer(tmax, inv_freq).float()
    # rope is now matrix [maxseqlen, dim/2]
    rope = torch.polar(torch.ones_like(rope), rope)
    return rope


def apply_rotary_emb(query, key, rope):
    query_ = query.float().reshape(*query.shape[:-1], -1, 2)
    query_ = torch.view_as_complex(query_)
    key_ = key.float().reshape(*key.shape[:-1], -1, 2)
    key_ = torch.view_as_complex(key_)
    rope = rope.view(1, query_.size(1), 1, query_.size(3))
    query_out = torch.view_as_real(query_ * rope).flatten(3)
    key_out = torch.view_as_real(key_ * rope).flatten(3)
    return query_out.type_as(query), key_out.type_as(key)


# Help functions for max_relative positions
# https://arxiv.org/abs/1803.02155

def relative_matmul(x: Tensor, z: Tensor,
                    transpose: bool) -> Tensor:
    """
    Helper function for relative positions attention.
    https://arxiv.org/pdf/1803.02155.pdf
    x shape [batch_size x heads x q_len x k_len]
    """
    batch_size = x.size(0)
    heads = x.size(1)
    length = x.size(2)
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.contiguous().view(length, heads * batch_size, -1)
    if transpose:
        z = z.transpose(1, 2)
    x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.view(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def gen_relative_positions(length: int,
                           max_relative_positions: int,
                           cache: bool = False,
                           device: Optional[torch.device] = None
                           ) -> Tensor:
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1,
                                    device=device).unsqueeze(0)
    else:
        range_vec = torch.arange(length, device=device)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


# Help functions to split model dim per head

def shape(x: Tensor, dim_per_head: int) -> Tensor:
    """
    Projection.
    [batchsize x length x modeldim]
    -> [batchsize x heads x length x dimperhead]
    """
    return x.view(x.size(0), x.size(1), -1, dim_per_head) \
        .transpose(1, 2)


def unshape(x: Tensor) -> Tensor:
    """
    Compute context.
    [batchsize x heads x length x dimperhead]
    -> [batchsize x length x modeldim]
    """
    return x.transpose(1, 2).contiguous() \
        .view(x.size(0), -1, x.size(1) * x.size(3))


class MultiHeadedAttention(nn.Module):
    # class MultiHeadedAttention(torch.jit.ScriptModule):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
       max_relative_positions (int): max relative positions
       attn_type: "self" or "context"
    """

    def __init__(self, head_count: int, model_dim: int, dropout: float = 0.1,
                 max_relative_positions: int = 0,
                 attn_type: str = None, add_qkvbias=False, sparse='softmax', hopfield_steps=1) -> None:

        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, model_dim, bias=add_qkvbias)
        self.linear_values = nn.Linear(model_dim, model_dim, bias=add_qkvbias)
        self.linear_query = nn.Linear(model_dim, model_dim, bias=add_qkvbias)

        if sparse == 'softmax':
            self.softmax = nn.Softmax(dim=-1)
        elif sparse == 'sparsemax':
            self.softmax = sparsemax
        elif sparse == 'entmax':
            self.softmax = entmax15
        
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim, bias=add_qkvbias)

        self.max_relative_positions = max_relative_positions
        self.attn_type = attn_type
        self.layer_cache = (False, {'keys': torch.tensor([]),
                                    'values': torch.tensor([])})

        if max_relative_positions > 0:
            # https://arxiv.org/pdf/1803.02155.pdf
            # in the paper they suggest either two embeds
            # relative_key / relative_value or only
            # relative_key. We implemented the same embed
            # for both.
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)
        else:
            self.relative_positions_embeddings = None

            if max_relative_positions == -1:  # rotary embeddings
                self.rope = rotaryembeddings(self.dim_per_head)

        self.hopfield_steps = hopfield_steps
        self.update_steps_eps = 1e-4

    def update_dropout(self, dropout: float) -> None:
        self.dropout.p = dropout

    # @torch.jit.script_method
    def forward(self, key: Tensor, value: Tensor,
                query: Tensor, mask: Optional[Tensor] = None,
                step: Optional[int] = 0
                ) -> Tuple[Tensor, Tensor]:
        """
        Compute the context vector and the attention vectors.

        Args:
           key (Tensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (Tensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (Tensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
           step (int): decoding step (used for Rotary embedding)
        Returns:
           (Tensor, Tensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """
        # 1) Project key, value, and query.
        # as a reminder at training layer_cache[0] remains False

        # for i in range(self.hopfield_steps):
        
        bsz = key.size(0)
        tgt_len, src_len = query.size(1), key.size(1)

        if self.layer_cache[0]:
            if self.attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                query = shape(query, self.dim_per_head)
                key = shape(key, self.dim_per_head)
                value = shape(value, self.dim_per_head)

                if self.max_relative_positions == -1:  # Rotary Embeddings
                    start_pos = step
                    seqlen = query.size(2)
                    rope = self.rope[start_pos:
                                     start_pos +
                                     seqlen].to(query.device)

                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    query, key = apply_rotary_emb(query,
                                                  key,
                                                  rope=rope)
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)

                if self.layer_cache[1]['keys'].numel() != 0:
                    key = torch.cat(
                        (self.layer_cache[1]['keys'], key),
                        dim=2)

                if self.layer_cache[1]['values'].numel() != 0:
                    value = torch.cat(
                        (self.layer_cache[1]['values'], value),
                        dim=2)
                self.layer_cache[1]['keys'] = key
                self.layer_cache[1]['values'] = value
            elif self.attn_type == "context":
                query = self.linear_query(query)
                query = shape(query, self.dim_per_head)
                if self.layer_cache[1]['keys'].numel() == 0:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key, self.dim_per_head)
                    value = shape(value, self.dim_per_head)
                else:
                    key, value = self.layer_cache[1]['keys'],\
                               self.layer_cache[1]['values']
                self.layer_cache[1]['keys'] = key
                self.layer_cache[1]['values'] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key, self.dim_per_head)
            value = shape(value, self.dim_per_head)
            query = shape(query, self.dim_per_head)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.relative_positions_embeddings is not None:
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = gen_relative_positions(
                key_len, self.max_relative_positions,
                cache=self.layer_cache[0],
                device=key.device)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix)
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key

        scores = scores.float()

        if mask is not None:
            # not 100% necessary but expand to nb of heads
            mask = mask.expand(-1, self.head_count, -1, -1)
            # now mask and scores have the same shape
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)

        # 4) Hopfield Iteration

        update_active_heads = torch.tensor([[[True]]] * self.head_count * bsz, device=query.device)
        assert update_active_heads.any(), "at least one head needs to be active."
        xi = attn
        xi_old = None

        while update_active_heads.any():


            active_xi = xi.masked_select(mask=update_active_heads).view(size=(-1, *xi.shape[1:]))
            active_k = key.masked_select(mask=update_active_heads).view(size=(-1, *key.shape[1:]))
            q = torch.masked_scatter(input=query, mask=update_active_heads, source=torch.bmm(active_xi, active_k))


            
            with torch.no_grad():
                xi_active = xi.view(size=(bsz, self.head_count, tgt_len, src_len))
                update_active_heads = (update_step < self.hopfield_steps) | (self.hopfield_steps < 0)
                if xi_old is not None:
                    update_active_heads &= ((xi_old - xi_active).norm(p=2, dim=(2, 3)).max(axis=0)[0]) > self.update_steps_eps 
                update_active_heads = update_active_heads.unsqueeze(dim=1).unsqueeze(dim=2).repeat(repeats=(bsz, 1, 1))
                xi_old = xi_active
            update_step += 1

            xi = torch.masked_scatter(input=xi, mask=update_active_heads, source=self.softmax(
                attn_output_weights.masked_select(mask=update_active_heads).view(size=(-1, *xi.shape[1:])), dim=-1))



        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.relative_positions_embeddings is not None:
            # We use the same embeddings for key and value
            relations_values = relations_keys
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)

        return output, attn
