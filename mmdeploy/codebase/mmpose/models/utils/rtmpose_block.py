# Copyright (c) OpenMMLab. All rights reserved.

import torch

from mmdeploy.core import FUNCTION_REWRITER
import torch.nn.functional as F

from mmpose.models.utils import rope


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.utils.rtmpose_block.RTMBlock.shift_mixing', backend='ncnn')
def rtmblock__shift_mixing__ncnn(ctx, self, x):
    """Rewrite `shift_mixing` for ncnn backend.

    ncnn does not support negative dimension for torch.chunk and torch.cat
    ncnn pad shape does not support float input
    """
    x_shift, x_pass = x.chunk(2, dim=x.dim() - 1)
    x_shift = torch.cat(
        [torch.zeros_like(x_shift[:, 0:1, :]), x_shift[:, :-1, :]], dim=1)
    x = torch.cat((x_shift, x_pass), dim=x.dim() - 1)
    return x


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.utils.rtmpose_block.ScaleNorm.forward', backend='ncnn')
def scalenorm__forward__ncnn(ctx, self, x):
    """Rewrite `shift_mixing` for ncnn backend.

    ncnn does not support negative dimension for torch.chunk and torch.cat
    ncnn pad shape does not support float input
    """
    norm = torch.norm(x, dim=2, keepdim=True)
    norm = norm * self.scale
    # Rewrite for ncnn binaryop broadcast.
    norm = norm.clamp(min=self.eps).unsqueeze(2)
    return (x.unsqueeze(2) / norm).squeeze(2) * self.g


@FUNCTION_REWRITER.register_rewriter(
    'mmpose.models.utils.rtmpose_block.RTMBlock._forward', backend='ncnn')
def rtmblock___forward_ncnn(ctx, self, inputs):
    """Rewrite `_forward` for ncnn backend.

    ncnn does not support negative dimension for Split op.
    """
    if self.attn_type == 'self-attn':
        x = inputs
    else:
        x, k, v = inputs

    x = self.ln(x)
    if self.shift is not None:
        x = self.shift_mixing(x)
    uv = self.uv(x)
    if self.attn_type == 'self-attn':
        uv = self.act_fn(uv)
        u = uv[..., :self.e]
        v = uv[..., 512:1024]
        base = uv[..., 2 * self.e:2 * self.e + self.s]

        q = (base.unsqueeze(1) * self.gamma[None, None, 0:1, :] +
             self.beta[None, None, 0:1, :]).squeeze(1)
        k = (base.unsqueeze(1) * self.gamma[None, None, 1:2, :] +
             self.beta[None, None, 1:2, :]).squeeze(1)

        if self.pos_enc:
            q = rope(q, dim=1)
            k = rope(k, dim=1)
    else:
        u, q = torch.split(self.act_fn(uv), [self.e, self.s], dim=uv.dim() - 1)

        if self.shift:
            k = self.shift_mixing(k)
            v = self.shift_mixing(v)

        k = self.k_fc(k)
        v = self.v_fc(v)

        if self.pos_enc:
            q = rope(q, 1)
            k = rope(k, 1)
    qk = torch.bmm(q, k.permute(0, 2, 1))
    if self.use_rel_bias:
        if self.attn_type == 'self-attn':
            bias = self.rel_pos_bias(q.size(1))
        else:
            bias = self.rel_pos_bias(q.size(1), k.size(1))
        qk += bias[:, :q.size(1), :k.size(1)]

    kernel = torch.square(F.relu(qk / self.sqrt_s))
    if self.dropout_rate > 0.:
        kernel = self.dropout(kernel)
    # Rewrite for ncnn to avoid arm fp16 cpu crash. Although
    # there is no broadcast.
    x = (u.unsqueeze(1) * torch.bmm(kernel, v).unsqueeze(1)).squeeze(1)
    x = self.o(x)

    return x
