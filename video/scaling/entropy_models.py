import math

import torch
import numpy as np
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


# pylint: disable=W0221
class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs,  torch.tensor(bound))
        return torch.clamp_min(inputs, bound)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, bound = ctx.saved_tensors
        pass_through_1 = inputs >= bound
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through * grad_output, None
# pylint: enable=W0221


class EntropyCoder():
    def __init__(self):
        super().__init__()

        from MLCodec_extensions_cpp import RansEncoder, RansDecoder
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from MLCodec_extensions_cpp import pmf_to_quantized_cdf as _pmf_to_cdf
        cdf = _pmf_to_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def reset(self):
        self.encoder.reset()

    def add_cdf(self, cdf, cdf_length, offset):
        enc_cdf_idx = self.encoder.add_cdf(cdf, cdf_length, offset)
        dec_cdf_idx = self.decoder.add_cdf(cdf, cdf_length, offset)
        assert enc_cdf_idx == dec_cdf_idx
        return enc_cdf_idx

    def encode_y(self, symbols, cdf_group_index):
        # symbols: int16, high 8 bits: int8 symbol to be encoded; low 8 bits: uint8 index to use
        assert symbols.dtype == torch.int16
        self.encoder.encode_y(symbols.cpu().numpy(), cdf_group_index)

    def encode_z(self, symbols, cdf_group_index, start_offset, per_channel_size):
        self.encoder.encode_z(symbols.to(torch.int8).cpu().numpy(),
                              cdf_group_index, start_offset, per_channel_size)

    def flush(self):
        self.encoder.flush()

    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))

    def decode_y(self, indexes, cdf_group_index):
        self.decoder.decode_y(indexes.to(torch.uint8).cpu().numpy(), cdf_group_index)

    def decode_and_get_y(self, indexes, cdf_group_index, device, dtype):
        rv = self.decoder.decode_and_get_y(indexes.to(torch.uint8).cpu().numpy(), cdf_group_index)
        rv = torch.as_tensor(rv)
        return rv.to(device).to(dtype)

    def decode_z(self, total_size, cdf_group_index, start_offset, per_channel_size):
        self.decoder.decode_z(total_size, cdf_group_index, start_offset, per_channel_size)

    def get_decoded_tensor(self, device, dtype, non_blocking=False):
        rv = self.decoder.get_decoded_tensor()
        rv = torch.as_tensor(rv)
        return rv.to(device, non_blocking=non_blocking).to(dtype)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.encoder.set_use_two_encoders(use_two_entropy_coders)
        self.decoder.set_use_two_decoders(use_two_entropy_coders)


class Bitparm(nn.Module):
    def __init__(self, qp_num, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1]), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1]), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty([qp_num, channel, 1]), 0, 0.01))
        else:
            self.a = None

    def forward(self, x, index):
        h = torch.index_select(self.h, 0, index)
        b = torch.index_select(self.b, 0, index)
        x = x * F.softplus(h) + b
        if self.final:
            return x

        a = torch.index_select(self.a, 0, index)
        return x + torch.tanh(x) * torch.tanh(a)


class AEHelper():
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self.cdf_group_index = None
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset


class BitEstimator(AEHelper, nn.Module):
    def __init__(self, qp_num=1, channel=1):
        super().__init__()
        self.f1 = Bitparm(qp_num, channel)
        self.f2 = Bitparm(qp_num, channel)
        self.f3 = Bitparm(qp_num, channel)
        self.f4 = Bitparm(qp_num, channel, True)
        self.qp_num = qp_num
        self.channel = channel

    def forward(self, x, index=0):
        return self.get_cdf(x, index)

    def get_logits_cdf(self, x, index=0):
        x = self.f1(x, index)
        x = self.f2(x, index)
        x = self.f3(x, index)
        x = self.f4(x, index)
        return x

    def get_cdf(self, x, index=0):
        return torch.sigmoid(self.get_logits_cdf(x, index))

    def get_prob(self, x, index=0):
        lower = self.get_cdf(x - 0.5, index)
        upper = self.get_cdf(x + 0.5, index)
        prob = upper - lower
        prob = LowerBound.apply(prob, 1e-9)
        return prob

    def update(self, entropy_coder):
        self.entropy_coder = entropy_coder

        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
            index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

            minima = medians + 8
            for i in range(8, 1, -1):
                samples = torch.zeros_like(medians) - i
                probs = self.forward(samples, index)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 8
            for i in range(8, 1, -1):
                samples = torch.zeros_like(medians) + i
                probs = self.forward(samples, index)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            # record the length of pmf for each channel
            pmf_length = maxima - minima + 1
            max_length = torch.max(pmf_length).item()
            offsets = -minima
            device = medians.device

            # offsets and pmf_length are stacked to simplify the input of the entropy coder
            # n_dim, max_length
            cdf_length = torch.zeros((self.qp_num, self.channel, 1), device=device)
            offset = torch.zeros((self.qp_num, self.channel, 1), device=device)

            entropy_coder = entropy_coder
            self.entropy_coder = entropy_coder

            for qp in range(self.qp_num):
                prob = torch.zeros((self.channel, max_length), device=device)
                for ch in range(self.channel):
                    values = torch.arange(minima[qp, ch], maxima[qp, ch] + 1, device=device)
                    values = values.reshape(-1, 1, 1, 1)
                    values = values - medians[qp, ch]

                    prob[ch, : pmf_length[qp, ch]] = self.get_prob(values, qp)[ch].reshape(-1)

                # pmf_length should be a torch.int32 and reside in cpu
                entropy_coder_precision = 16
                # cdf stores all cdfs catenated together
                cdf = torch.zeros((self.channel, max_length + 2), dtype=torch.int32, device='cpu')
                cdf_length[qp] = torch.IntTensor(pmf_length)[qp].unsqueeze(-1)
                offset[qp] = torch.IntTensor(offsets)[qp].unsqueeze(-1)

                for ch in range(self.channel):
                    prob_ch = prob[ch, : pmf_length[qp, ch]]
                    cdf[ch, : pmf_length[qp, ch] + 2] = EntropyCoder.pmf_to_quantized_cdf(prob_ch.tolist(), entropy_coder_precision)
                cdf = cdf.cuda()

                cdf = cdf.reshape(-1, max_length + 2)
                cdf_length = cdf_length.reshape(-1, 1)
                offset = offset.reshape(-1, 1)

                self.cdf_index = entropy_coder.add_cdf(cdf, cdf_length, offset)

    def compress(self, x, index=0):
        assert self.entropy_coder is not None
        x = x.detach().int()
        cdf_idx = self.cdf_index.reshape(-1)[index]
        self.entropy_coder.encode_z(x, cdf_idx, 0, self.channel)

    def decompress(self, size, index=0):
        assert self.entropy_coder is not None
        cdf_idx = self.cdf_index.reshape(-1)[index]
        self.entropy_coder.decode_z(size, cdf_idx, 0, self.channel)
        rv = self.entropy_coder.get_decoded_tensor(device='cuda', dtype=torch.int32)
        return rv
