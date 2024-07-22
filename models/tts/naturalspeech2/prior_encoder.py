# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.naturalpseech2.transformers import (
    TransformerEncoder,
    DurationPredictor,
    PitchPredictor,
    LengthRegulator,
)


class PriorEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc_emb_tokens = nn.Embedding(
            cfg.vocab_size, cfg.encoder.encoder_hidden, padding_idx=0
        )
        self.enc_emb_tokens.weight.data.normal_(mean=0.0, std=1e-5)
        self.encoder = TransformerEncoder(
            enc_emb_tokens=self.enc_emb_tokens, cfg=cfg.encoder
        )

        self.duration_predictor = DurationPredictor(cfg.duration_predictor)
        self.pitch_predictor = PitchPredictor(cfg.pitch_predictor)
        self.length_regulator = LengthRegulator()

        self.pitch_min = cfg.pitch_min
        self.pitch_max = cfg.pitch_max
        self.pitch_bins_num = cfg.pitch_bins_num

        pitch_bins = torch.exp(
            torch.linspace(
                np.log(self.pitch_min), np.log(self.pitch_max), self.pitch_bins_num - 1
            )
        )
        self.register_buffer("pitch_bins", pitch_bins)

        self.pitch_embedding = nn.Embedding(
            self.pitch_bins_num, cfg.encoder.encoder_hidden
        )

    def forward(
        self,
        phone_id,
        duration=None,
        pitch=None,
        phone_mask=None,
        mask=None,
        ref_emb=None,
        ref_mask=None,
        is_inference=False,
        flow=False,
    ):
        """
        input:
        phone_id: (B, N)
        duration: (B, N)
        pitch: (B, T)
        phone_mask: (B, N); mask is 0
        mask: (B, T); mask is 0 对齐batch时用来遮空帧
        ref_emb: (B, d, T')
        ref_mask: (B, T'); mask is 0

        output:
        prior_embedding: (B, d, T)
        pred_dur: (B, N)
        pred_pitch: (B, T)
        """
        # TODO: 流式生成需要解决三个部分的问题：
        # 1. phone_id 过 self.encoder (TransfomerEncoder) 时, 添加 attn_mask
        # 2. self.duration_predictor (DurationPredictor, 含Conv1d) 生成 duration 时, 改为流式
        # 3. self.pitch_predictor (PitchPredictor, 含Conv1d) 生成 pitch 时, 改为流式
        
        # 构造一个phone_id的attention mask, 可以往后看lookahead个音素
        if flow:
            look_ahead = self.cfg.look_ahead
            phone_attn_mask = torch.zeros(phone_id.shape[1], phone_id.shape[1], device=phone_id.device, dtype=torch.bool)
            for i in range(phone_id.shape[1]):
                phone_attn_mask[i, max(0, i-look_ahead):min(i+look_ahead+1, phone_id.shape[1])] = True
        else:
            phone_attn_mask = None

        x = self.encoder(phone_id, phone_mask, phone_attn_mask, ref_emb.transpose(1, 2)) # (B, N, d) d default 512

        # 将x切分成长度为blk_size的块,每个块的两边加上padding
        if flow:
            blk_size = self.cfg.dur_blk_size
            padding = self.cfg.dur_blk_padding
            dur_pred_out = {
                "dur_pred_log": torch.zeros(x.shape[0], x.shape[1], device=x.device),
                "dur_pred": torch.zeros(x.shape[0], x.shape[1], device=x.device),
                "dur_pred_round": torch.zeros(x.shape[0], x.shape[1], dtype=torch.long),
            }
            for i in range(0, x.shape[1], blk_size):
                view = slice(max(0, i-padding), min(i+blk_size+padding, x.shape[1]))
                save_view = slice(i, min(i+blk_size, x.shape[1]))
                get_view = slice(min(i, padding), min(i, padding) + save_view.stop - save_view.start)
                dur_pred_out_blk = self.duration_predictor(x[:, view], phone_mask, ref_emb, ref_mask)
                dur_pred_out["dur_pred_log"][:, save_view] = dur_pred_out_blk["dur_pred_log"][:, get_view]
                dur_pred_out["dur_pred"][:, save_view] = dur_pred_out_blk["dur_pred"][:, get_view]
                dur_pred_out["dur_pred_round"][:, save_view] = dur_pred_out_blk["dur_pred_round"][:, get_view]
        else:
            dur_pred_out = self.duration_predictor(x, phone_mask, ref_emb, ref_mask) # duration和pitch受ref_emb影响
        # dur_pred_out: {dur_pred_log, dur_pred, dur_pred_round}

        if is_inference or duration is None: # 这里通过预测的duration拓展了x的长度
            x, mel_len = self.length_regulator(
                x,
                dur_pred_out["dur_pred_round"],
                max_len=torch.max(torch.sum(dur_pred_out["dur_pred_round"], dim=1)),
            )
        else:
            x, mel_len = self.length_regulator(x, duration, max_len=pitch.shape[1]) # (B, T, d)

        if flow:
            frame_size = self.cfg.pit_blk_size
            padding = self.cfg.pit_blk_padding
            pitch_pred_log = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            for i in range(0, x.shape[1], frame_size):
                view = slice(max(0, i-padding), min(i+frame_size+padding, x.shape[1]))
                save_view = slice(i, min(i+frame_size, x.shape[1]))
                get_view = slice(min(i, padding), min(i, padding) + save_view.stop - save_view.start)
                pitch_pred_log[:, save_view] = self.pitch_predictor(x[:, view], mask, ref_emb, ref_mask)[:, get_view]
        else:
            pitch_pred_log = self.pitch_predictor(x, mask, ref_emb, ref_mask)

        if is_inference or pitch is None:
            pitch_tokens = torch.bucketize(pitch_pred_log.exp(), self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)
        else:
            pitch_tokens = torch.bucketize(pitch, self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)

        x = x + pitch_embedding

        if (not is_inference) and (mask is not None):
            x = x * mask.to(x.dtype)[:, :, None]

        prior_out = {
            "dur_pred_round": dur_pred_out["dur_pred_round"],
            "dur_pred_log": dur_pred_out["dur_pred_log"],
            "dur_pred": dur_pred_out["dur_pred"],
            "pitch_pred_log": pitch_pred_log,
            "pitch_token": pitch_tokens,
            "mel_len": mel_len,
            "prior_out": x,
        }

        return prior_out
