# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
import soundfile as sf
import numpy as np

from models.tts.naturalspeech2.ns2 import NaturalSpeech2
from encodec import EncodecModel
from encodec.utils import convert_audio
from utils.util import load_config

from text import text_to_sequence
from text.cmudict import valid_symbols
from text.g2p import preprocess_english, read_lexicon

import torchaudio


class NS2Inference:
    def __init__(self, args, cfg):
        self.cfg = cfg
        self.args = args

        self.model = self.build_model()
        self.codec = self.build_codec()

        self.symbols = valid_symbols + ["sp", "spn", "sil"] + ["<s>", "</s>"]
        self.phone2id = {s: i for i, s in enumerate(self.symbols)}
        self.id2phone = {i: s for s, i in self.phone2id.items()}

    def build_model(self):
        model = NaturalSpeech2(self.cfg.model)
        model.load_state_dict(
            torch.load(
                os.path.join(self.args.checkpoint_path, "pytorch_model.bin"),
                map_location="cpu",
            )
        )
        model = model.to(self.args.device)
        return model

    def build_codec(self):
        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model = encodec_model.to(device=self.args.device)
        encodec_model.set_target_bandwidth(12.0) # 带宽决定量化器层数 n_q = 16
        return encodec_model

    def get_ref_code(self):
        ref_wav_path = self.args.ref_audio
        ref_wav, sr = torchaudio.load(ref_wav_path)
        ref_wav = convert_audio(
            ref_wav, sr, self.codec.sample_rate, self.codec.channels
        )
        ref_wav = ref_wav.unsqueeze(0).to(device=self.args.device)

        with torch.no_grad():
            encoded_frames = self.codec.encode(ref_wav) # encoded_frames[0][0].shape: (B, 16, 75*seconds)
            # print("encoded_frames[0]: ", encoded_frames[0][0].shape)
            ref_code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        # print("ref_code: ", ref_code.shape) same with encoded_frames[0][0].shape

        ref_mask = torch.ones(ref_code.shape[0], ref_code.shape[-1]).to(ref_code.device)
        # print(ref_mask.shape)

        return ref_code, ref_mask

    def inference(self):
        from time_test import Timer
        with Timer() as t:
            t.name = "Prepare code & phone"
            ref_code, ref_mask = self.get_ref_code()
            # print("ref_code: ", ref_code) # discrete
            # print("ref_code.shape: ", ref_code.shape) # (B, K, T), K is the number of codebooks
            # print("ref_mask: ", ref_mask) # (B, T) all true. 标准库的实现中1代表忽略，0代表保留，在本仓库的transformers.py中输入时取反了，所以这里全1.

            lexicon = read_lexicon(self.cfg.preprocess.lexicon_path)
            if self.args.text != "":
                phone_seq = preprocess_english(self.args.text, lexicon)
            else:
                with open(self.args.text_path, 'r', encoding='utf-8') as file:
                    text = file.read().replace('\r', '').replace('\n', ' ')
                phone_seq = preprocess_english(text, lexicon)
            print(phone_seq)

            phone_id = np.array(
                [
                    *map(
                        self.phone2id.get,
                        phone_seq.replace("{", "").replace("}", "").split(),
                    )
                ]
            )
            phone_id = torch.from_numpy(phone_id).unsqueeze(0).to(device=self.args.device)
            print("Phone nums: ", phone_id.shape[-1])

        flow = True

        x0, prior_out = self.model.inference(
            ref_code, phone_id, ref_mask, self.args.inference_step, flow=flow
        )
        # print(prior_out["dur_pred"]) 
        # print(prior_out["dur_pred_round"]) # (1, N) 每个音素的持续帧数
        # print(torch.sum(prior_out["dur_pred_round"])) # 总帧数
        
        with Timer() as t:
            t.name = "Decoder"
            # latent_ref = self.codec.quantizer.vq.decode(ref_code.transpose(0, 1)) # (B, 128, T)
            # ref_wav = self.codec.decoder(latent_ref)
            if not flow:
                rec_wav = self.codec.decoder(x0) # x0: (1, 128, T), rec_wav: (1, 1, L) L = T*320
            else:
                samples_per_frame = 320
                rec_wav_chunks = torch.zeros(1, 1, x0.shape[-1]*samples_per_frame).to(x0.device)
                blk_size = self.cfg.model.dec_blk_size
                padding = self.cfg.model.dec_blk_padding
                for i in range(0, x0.shape[-1], blk_size):
                    view = slice(max(0, i-padding), min(i+blk_size+padding, x0.shape[-1]))
                    save_view = slice(i*samples_per_frame, min(i+blk_size, x0.shape[-1])*samples_per_frame)
                    get_view = slice(min(i, padding)*samples_per_frame, min(i, padding)*samples_per_frame + save_view.stop - save_view.start)
                    rec_chunk = self.codec.decoder(x0[:, :, view])
                    rec_wav_chunks[:, :, save_view] = rec_chunk[:, :, get_view]

        print("look_ahead: ", self.cfg.model.prior_encoder.look_ahead)
        print("dur_blk_size: ", self.cfg.model.prior_encoder.dur_blk_size)
        print("dur_blk_padding: ", self.cfg.model.prior_encoder.dur_blk_padding)
        print("pit_blk_size: ", self.cfg.model.prior_encoder.pit_blk_size)
        print("pit_blk_padding: ", self.cfg.model.prior_encoder.pit_blk_padding)
        print("multidiffuion window_size: ", self.cfg.model.diffusion.window_size)
        print("multidiffuion stride: ", self.cfg.model.diffusion.stride)
        print("dec_blk_size: ", self.cfg.model.dec_blk_size)
        print("dec_blk_padding: ", self.cfg.model.dec_blk_padding)
        print("Write to file {}".format("out_"+("chunks_" if flow else "")+self.args.ref_audio.split('/')[-1].split('.')[0]))
        os.makedirs(self.args.output_dir, exist_ok=True)

        if not flow:
            sf.write(
                "{}/{}.wav".format(
                    self.args.output_dir, "out_"+self.args.ref_audio.split('/')[-1].split('.')[0]
                ),
                rec_wav[0, 0].detach().cpu().numpy(),
                samplerate=24000,
            )
        else:
            sf.write(
                "{}/{}.wav".format(
                    self.args.output_dir, "out_chunks_"+self.args.ref_audio.split('/')[-1].split('.')[0]
                ),
                rec_wav_chunks[0, 0].detach().cpu().numpy(),
                samplerate=24000,
            )

    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--ref_audio",
            type=str,
            default="",
            help="Reference audio path",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
        )
        parser.add_argument(
            "--inference_step",
            type=int,
            default=200,
            help="Total inference steps for the diffusion model",
        )
