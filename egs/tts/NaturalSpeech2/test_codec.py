# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio


class TestCodec:
    def __init__(self):
        self.device = "cuda:0"
        self.codec = self.build_codec()
        self.get_ref_code()

    def build_codec(self):
        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model = encodec_model.to(device=self.device)
        encodec_model.set_target_bandwidth(12.0)
        return encodec_model

    def get_ref_code(self):
        ref_wav_path = "/home/srt15/amphion/egs/tts/NaturalSpeech2/prompt_example/cry.mp3"
        ref_wav, sr = torchaudio.load(ref_wav_path)
        ref_wav = convert_audio(
            ref_wav, sr, self.codec.sample_rate, self.codec.channels
        )
        ref_wav = ref_wav.unsqueeze(0).to(device=self.device)

        with torch.no_grad():
            encoded_frames = self.codec.encode(ref_wav)
            print("encoded_frames[0]: ", encoded_frames[0][0].shape)
            ref_code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        print("ref_code: ", ref_code.shape)

        ref_mask = torch.ones(ref_code.shape[0], ref_code.shape[-1]).to(ref_code.device)
        # print(ref_mask.shape)

        return ref_code, ref_mask
    
if __name__ == "__main__":
    TestCodec()
    # (1, 16, 482)
    # cry.mp3 length 6.42s
    # 6.42*2.4k/482 ~ 320.   =>  hop_size = 320