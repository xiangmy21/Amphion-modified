import os
import torch
import numpy as np

from encodec import EncodecModel
from encodec.utils import convert_audio

from text.cmudict import valid_symbols

import torchaudio

import os
import torchaudio
from tqdm import tqdm
from glob import glob
from collections import defaultdict



def libritts_statistics(data_dir):
    speakers = []
    distribution2speakers2pharases2utts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    distribution_infos = glob(data_dir + "/*")

    for distribution_info in distribution_infos:
        distribution = distribution_info.split("/")[-1]
        print(distribution)

        speaker_infos = glob(distribution_info + "/*")

        if len(speaker_infos) == 0:
            continue

        for speaker_info in speaker_infos:
            speaker = speaker_info.split("/")[-1]

            speakers.append(speaker)

            pharase_infos = glob(speaker_info + "/*")

            for pharase_info in pharase_infos:
                pharase = pharase_info.split("/")[-1]

                utts = glob(pharase_info + "/*.wav")

                for utt in utts:
                    uid = utt.split("/")[-1].split(".")[0]
                    distribution2speakers2pharases2utts[distribution][speaker][
                        pharase
                    ].append(uid)

    unique_speakers = list(set(speakers))
    unique_speakers.sort()

    print("Speakers: \n{}".format("\t".join(unique_speakers)))
    return distribution2speakers2pharases2utts, unique_speakers

class GenerateCode:
    def __init__(self):
        self.device = "cuda:0"
        self.codec = self.build_codec()
        # print(self.codec.sample_rate)
        # return
        print("-" * 10)
        print("Preparing samples for libritts...\n")
        output_path = "/home/srt15/amphion/data/libritts"
        dataset_path = "/home/srt15/LibriTTS"

        # Load
        libritts_path = dataset_path

        distribution2speakers2pharases2utts, _ = libritts_statistics(
            libritts_path
        )

        for distribution, speakers2pharases2utts in tqdm(
            distribution2speakers2pharases2utts.items()
        ):
            for speaker, pharases2utts in tqdm(speakers2pharases2utts.items()):
                pharase_names = list(pharases2utts.keys())

                for chosen_pharase in pharase_names:
                    for chosen_uid in pharases2utts[chosen_pharase]:
                        uid = "{}#{}#{}#{}".format(
                                distribution, speaker, chosen_pharase, chosen_uid
                            )
                        wav_path = os.path.join(dataset_path, "{}/{}/{}/{}.wav".format(
                            distribution, speaker, chosen_pharase, chosen_uid
                        ))
                        assert os.path.exists(wav_path)
                        code = self.get_code(wav_path).cpu().numpy()
                        save_path = os.path.join(output_path, "code", uid)
                        np.save(save_path, code)


    def build_codec(self):
        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model = encodec_model.to(device=self.device)
        encodec_model.set_target_bandwidth(12.0)
        return encodec_model

    def get_code(self, wav_path):
        ref_wav, sr = torchaudio.load(wav_path)
        ref_wav = convert_audio(
            ref_wav, sr, self.codec.sample_rate, self.codec.channels
        )
        ref_wav = ref_wav.unsqueeze(0).to(device=self.device)

        with torch.no_grad():
            encoded_frames = self.codec.encode(ref_wav)
            ref_code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        # print("ref_code: ", ref_code.shape)

        return ref_code[0]
    
if __name__ == "__main__":
    # code = np.load("/home/srt15/amphion/data/libritts/code/train-clean-100#19#198#19_198_000000_000000.npy")
    # print(code[0].shape)
    GenerateCode()