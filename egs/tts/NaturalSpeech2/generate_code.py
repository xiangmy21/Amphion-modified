import os
import torch
import numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
from tqdm import tqdm
from glob import glob
from collections import defaultdict

output_path = "/home/srt15/amphion/data/LJSpeech"
dataset_path = "/home/srt15/LJSpeech"
dataset="LJSpeech" # "LJSpeech" or "LibriTTS"

def libritts_statistics(data_dir):
    speakers = []
    distribution2speakers2phrases2utts = defaultdict(
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

            phrase_infos = glob(speaker_info + "/*")

            for phrase_info in phrase_infos:
                phrase = phrase_info.split("/")[-1]

                utts = glob(phrase_info + "/*.wav")

                for utt in utts:
                    uid = utt.split("/")[-1].split(".")[0]
                    distribution2speakers2phrases2utts[distribution][speaker][phrase].append(uid)

    unique_speakers = list(set(speakers))
    unique_speakers.sort()

    print("Speakers: \n{}".format("\t".join(unique_speakers)))
    return distribution2speakers2phrases2utts, unique_speakers

def ljspeech_statistics(data_dir):
    speakers = ["LJSpeech"]
    distribution2speakers2phrases2utts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    # Assuming all .wav files are directly under data_dir/wavs
    wav_files = glob(os.path.join(data_dir, "wavs", "*.wav"))

    for wav_file in wav_files:
        uid = os.path.basename(wav_file).split(".")[0]
        distribution2speakers2phrases2utts["LJSpeech"]["LJSpeech"]["default"].append(uid)

    return distribution2speakers2phrases2utts, speakers

class GenerateCode:
    def __init__(self, dataset):
        self.device = "cuda:1"
        self.codec = self.build_codec()
        print("-" * 10)
        print(f"Preparing samples for {dataset}...\n")

        if dataset == "LibriTTS":
            self.statistics_function = libritts_statistics
        else:
            self.statistics_function = ljspeech_statistics

        distribution2speakers2phrases2utts, _ = self.statistics_function(dataset_path)

        for distribution, speakers2phrases2utts in tqdm(
            distribution2speakers2phrases2utts.items()
        ):
            for speaker, phrases2utts in tqdm(speakers2phrases2utts.items()):
                phrase_names = list(phrases2utts.keys())

                for chosen_phrase in phrase_names:
                    for chosen_uid in phrases2utts[chosen_phrase]:
                        if dataset == "LibriTTS":
                            uid = "{}#{}#{}#{}".format(
                                distribution, speaker, chosen_phrase, chosen_uid
                            )
                            wav_path = os.path.join(self.dataset_path, "{}/{}/{}/{}.wav".format(distribution, speaker, chosen_phrase, chosen_uid))
                        else :
                            uid = chosen_uid
                            wav_path = os.path.join(dataset_path, "wavs", "{}.wav".format(chosen_uid))  
                        assert os.path.exists(wav_path)
                        code = self.get_code(wav_path).cpu().numpy()
                        save_path = os.path.join(output_path, "code", uid)
                        save_dir = os.path.dirname(save_path)
                        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
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
        return ref_code[0]

if __name__ == "__main__":
    GenerateCode(dataset)
