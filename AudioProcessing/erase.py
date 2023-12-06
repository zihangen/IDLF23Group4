import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from pydub import AudioSegment
# add noise to audio files
class ToMono:
    def __init__(self, channel_first=True):
        self.channel_first = channel_first

    def __call__(self, x):
        assert len(x.shape) == 2, "Can only take two dimenshional Audio Tensors"
        output = torch.mean(x, dim=0, keepdim=True) if self.channel_first else torch.mean(x, dim=1)
        return output

class AudioProcessor: # noises_dir contains noise files
    def __init__(self, noises_dir):
        self.noises_dir = noises_dir
        
    def randomly_erase(self, waveform, sample_rate):

        erase_rate = 0.05 # erase 5% of the total length

        # Convert waveform to numpy array if it's a torch tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        res = waveform
        
        # Calculate the maximum possible length of the mask
        max_mask_len = int(erase_rate * waveform.shape[1])

        # Choose a random start point for the mask
        start_point = np.random.randint(0, waveform.shape[1] - max_mask_len)

        # Apply the mask
        res[:, start_point:start_point + max_mask_len] = 0

        # Convert back to torch tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        return waveform

    def process_file(self, file_path, output_dir): 
        print("filepath", file_path)
        if not os.path.exists(output_dir):
            print("created dir", output_dir)
            os.makedirs(output_dir, exist_ok=True)

        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.size(0) > 1:
            mono = ToMono(channel_first=True)
            waveform = mono(waveform)

        # Save original as WAV
        original_wav_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + "_original.wav")
        torchaudio.save(original_wav_path, waveform, sample_rate)

        # Apply and save random erase
        erased_waveform = self.randomly_erase(waveform, sample_rate)
        erased_wav_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + "_erased.wav")
        torchaudio.save(erased_wav_path, erased_waveform, sample_rate)

    def convert_flac_to_wav(self, flac_file_path, wav_file_path):
        audio = AudioSegment.from_file(flac_file_path, format="flac")
        audio.export(wav_file_path, format="wav")
        print(f"Converted {flac_file_path} to {wav_file_path}")

    def convert_directory(self, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".flac"):
                    flac_path = os.path.join(root, file)
                    wav_path = os.path.splitext(flac_path)[0] + ".wav"
                    self.convert_flac_to_wav(flac_path, wav_path)

def processfiles(directory, noises_dir):
    processor = AudioProcessor(noises_dir)
    processor.convert_directory(directory)
    for path in os.listdir(directory):
        root, file = os.path.split(path)
        if file.endswith('.wav'):
            outdir = os.path.join(directory, os.path.splitext(path)[0])
            fullpath = os.path.join(directory, path)
            processor.process_file(fullpath, outdir)


# Replace with actual paths
if __name__ == "__main__":
    processfiles('LibriSpeech/dev-clean/84/121550', 'noise')