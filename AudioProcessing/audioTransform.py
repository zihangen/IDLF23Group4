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
        self.convert_noise_files()
        self.noise_files = [os.path.join(noises_dir, f) for f in os.listdir(noises_dir) if f.endswith('.wav')]

    def convert_noise_files(self):
        for file in os.listdir(self.noises_dir):
            if file.endswith('.mp3'):
                mp3_path = os.path.join(self.noises_dir, file)
                wav_path = os.path.join(self.noises_dir, os.path.splitext(file)[0] + '.wav')
                waveform, sample_rate = torchaudio.load(mp3_path)
                torchaudio.save(wav_path, waveform, sample_rate)
                os.remove(mp3_path)

    def randomly_erase(self, waveform, sample_rate):
        # Convert waveform to numpy array if it's a torch tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        res = waveform
        # Calculate the maximum possible length of the mask as 10% of the waveform length
        max_mask_len = int(0.1 * waveform.shape[1])

        # Choose a random start point for the mask
        start_point = np.random.randint(0, waveform.shape[1] - max_mask_len)

        # Apply the mask
        res[:, start_point:start_point + max_mask_len] = 0

        # Convert back to torch tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        return waveform

    def overlay_with_background_noise(self, waveform, noise_file):
        noise_waveform, sample_rate = torchaudio.load(noise_file)
        if noise_waveform.size(0) > 1:
            mono = ToMono(channel_first=True)
            noise_waveform = mono(noise_waveform)
        if "rir" in str(noise_file):
            #print("rir noise")
            rir = noise_waveform[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
            rir = rir / torch.linalg.vector_norm(rir, ord=2)
            augmented = F.fftconvolve(waveform, rir)
            return augmented
        
        snr_dbs = torch.tensor([1])
        # Resize noise to match the waveform length
        #print("wavesizes", noise_waveform.size(), waveform.size())
        if noise_waveform.size(1) > waveform.size(1):
            noise_waveform = noise_waveform[:, :waveform.size(1)]
        else:
            noise_waveform = torch.nn.functional.pad(noise_waveform, (0, waveform.size(1) - noise_waveform.size(1)))
        #print(waveform.size(), noise_waveform.size(), snr_dbs.size())
        return F.add_noise(waveform, noise_waveform, snr_dbs)

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

        # Overlay with each background noise and save
        for noise_file in self.noise_files:
            overlay_waveform = self.overlay_with_background_noise(waveform, noise_file)
            overlay_wav_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_overlay_{os.path.basename(noise_file)}")
            torchaudio.save(overlay_wav_path, overlay_waveform, sample_rate)
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
    #processor.convert_directory(directory)
    for path in os.listdir(directory):
        root, file = os.path.split(path)
        if file.endswith('.wav'):
            outdir = os.path.join(directory, os.path.splitext(path)[0])
            fullpath = os.path.join(directory, path)
            processor.process_file(fullpath, outdir)


# Replace with actual paths
if __name__ == "__main__":
    processfiles('/Users/bli/Desktop/dlproject/longData', '/Users/bli/Desktop/dlproject/longData/noises')
