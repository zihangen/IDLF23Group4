import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from pydub import AudioSegment
import argparse
# add noise to audio files
parser = argparse.ArgumentParser(description = "audio transforms")
parser.add_argument('--noise_type', type=str, default='random', help='desired noise type, possible values are "overlap", "white", "erase", "background"')
parser.add_argument('--noise_files', type=str, default='./noise_files', help='directory of noise files to add')
parser.add_argument('--clean_files', type=str, default='./data', help='directory of clean files to process')
parser.add_argument('--output_dir', type=str, default='./processed_data', help='output directory')
parser.add_argument('--snr', type=int, default=1, help="snr for noise mixing")
opt = parser.parse_args()

class ToMono:
    def __init__(self, channel_first=True):
        self.channel_first = channel_first
    def __call__(self, x):
        assert len(x.shape) == 2, "Can only take two dimenshional Audio Tensors"
        output = torch.mean(x, dim=0, keepdim=True) if self.channel_first else torch.mean(x, dim=1)
        return output

class AudioProcessor: # noises_dir contains noise files, audio_dir contains speech files
    def __init__(self, noises_dir=opt.noise_files, root=opt.clean_files, sample_rate=16000):
        self.clean_dir, self.processed_dir = self.create_dir(opt.output_dir) # create the output dir and clean/noisy folders
        self.noises_dir = noises_dir
        self.root = root
        self.sample_rate = sample_rate
        self.convert_noise_files()
        self.noise_files = [os.path.join(noises_dir, f) for f in os.listdir(noises_dir) if f.endswith('.wav')]
        self.mode = opt.noise_type
    

    # creates a clean and a noise dir and return thier paths
    def create_dir(self, output_dir): 
        if not os.path.exists(output_dir):
            print("created dir", output_dir)
            os.makedirs(output_dir, exist_ok=True)
        clean_dir = os.path.join(output_dir, "clean")
        noise_dir = os.path.join(output_dir, "noise")
        if not os.path.exists(clean_dir):
            os.makedirs(clean_dir, exist_ok=True)
            print("created dir", clean_dir)
        if not os.path.exists(noise_dir):
            os.makedirs(noise_dir, exist_ok=True)
            print("created dir", noise_dir)
        return clean_dir, noise_dir

    def convert_noise_files(self):
        for file in os.listdir(self.noises_dir):
            if file.endswith('.mp3'):
                mp3_path = os.path.join(self.noises_dir, file)
                wav_path = os.path.join(self.noises_dir, os.path.splitext(file)[0] + '.wav')
                waveform, sample_rate = torchaudio.load(mp3_path)
                if sample_rate != self.sample_rate:
                    waveform = self.resample(waveform, sample_rate)
                torchaudio.save(wav_path, waveform, sample_rate)
                os.remove(mp3_path)

    def convert_flac_to_wav(self, flac_file_path, wav_file_path):
            audio = AudioSegment.from_file(flac_file_path, format="flac")
            audio.export(wav_file_path, format="wav")
            print(f"Converted {flac_file_path} to {wav_file_path}")

    def convert_flac_files_in_dir(self, num_file=None):
        # create a wav for each flac file, save to self.clean_dir
        count = 0
        for root, dirs, files in os.walk(self.root):
                for file in files:
                    if file.endswith(".flac"):
                        flac_path = os.path.join(root, file)
                        wav_path = os.path.join(self.clean_dir, os.path.splitext(file)[0] + ".wav")
                        self.convert_flac_to_wav(flac_path, wav_path)
                        count += 1
                        if num_file != None and count > num_file:
                            break
        print(count, "files loaded")

                            
                            
    def resample(self, waveform, orig_sample_rate):
        # resample audio segments to match
        resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=self.sample_rate)
        waveform = resampler(waveform)
        return waveform
    
    def load_wav(self, filename):
        waveform, samplerate = torchaudio.load(filename)
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            mono = ToMono()
            waveform = mono(waveform)
        # Resample noise waveform to match target sample rate
        if samplerate != self.sample_rate:
            waveform = self.resample(waveform, self.sample_rate)
        return waveform
    
    # randomly erase up to 5 segments each being 100 ms
    def randomly_erase(self, waveform, sample_rate):
    # Convert waveform to numpy array if it's a torch tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        # Calculate the number of samples for 100ms
        num_samples_for_100ms = int(0.1 * sample_rate)  # 100ms = 0.1 seconds
        # Randomly choose the number of segments to erase (between 1 and 5)
        num_segments = np.random.randint(1, 6)
        for _ in range(num_segments):
            # Ensure the segment does not exceed waveform length
            if num_samples_for_100ms >= waveform.shape[1]:
                break
            # Choose a random start point for the mask
            start_point = np.random.randint(0, waveform.shape[1] - num_samples_for_100ms)
            # Apply the mask
            waveform[:, start_point:start_point + num_samples_for_100ms] = 0
        # Convert back to torch tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        return waveform

    def overlay_with_background_noise(self, waveform, noise_file, snr=opt.snr):
    # Load noise file
        noise_waveform = self.load_wav(noise_file)
        # Loop to extend noise waveform to match length of the main waveform
        while noise_waveform.size(1) < waveform.size(1):
            noise_waveform = torch.cat((noise_waveform, noise_waveform), dim=1)
        # Trim noise waveform if it's longer than the main waveform
        if noise_waveform.size(1) > waveform.size(1):
            noise_waveform = noise_waveform[:, :waveform.size(1)]
        # Process RIR
        if "rir" in str(noise_file):
            rir = noise_waveform[:, int(self.sample_rate * 1.01): int(self.sample_rate * 1.3)]
            rir = rir / torch.linalg.vector_norm(rir, ord=2)
            augmented = F.fftconvolve(waveform, rir)
            return augmented
        # Prepare SNR
        snr_dbs = torch.tensor([snr])
        # Pad noise waveform if it's shorter than the main waveform
        if noise_waveform.size(1) < waveform.size(1):
            noise_waveform = torch.nn.functional.pad(noise_waveform, (0, waveform.size(1) - noise_waveform.size(1)))
        # Add noise to the waveform
        return F.add_noise(waveform, noise_waveform, snr_dbs)
    
    def save_file(self, waveform, output_name, sampled_noise):
        print("saving processed file", output_name, "as", f"{output_name}_{os.path.basename(sampled_noise)}")
        overlay_wav_path = os.path.join(self.processed_dir, f"{output_name}_{os.path.basename(sampled_noise)}")
        torchaudio.save(overlay_wav_path, waveform, self.sample_rate)

    def process_file(self, file_path, output_name): 
        waveform = self.load_wav(file_path)
        # Overlay with each background noise and save
        # possible values "overlap", "white", "erase", "background"'
        if self.mode == 'random':
            sampled_noise = random.choice(self.noise_files)
            overlay_waveform = self.overlay_with_background_noise(waveform, sampled_noise)
            self.save_file(overlay_waveform, output_name, sampled_noise)

        elif self.mode == 'white':
            # find the white noise file to overlap
            sampled_noise = [n for n in os.listdir(self.noise_files) if 'white' in n]
            assert(len(sampled_noise) == 1)
            sampled_noise = sampled_noise[0]
            overlay_waveform = self.overlay_with_background_noise(waveform, sampled_noise)
            self.save_file(overlay_waveform, output_name, sampled_noise)
        # Apply and save random erase
        elif self.mode == 'overlap':
            sampled_noise = [n for n in os.listdir(self.noise_files) if 'overlap' in n]
            sampled_noise = random.choice(sampled_noise)
            overlay_waveform = self.overlay_with_background_noise(waveform, sampled_noise)
            self.save_file(overlay_waveform, output_name, sampled_noise)
        elif self.mode == 'erase':
            erased_waveform = self.randomly_erase(waveform, self.sample_rate)
            self.save_file(erased_waveform, output_name, 'erased.wav')
        elif self.mode == 'background':
            sampled_noise = [n for n in os.listdir(self.noise_files) if 'background' in n]
            sampled_noise = random.choice(sampled_noise)
            overlay_waveform = self.overlay_with_background_noise(waveform, sampled_noise)
            self.save_file(overlay_waveform, output_name, sampled_noise)
        else:
            print("invalid noise mode received:", self.mode)


    def process_all_files(self): # generate noisy files from clean files
        for i, filename in enumerate(os.listdir(self.clean_dir)):
            if ('.wav' in filename):
                self.process_file(os.path.join(self.clean_dir, filename), os.path.basename(filename).split('.')[0])
    

# Replace with actual paths
if __name__ == "__main__":
    processor = AudioProcessor()
    #overlay_dir = '/Users/bli/Desktop/dlproject/noises/'
    #flac_file_paths = [os.path.join(overlay_dir, p) for p in os.listdir(overlay_dir)]
    #wav_file_paths = []
    #for i, p in enumerate(flac_file_paths):
    #    if 'flac' in p:
    #        wav_file_path = os.path.join(overlay_dir, f"{i}overlay.wav")
    #        processor.convert_flac_to_wav(p, wav_file_path)
    processor.convert_flac_files_in_dir(num_file=10)
    processor.process_all_files()