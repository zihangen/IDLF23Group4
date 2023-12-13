import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def visualize_soundwave(wav_file, output_image):
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # Create a time array in seconds
    times = np.arange(len(data)) / float(sample_rate)

    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(times, data, color="skyblue", alpha=0.9)
    plt.xlim(times[0], times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    #plt.title('Sound Wave of Clean Input')
    plt.title('Sound Wave of Input with Overlap Speech')
    plt.savefig(output_image)
    plt.close()

# Example usage
visualize_soundwave('overlap.wav', 'overlap.png')
