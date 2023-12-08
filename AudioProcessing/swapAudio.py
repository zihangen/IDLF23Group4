from moviepy.editor import VideoFileClip, AudioFileClip
import argparse
# swaps the audio of two videos, we need to do this before calculating lsp score as we need
# lsp score takes in a video and calculates how closely the lipsync is matched with the audio
# if we use noisy audio, then the generated score is not reflective of how the real audio's lip movements
# thus we need to swap the audio of the talking head video generated with noisy audio with the audio of 
# the talking head video generated with clean audio input
def swap_audio(clean_file, noisy_file, output_file):
    # Load the video files
    clip1 = VideoFileClip(clean_file)
    clip2 = VideoFileClip(noisy_file)

    # Extract audio from both videos
    audio1 = clip1.audio

    # Swap audio tracks
    clip2_with_audio1 = clip2.set_audio(audio1)

    # Write the output files
    clip2_with_audio1.write_videofile(output_file)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "swap audio")
    parser.add_argument('--orig', type=str, default='', help='input clip path with clean audio')
    parser.add_argument('--noisy', type=str, default='input clip path with noisy audio')
    parser.add_argument('--outpath', type=str, default='', help='output file path')
    opt = parser.parse_args()
    swap_audio(clean_file=opt.orig, noisy_file=opt.noisy, output_file=opt.outpath)

