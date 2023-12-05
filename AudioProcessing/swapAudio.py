from moviepy.editor import VideoFileClip, AudioFileClip
# swaps the audio of two videos, we need to do this before calculating lsp score as we need
# lsp score takes in a video and calculates how closely the lipsync is matched with the audio
# if we use noisy audio, then the generated score is not reflective of how the real audio's lip movements
# thus we need to swap the audio of the talking head video generated with noisy audio with the audio of 
# the talking head video generated with clean audio input
def swap_audio(mp4_file1, mp4_file2, output_file2):
    # Load the video files
    clip1 = VideoFileClip(mp4_file1)
    clip2 = VideoFileClip(mp4_file2)

    # Extract audio from both videos
    audio1 = clip1.audio

    # Swap audio tracks
    clip2_with_audio1 = clip2.set_audio(audio1)

    # Write the output files
    clip2_with_audio1.write_videofile(output_file2)

# Example usage
f = "whitenoise"
swap_audio("/Users/bli/Desktop/dlproject/7850-111outvods/vods/aorig.mp4", f"/Users/bli/Desktop/dlproject/7850-111outvods/vods/{f}.mp4", 
            f"{f}.mp4")
