import os
import subprocess
import re, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--vid_dir", type=str, default='/home/IDLF23Group4/GeneFace/infer_out/May/pred_video', required=False)
parser.add_argument("--out_dir", type=str, default='/home/IDLF23Group4/outputs/', required=False)
args = parser.parse_args()
# Directory containing the video files
video_dir = args.vid_dir

# Function to extract the prefix from a filename
def extract_prefix(filename):
    match = re.match(r'(\d+-\d+-\d+)_.*\.mp4', filename)
    return match.group(1) if match else None
confidences = {} # {prefix:[]scores}
def run_command(command, prefix, postfix):
    try:
        result = subprocess.run(command, check=True, shell=True, text=True, capture_output=True)
        # Print the output
        confidence = result.stdout.splitlines()[-1]
        if prefix not in confidences and postfix == 'sync':
            confidence = confidence.split('\t')[1]
            confidences[prefix] = [confidence]
        elif postfix == 'sync':
            confidence = confidence.split('\t')[1]
            confidences[prefix].append(confidence)
    except subprocess.CalledProcessError as e:
        # Handle errors
        print("Error occurred:")
        print(e.stderr)

# Read all filenames in the directory
files = os.listdir(video_dir)
files = [f for f in files if f.endswith('.mp4')]
#print(files)

# Group files by their prefixes
prefixes = {}
for file in files:
    prefix = extract_prefix(file)
    if prefix:
        prefixes.setdefault(prefix, []).append(file)

#print(prefixes)
# Process each group of files with the same prefix
for prefix, matched_files in prefixes.items():
    if len(matched_files) >= 2:
        print(matched_files)
        # Find the original and noisy files
        original_file = None
        noisy_files = []
        for f in matched_files:
            #print("f", f)
            if 'enhanced' in f or 'noisy' in f:
                print(f)
                noisy_files.append(f)
            else:
                original_file = f
        for f in noisy_files:
            # run calc_syncnet.sh
            swap_audio_cmd = f'python /home/IDLF23Group4/AudioProcessing/swapAudio.py --orig {video_dir}/{original_file} --noisy {video_dir}/{f} --outpath {video_dir}/new_noise_swapped.mp4'
            run_command(swap_audio_cmd, prefix, 'swap')
            syncnet_dir = '/home/IDLF23Group4/syncnet_python'
            os.chdir(syncnet_dir)
            run_pipeline_cmd = f'python run_pipeline.py --videofile {video_dir}/new_noise_swapped.mp4 --reference new_noise_swapped --data_dir {syncnet_dir}/syncnet_outputs'
            run_syncnet_cmd = f'python run_syncnet.py --videofile {video_dir}/new_noise_swapped.mp4 --reference new_noise_swapped --data_dir {syncnet_dir}/syncnet_outputs'
            run_command(run_pipeline_cmd, prefix, 'pipe')
            run_command(run_syncnet_cmd, prefix, 'sync')
print("finished")
for k in confidences:
    print(k, ':', confidences[k])