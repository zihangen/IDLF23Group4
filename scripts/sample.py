#sample
import argparse
import subprocess

def run_command(command):
    try:
        # Execute the combined command
        result = subprocess.run(command, check=True, shell=True, text=True, capture_output=True)
        
        # Print the output
        print("Command output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # Handle errors
        print("Error occurred:")
        print(e.stderr)

# Set up argument parser
parser = argparse.ArgumentParser(description='Run multiple shell scripts with different environments.')
parser.add_argument('--noisy_dir', type=str, required=True, help='Directory containing noisy audio files')
parser.add_argument('--out_dir', type=str, required=True, help='Directory to output enhanced audio files')
parser.add_argument('--model_path', type=str, required=True, help='Path to the denoiser model file')

# Parse arguments
args = parser.parse_args()

# Initialize Conda for script (modify this line based on your Conda installation path)
init_conda = 'source ~/miniconda3/etc/profile.d/conda.sh'

# Denoiser script
denoiser_cmd = f'''
    {init_conda} &&
    conda activate denoiser &&
    cd denoiser &&
    python -m denoiser.enhance --model_path={args.model_path} --noisy_dir={args.noisy_dir} --out_dir={args.out_dir}
    '''
run_command(denoiser_cmd)

# Geneface environment and scripts
geneface_cmd = f'''
    {init_conda} &&
    conda activate geneface &&
    bash scripts/infer_postnet.sh &&
    bash scripts/infer_lm3d_radnerf.sh
    '''
run_command(geneface_cmd)

# Calculate Syncnet environment and script
syncnet_cmd = f'''
    {init_conda} &&
    conda activate calculate_syncnet &&
    /scripts/calculate_syncnet_swap.sh
    '''
run_command(syncnet_cmd)
