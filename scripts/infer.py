import sys, os, subprocess
import argparse
def run_command(command):
    try:
        result = subprocess.run(command, check=True, shell=True, text=True, capture_output=True)
        # Print the output
        print("Command output:")
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        # Handle errors
        print("Error occurred:")
        print(e.stderr)
    
parser = argparse.ArgumentParser()
parser.add_argument("--denoise_pretrain", type=bool, default=False, required=False, help='use pretrained denoise model')
parser.add_argument("--denoiser_model",type=str,required=True, default='./models/finetune.th', help='path to denoiser model')
parser.add_argument("--input_dir", required=True, type=str, default='', help="input dir to denoiser")
parser.add_argument("--out_dir", required=False, type=str, default='/home/IDLF23Group4/GeneFace/data/raw/val_wavs')
parser.add_argument("--geneface_dir", required=False, type=str, default="/home/IDLF23Group4/GeneFace")
args = parser.parse_args()
print(args)
#subprocess.run("ls")
if (args.denoise_pretrain):
    denoise = f"python -m denoiser.enhance --dns48 --noisy_dir={args.input_dir} --out_dir={args.out_dir}"
else:
    denoise = f"python -m denoiser.enhance --model_path={args.denoiser_model} --noisy_dir={args.input_dir} --out_dir={args.out_dir}"
# Combine commands into a single command string
conda_location = '/opt/conda/lib/python3.7/site-packages/conda/shell/etc/profile.d/conda.sh'
init_conda = f'. {conda_location}'
denoiser_cmd = f'''
    {init_conda} &&
    conda activate denoiser &&
    cd denoiser &&
    {denoise} && 
    conda deactivate
    '''
run_command(denoiser_cmd)
video_id='May'
postnet_ckpt_steps=4000
geneface_pythonpath = args.geneface_dir
postnet_cmd = f'''
    {init_conda} &&
    conda activate geneface &&
    cd {args.geneface_dir} &&
    export PYTHONPATH={geneface_pythonpath} &&
    export CUDA_VISIBLE_DEVICES=0 &&
    export Video_ID={video_id} &&
    export Postnet_Ckpt_Steps={postnet_ckpt_steps} &&
'''

radnerf_cmd = f'''
    {init_conda} &&
    conda activate geneface && ls &&
    cd {args.geneface_dir} &&
    export CUDA_VISIBLE_DEVICES=0 &&
    export PYTHONPATH={geneface_pythonpath}&&
    export Video_ID={video_id} &&
'''
# out_dir contains noise files
for i, file in enumerate(os.listdir(args.out_dir)):
    print("filename in dir", file)
    if '.wav' in file and '_16k' not in file:
        wav_id = os.path.splitext(file)[0]
        postpend = ' &&'
        if i == len(os.listdir(args.out_dir)) - 1:
            postpend = ''
        postnet_cmd += f'''
            python inference/postnet/postnet_infer.py --config=checkpoints/{video_id}/lm3d_postnet_sync/config.yaml --hparams="infer_audio_source_name=data/raw/val_wavs/{wav_id}.wav,infer_out_npy_name=infer_out/{video_id}/pred_lm3d/{wav_id}.npy,infer_ckpt_steps={postnet_ckpt_steps}" --reset{postpend}
        '''
        radnerf_cmd += f'''
            python inference/nerfs/lm3d_radnerf_infer.py --config=checkpoints/{video_id}/lm3d_radnerf_torso/config.yaml --hparams="infer_audio_source_name=data/raw/val_wavs/{wav_id}.wav,infer_cond_name=infer_out/{video_id}/pred_lm3d/{wav_id}.npy,infer_out_video_name=infer_out/{video_id}/pred_video/{wav_id}_radnerf_torso_smo.mp4" --infer{postpend}
        '''
#print(postnet_cmd)
run_command(postnet_cmd)
run_command(radnerf_cmd)
