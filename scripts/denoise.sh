# denoise
noisydir=$1
outdir=$2
modelpath=$3
cd denoiser
python -m denoiser.enhance --model_path=$modelpath --noisy_dir=$noisydir --out_dir=$outdir
