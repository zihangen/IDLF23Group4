# denoise
noisydir=$1
outdir=$2
cd denoiser
conda activate denoiser
python -m denoiser.enhance --dns48 --noisy_dir=$noisydir --out_dir=$outdir
