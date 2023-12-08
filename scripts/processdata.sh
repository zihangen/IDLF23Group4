ORIGPATH=/home/LibriSpeech/dev-clean
NOISESPATH=/home/IDLF23Group4/AudioProcessing/noises
snr=0
NOISETYPE=random
OUTPUTDIR=/home/IDLF23Group4/ProcessedData/small-valid
PROCESSEDDIR=
PARSEDCLEAN=
MAXLEN=10
python audioTransform.py \
  --noise_type ${NOISETYPE} --noise_files ${NOISESPATH} --clean_files ${ORIGPATH}\
  --snr ${snr} --output_dir ${OUTPUTDIR}  --max_len ${MAXLEN}
  #--parsed_clean=${PARSEDCLEAN} --parsed_noise=${PROCESSEDDIR}