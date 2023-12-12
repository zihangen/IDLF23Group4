
#!/bin/bash

#196-122152-0014_radnerf_torso_smo.mp4
#IDLF23Group4/GeneFace/infer_out/May/pred_video/196-122152-0014_whitenoise_dns48_radnerf_torso_smo.mp4
#196-122152-0014_radnerf_torso_smo
#IDLF23Group4/GeneFace/infer_out/May/pred_video/196-122152-0014_whitenoise_enhanced_radnerf_torso_smo.mp4
# 196-122152-0014_whitenoise_enhanced_radnerf_torso_smo.mp4
export ORIGNAME='196-122152-0014_radnerf_torso_smo'
export NOISYNAME='196-122152-0014_whitenoise_noisy_radnerf_torso_smo'
export OUTNAME=new_noise_swapped
export VIDEODIR=/home/IDLF23Group4/GeneFace/infer_out/May/pred_video/
s=1 # set to 0 to calculate confidence of origname, set to 1 for noisyname
if [ $s -eq 1 ];
then
  python /home/IDLF23Group4/AudioProcessing/swapAudio.py \
    --orig ${VIDEODIR}${ORIGNAME}.mp4 --noisy ${VIDEODIR}${NOISYNAME}.mp4 --outpath ${VIDEODIR}${OUTNAME}.mp4
  cd /home/IDLF23Group4/syncnet_python/
  python /home/IDLF23Group4/syncnet_python/run_pipeline.py \
    --videofile ${VIDEODIR}${OUTNAME}.mp4 --reference ${OUTNAME} --data_dir /home/IDLF23Group4/syncnet_python/syncnet_outputs
  python /home/IDLF23Group4/syncnet_python/run_syncnet.py \
    --videofile ${VIDEODIR}${OUTNAME}.mp4 --reference ${OUTNAME} --data_dir /home/IDLF23Group4/syncnet_python/syncnet_outputs
else
  #echo "else"
  cd /home/IDLF23Group4/syncnet_python/
  python /home/IDLF23Group4/syncnet_python/run_pipeline.py \
    --videofile ${VIDEODIR}${ORIGNAME}.mp4 --reference ${ORIGNAME} --data_dir /home/IDLF23Group4/syncnet_python/syncnet_outputs
  python /home/IDLF23Group4/syncnet_python/run_syncnet.py \
    --videofile ${VIDEODIR}${ORIGNAME}.mp4 --reference ${ORIGNAME} --data_dir /home/IDLF23Group4/syncnet_python/syncnet_outputs
fi