# IDLF23Group4
## Setup  
Please follow the setup instructions in: https://github.com/yerfor/GeneFace to setup the conda environment required to run the baseline Geneface model called 'Geneface'. Note you might have to manually change some code in the repo cloned from Geneface to deal with updated dependencies that are not backwards compatible. There is at least one thing that you need to change at the time of this README:   Search for code segment 'LandmarksType._2D' in the github repo of geneface and change all of them to 'LandmarksType.TWO_D'.   
Please install the requirements for Syncnet model from https://github.com/joonson/syncnet_python/tree/6efbb1c305c23f47a62b09cf4215a8ac45e97d49 in the same 'Geneface' conda environment.  
Please follow the setup instructions in: https://github.com/facebookresearch/denoiser to setup another conda environment for denoising model called 'denoiser'.  
If you do not follow these steps, inference script would not run properly  
  
After you have installed the necessary dependencies, you could run:  
python infer.py --denoiser_model=<path_to_model.th_file> --input_dir=<path_to_input_audio_directory> --out_dir<path_to_the_/Geneface/data/raw/_directory> --geneface_dir<path_to_your_cloned_geneface_repo>  
then you should see outputted videos in Geneface/infer_out/<target_video_name>/pred_video/ folder  

## NOTES:  
If you are setting up the geneface repo for the first time, it will have to run training scripts for a target head video, which could run for a couple hours depending on the video length.(We processed a 4 minute video on a P100 GPU and it took around 4 hours)    

## Training:
To train the denoising model to your own specifications or finetune based on an existing model, follow the detailed instructions on https://github.com/facebookresearch/denoiser  

## Visualization  
Under the Visualize folder, there are scripts to visualize training results and a script called stack.py to compare frame by frame results of the videos.  

## Data Processing  
In AudioProcessing/ folder, there is a script called audioTransform to generate noisy input from processed data. use -h flag to see what arguments are needed. You need to create your own noise folder with some noise files to modify the input audio with. Please append the noise files' filename with white/overlap/background for proper processing.  

## Syncnet Score Calculation  
to calculate the Syncnet confidence score for one of the videos. you could use the calculate_syncnet.sh file in scripts. Change the following fields:  ORIGNAME='name of original video without extension'  
NOISYNAME='name of noisy video without extension'  
OUTNAME='output name'  
VIDEODIR='directory where the video and noisy video could be found'  

