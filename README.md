# go_sgf_transcription
There are three major steps for transcription:
1. lattice estimation
2. board estimation
3. move estimation

The first step lattice estimation is to figure out where the board is relative to the camera. This step works on a single image but the assumption is that the board is not moving relative to the camera so we can also average across the entire video for more accuracy. 
The second step board estimation is to determine where the stones are on the board. This step will estimate the location of all the stones in every image of the video. However given that many images have obstructions such as hands, there are some challenges with generating an sgf from just running board estimation on a sequence of images in the video
The last step is to estimate a given move. This step will compare the board estimates of two consecutive images and figure out which move was made between them. It is assumed that the camera captures one image per clock button press however there are some attempts to error correct for missing moves etc. Currently, this step doesn't succeed all the time. 

## Installation
```
sudo apt install libopencv-dev python3-opencv
pip install -r requirements.txt
```

## Lattice Estimation
This step estimates the location of all the grid points. It will average over the entire video so if the camera is moving relative to the board there would be issues. It is fairly computationally intensive because there are a lot of frames in the video where hands are obstructing the board. 

This is mostly a debugging tool to see what the lattice estimation step results are:
```
python lattice_estimation.py \
--video_path=data/test_video.mp4 \
--train_lattice_estimation \
--lattice_estimation_parameters_path=data/test_video_lattice_estimation.npy \
--debug
```

## Board Estimation
This step estimates the stone colors and lighting conditions after the lattice has been estimated. There were some issues with arm skin tones being similar color as the board causing false detections etc so this is more complicated then necessary. If there is sufficient data, it would make more sense to convert this step into a neural net. However given the lack of data, it is doing parameter estimation for a modified version of gaussian mixture models. The parameters can actually be hard coded assuming a standard board and it will likely work fine. 

This is mostly a debugging tool to see what the board estimation step results are:
```
python board_estimation.py \
--video_path=data/test_video.mp4 \
--lattice_estimation_parameters_path=data/test_video_lattice_estimation.npy \
--board_estimation_parameters_path=data/test_video_board_estimation.npy \
--train_board_estimation \
--debug
``````

## Move Estimation
This step will run the lattice estimation and board estimation steps internally and not use any saved results as those are for debugging. There is an attempt to recover from players not hitting the clock after their move however there is limited success. This step doesn't quite work and is still a work in progress.  

This is the main entrypoint for the transcription:
```
python move_estimation.py --video_path=data/test_video.mp4
```