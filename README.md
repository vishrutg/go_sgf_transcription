# Go SGF Transcription
Given a video of a game of go, create an sgf file representing the game. There are three major steps each described in the sections below

## Lattice Estimation
Given an image or a video that is rectified, the board pose must be estimated relative to the camera. Some initial estimates for the relative pose are estimated and then further refined by a brute force search. There are some assumptions made such as the board width, board height, board line spacing width, board line spacing height, stone diameter etc. found from https://senseis.xmp.net/?EquipmentDimensions

#### Line Detection
Using canny edge detector, find the lines in the image. There are several parameters that need to be fine-tuned depending on image size and camera pose. 
#### Estimate relative pose z rotation
Given a set of lines in the image from the previous step, find 2 sets of N equi-distant parallel lines where each set of parallel lines are perpendicular to each other. These are the grid lines of the board and N is the board size (usually 19). It is assumed that the camera is not looking at the board at an oblique angle so the lines should be roughly perpendicular. The z rotation of the board relative to the camera can be estimated by the angle estimated lines. 
#### Estimate relative pose z and board center point in the image
We can also estimate how far the board is from the camera by using the distance between parallel lines in the image and the known distance between grid lines on the board to calculate. We can also find the middle of the board by finding the center line of both sets of parallel lines and their intersection.  
#### Estimate relative pose x rotation, relative pose y rotation
It was earlier assumed that the camera is not looking obliquely at the board but some small angles can be accomodated. Using some epipolar geometry, we can estimate the board tilt along both axes using the 2 sets of parallel lines calculated in the previous steps. In practice, these estimates are extremely noisy since the noise from previous steps compounds
#### Estimate relative pose x, relative pose y
We previously estimated where the board center point was in the image and given all the other estimated parameters, we can estimate the exact position of the camera using back project. However it was empirically found to be very noisy and as a result, a more complicated procedure has to be done. A board grid lines correlation filter is generated and applied to the original line detection results.  
#### Brute Force Refinement
The previous steps give an initial estimate but are not precise enough. Given the initial estimate of the camera pose relative to the board, we can grid search a set of poses close the initial estimation of the pose. This results in a much better pose at a relatively cheap cost. The brute force matching is done by rendering an estimate of what the board grid lines would look line given the candidate relative camera pose and how closely they match the lines found frmo the line detection step. 
#### Obstruction Map
There are many images where part of the board is obstructed by player arms. It is necessary to estimate what part of the board is obstructed given any image. This is currently done by checking if the expected grid lines are not present after running the image through the line detection step. 
### Code Examples
The lattice parameters can be estimated on the test data provided by running the following:

`python3 lattice_estimation.py --video_path=data/test_video.mp4 --board_size=19 --num_x_bins=3 --num_y_bins=3 --num_z_bins=3 --num_x_rot_bins=3 --num_y_rot_bins=3 --num_z_rot_bins=3 --x_search_range_mm=20 --y_search_range=20 --z_search_range_mm=20 --z_rot_search_range_degrees=3 --debug --train_lattice_estimation --lattice_estimation_parameters_path=data/test_video_lattice_estimation.npy`

The following command runs the lattice estimation on a single image and provides better debug results:

`python3 lattice_estimation.py --img_path=data/test_video_00001.jpg --board_size=19 --num_x_bins=3 --num_y_bins=3 --num_z_bins=3 --num_x_rot_bins=3 --num_y_rot_bins=3 --num_z_rot_bins=3 --x_search_range_mm=20 --y_search_range=20 --z_search_range_mm=20 --z_rot_search_range_degrees=3 --debug --train_lattice_estimation --lattice_estimation_parameters_path=data/test_img_lattice_estimation.npy`

These commands will save the lattice parameters to `data/test_video_lattice_estimation.npy` or `data/test_img_lattice_estimation.npy`. The num_< param >_bins are parameters for the brute force refinement step and larger values will produce a more accurate result. 

Future runs can re-use the lattice results with the following command:
`python3 lattice_estimation.py --video_path=data/test_video.mp4 --debug --lattice_estimation_parameters_path=data/test_video_lattice_estimation.npy`

### Todo
1. Test with 9x9 boards
2. Test with canny line detection with different camera params
3. Improve the obstruction map and test if it works when board is full of stones

## Board Estimation
Using the estimated relative pose of the camera, the pixel rgb values at the various grid intersection points can be used to check if there is a black or white stone there. However because there can be slight errors in localizing camera position, the grid lines can be slightly off. As a result instead of looking at the rgb values of the exact grid intersection point, instead look at the rgb values in a circle around the grid intersection point and combine the rseults to get an overall probability of black or white stone. This also solves the problem of if the stone is not properly centered on the grid line.
#### Color and Lighting Estimation
The black and white stone colors are estimated. This has to be done because various cameras have different lighting parameters etc. The board lighting is also estimated because some parts of the board sometimes have very different brightness.  
#### Montecarlo Sampling
Combining the color estimates of all the points in a circle around the estimated grid intersection point is not trivial and is easier to use monte carlo method. These values must be precalculated.  
### Code Examples
This will train the various parameters:

`python3 board_estimation.py --video_path=data/test_video.mp4 --board_size=19 --lattice_estimation_parameters_path=data/test_video_lattice_estimation.npy --board_estimation_parameters_path=data/test_video_board_estimation.npy --train_board_estimation`

This command can be used to view the result:

`python3 board_estimation.py --video_path=data/test_video.mp4 --board_size=19 --lattice_estimation_parameters_path=data/test_video_lattice_estimation.npy --board_estimation_parameters_path=data/test_video_board_estimation.npy --debug`
### Todo
1. Improve the obstruction map
2. Speed up code or look into better approximation

## Move Estimation
### Code Examples
### Todo
