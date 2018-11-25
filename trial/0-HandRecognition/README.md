# Experiment - Hand Recognition

Basic experiment to implement hand recognition to compare timing/speed with another 
implementation from 0-OpenCVIntegration.

This experimental attempt is  based on the tutorial from: https://gogul09.github.io/software/hand-gesture-recognition-p1

Author of tutorial: Gogul

Note - excuse the mess of a class, this is just a rough implementation

## Usage
* Run the script.
* WAIT until '30 frames' has passed before attempting to add your hand in the green
target box. This is because the script needs to first average the background, if the
background keeps changing (such as a result of a moving hand) then the background average
will naturally be erratic.
* Once the threshold window opens (the grayscaled smaller window), then you can show the
camera your hand and observe the red contours around it!

