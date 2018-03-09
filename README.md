# Synopsis

This project is currently under development. The project is aimed at doing real time facial 
recognition from a live video stream. Facial recognition code extract faces from video streams 
using Haarcascades, passes labeled face images to a face recognition classifier for training. 
Once trained the classifier can be used to monitor a real time video feed. Code is also being 
developed to communicate face classification data (primarily who was recognized) to a server in 
real time. 


# Brief Description of Files
## user_identification.py

The user_identification.py application is called from the command line in one of 3 modes
'-capture', '-train' or 'track'

### Capture Mode
In capture mode, the application captures face images and labels them with a known user ID 
found in file:

    'training_data/userID_key.csv'  

The userID_key assigns a unique integer to a user name. The application loads this information
creates a video stream and display window. Once the user who's face is to be captured is visible 
in the video feed and ready, with the display window in focus, press y. Then change focus to the
command terminal where you will be shown name number pairs. You will be prompted to enter the 
number corresponding with the name of the user who's face is being captured. 
The application then captures a number of face images, labels them with the assigned user and saves 
this data for later use in training. 

### Train Mode 
Called in train mode, the application uploads previously saved, labeled face data and passes
this to a face recognizer for training

### Track Mode
The application loads the previously trained recognizer, detects faces in a live video feed and
if they are recognized within a certain threshold, makes that user the current user. If the 
current user is different from a previous user, the application sends a message to a server 
indicating that the user has changed and transmits the new user information. 







## face_tracking_v3.py
The face_tracking.py and face_tracking2.py files in Face_Recognition_User_ID/Face_Recognition_User_ID 
include code for capturing faces from a video stream using Haarcascades for use in training. Code 
in user_identification.py includes code for uploading a pre-trained classifier, capturing faces 
from a live video feed, sending those faces to the classifier for identification. Code in nodestuff 
is written in node.js and is for communicating the facial recognition data between the python applications 
and a node.js server app. 

### Definition of Functions in face_tracking_v2
face_tracking_v2 can capture faces from a video file, use those images to train a face 
recognizer, then track faces in this or other files. The software frames recognized users (labels 0 or 1)
with a colored rectangle. The script saves a new video file with target faces framed with this rectangles


#### def main():
Call the script face_tracking_v2 from the command line. The arg, var1 can be either -capture
which captures images from the video and creates training data, or -track which does face 
recognition, tracking and which generates the live and saved video with rectangles of different 
colors around individuals labeled 0 or 1




#### vid_cap_write()
opens a connection to a connected camera, displays the video in a window
and writes that video to a file output.avi in the format fourcc XVID, 20 frames/s and 640/480

#### create_training_data(file_name = 'training_data2.csv')
This function opens a training data csv file which contains image numbers
and labels. These data are split into lists, face_img_names and face_labels
the code then reads each image, converts to gray scale and 
appends to face_imgs, a list of actual images
The function then returns the face_imgs and face_labels lists to be used for
training

#### train_face_recognizer(face_recognizer, faces, labels)
Takes faces and labels generated from create_training_data(file_name = 'training_data2.csv')
and use to train a face recognizer

#### capture_images_from_video(vid_name = '../images/TestV2.mp4')
generate images and labels from a video file. Code reads in video and frame by 
frame, finds faces in image and saves them to a file. The code also saves a 
list of images by name and labels that are generated using real time recognition. 
These labels are incorrect in many cases but give the munger a good start. This
also establishes the structure of the image name / labels data file. 

#### def save_training_images(face_imgs, img_count):
takes a list of images (of faces) and the number of images and saves the 
images to files

#### def save_training_labels(face_labels, img_names, file_name = 'training_data.csv'):
takes list of face labels, image names and saves them to a .csv file for 
use in training


#### def train_recognizer(face_imgs, face_labels):
trains a face recognizer using list of images, face_imgs and labels,
face_labels

#### def track_faces(face_cascade, face_recognizer, video_name = 'TestV1.mp4',  write = False ): 
Recognizes faces in a video file using the face detector and trained 
face recognizer. saves a new video which is basically the old video with 
colored rectangles drawn around faces labeled 0 or 1. Different labels
have different colors. The video is also displayed in a window during 
analysis. 

# Motivation

This code is being developed as an example real-time user identification that might be used by an 
entertainment or video game system. 

# Installation

Packages are still under development


# Tests

Tests are still under development

# Contributors

Rob Beetel, pretty much me so far. 

# License

None currently. 

