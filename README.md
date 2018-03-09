# Synopsis

The project is aimed at doing real time facial 
recognition from a live video stream. Facial recognition code extract faces from video streams 
using Haarcascades, passes labeled face images to a face recognition classifier for training. 
Once trained the classifier can be used to monitor a real time video feed. Code is also being 
developed to communicate face classification data (primarily who was recognized) to a server in 
real time. 


# Description of Application

## user_identification_v2.py

The user_identification.py application is called from the command line in one of 3 modes
'-capture', '-train' or 'track'

The application in -capture mode captures face images for a users to be recognized. In -train mode
those images are used to train a face recognition algorithm. Finally, when called in -track
mode the app will recognize user, frame the users face and display the users name. The application
can also send an update to a remote server to let the server know the current user has changed. 
The remote server could be any application where user ID is tracked. 



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
this to a face recognizer for training. The recognizer is saved as 'model/face_recognizer.yml'. 


### Track Mode
The application loads the previously trained recognizer, detects faces in a live video feed and
if they are recognized within a certain threshold, makes that user the current user. If the 
current user is different from a previous user, the application sends a message to a server 
indicating that the user has changed and transmits the new user information. 




### Definition of Functions in user_identification_v2.py


#### def main():
Call the script user_identification_v2.py from the command line. The arg, var1 can be either -capture
which captures images from the video and creates training data, -train which loads the training data
and trains a facial recognition algorithm or -track which does face recognition, tracking and which 
generates the live and saved video with rectangles of different colors around individuals 
labeled 0 or 1


#### def create_training_data(file_name = 'training_data/userID_training_data.csv')
import training data from file and format in way to pass to the train_recognizer()
function. This function opens a training data csv file which contains image numbers
and labels. These data are split into lists, face_img_names and face_labels
the code then reads each image, converts to gray scale and 
appends to face_imgs, a list of actual images
The function then returns the face_imgs and face_labels lists to be used for
training


#### def train_recognizer(face_imgs, face_labels, userID_model)
import and train face recognizer using previously generated training data


#### def import_face_detector()
import haarcascades face detector


#### def import_face_recognizer(userID_model = 'model/face_recognizer.yml' )
import previously trained face recognizer


#### def capture_faces(cap, face_detector, face_imgs, face_labels, 
                                    face_label_names, userID_dict)
generate images and labels from a video file. Code reads in video and frame by 
frame, finds faces in image and saves them to a file. The code also saves a 
list of images by name and labels that are generated using real time recognition. 
These labels are incorrect in many cases but give the munger a good start. This
also establishes the structure of the image name / labels data file. 


#### def track_faces(cap, face_detector, face_recognizer, userID_dict, 
                                        monitorID, url, log_file):
track faces in real time video. If a user is recognized an update signal is 
sent to a server. 


#### def save_training_labels(face_labels, img_count, file_name = 'userID_training_data.csv')
takes list of face labels, image names and saves them to a .csv file for 
use in training


#### def save_training_images(face_imgs, img_count)
save training images. 


#### def user_change_alert(userID, monitorID, url, log_file, frame)
called by track faces when a new user is recognized. sends a message to a 
server at the given url with information about the new user. 


# Brief Description of Additional Files

## training/userID_key.csv

This file contains the user name / user ID pairs which define the users to be recognized

    0 Azita
    1 Rob 
    2 Cheryl
etc

## training/userID_training_data.csv

This file contains face image file names along with their user ID labels. 
Image Name  Label    
img_0.png     2
img_1.png     2
    .
    .
img_99.png    0
img_99.png    0

## model/face_recognizer.yml

This file contains the trained face recognizer.

## haarcascades/haarcascade_frontalface_alt.xml

This file contains the face detector. There are other face detector algorithms available
including: 
    haarcascade_frontalface_alt_tree.xml
    haarcascade_frontalface_alt2.xml


## log/userID_log.log

This is the programs log file. 






# Motivation

This code is being developed as an example real-time user identification that might be used by an 
entertainment or video game system. 

# Installation

Installation packages are still under development


# Tests

Tests are still under development

# Contributors

Rob Beetel, pretty much me so far. 

# License

None currently. 

