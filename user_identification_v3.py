import numpy as np
import cv2
from matplotlib import pyplot as plt
import requests
import csv
import os
import sys
import time
import logging




'''************************************************************************
import training data from file and format in way to pass to the train_recognizer()
function
************************************************************************'''
def create_training_data(file_name = 'training_data/userID_training_data.csv'):
 
    face_img_names = []
    face_imgs = []
    face_labels = []

    print("open training data")
    try:
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            next(reader, None)
            for row in reader:

                split_row = row[0].split(',')
                #print(split_row)
                face_img_names.append(split_row[0])
                face_labels.append(int(split_row[1]))
                print(split_row[0] + " " + split_row[1])
            #print(face_img_names)
            #print(face_labels)
        csvfile.close()
    except OSError as e: 
        print(e)
        logging.error(e)
        return None, None
    
    
    
    filename = 'images/' + face_img_names[0]
    try: 
        img = cv2.imread(filename, 0)
        
        
    except OSError as e:
        print(e)
        logging.error(e)
        return None, None
        
    cv2.imshow('frame', img)
    print(img.shape)
    
    
    for name in face_img_names:
        filename = 'images/' + name
        print(filename)
        try: 
            img = cv2.imread(filename, 0)
        except OSError as e:
            print("Error loading image file:")
            print(e)
            logging.error("Error loading image file {}".format(e))
            return None, None
        # convert BGR image to grayscale
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        img.resize(200,200)
        print(img.shape)
        face_imgs.append(img)
     
   
    print("we have this many {} image files".format(len(face_imgs)))
    
   
    
    return face_imgs, face_labels
    
    
    
 
'''************************************************************************
import and train face recognizer using previously generated training data
************************************************************************'''
def train_recognizers(face_recognizers, face_imgs, face_labels, userID_models):

    
    for recognizer in face_recognizers.keys():
        try: 
            face_recognizers[recognizer].train(face_imgs, 
                                                   np.array(face_labels))
        except cv2.error:
            logging.error(e)
            print(e)
            return None
        
        face_recognizers[recognizer].write(userID_models + recognizer + '.yaml')

       
        '''
        count = 0
        for img in face_imgs:
            label, confidence = face_recognizers[recognizer].predict(img)
            print("{} predicted = {} with confidence = {}  actual = {}".format(recognizer, 
                                              label, confidence, face_labels[count]))
            count+=1 
  
        '''
    #label, confidence = face_recognizer.predict(img)
    return face_recognizers




'''************************************************************************
import haarcascades face detector
************************************************************************'''
def import_face_detector():
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    print(face_cascade)
    return face_cascade

'''************************************************************************
import face recognizer
************************************************************************'''
def import_face_recognizer(userID_model, face_recognizers): 
    
    for key in face_recognizers.keys():
        recognizer_file = userID_model + key + '.yaml'
        print(recognizer_file)
        face_recognizers[key].read(recognizer_file)
    
    return face_recognizers
 
# ADD CODE FOR IMPORTING PRE-TRAINED RECOGNIZERS 
# COULD IMPORT FACE AND LABELS



'''************************************************************************
capture faces from real time video for use in training the face recognizer

************************************************************************'''
def capture_faces(cap, face_detector, face_imgs, face_labels, 
                                    face_label_names, userID_dict, size_threshold):
                                        
    # parameters for how many frames to capture and how often
    nth_frame = 1
    num_images = 49
    # number of frames to capture
    num_frames = nth_frame * num_images
    
    # number of consecutively captured frames
    capturing = False
    captured_frames = 0
    
    new_user_msg = "Press (y) when you are ready to begin capture\n and move to terminal"
    
    capturing_msg = "Please wait while I learn your face...."
    
    
    cv2.namedWindow('Training', cv2.WINDOW_NORMAL)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      
    
    # code for writing label onto image
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    # WILL ADD BOTTOM CORNER IN LOOP
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    
    
    
    print("Beginning Capture Phase")
    # capture images from video feed
    
    
    while(True): 
    #for i in range(num_frames): 
        
        ret, frame = cap.read()
        #print("captured frame")
        
        #operate on the frame
        # converting to grayscale for recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  
        # find faces in image
        faces = face_detector.detectMultiScale(gray)
        # get bounding box for each detected face
        # check for number of faces first? 
        for (x,y,w,h) in faces:
            
            # FOR FACES ABOVE CERTAIN SIZE
            # frame images using a rectangle, providing feedback to user
            # capturing every 10th frame to create delay in capture to get
            # a variety of image examples. 

            print("Faces Detected")
            if ((w > size_threshold) & (h > size_threshold)):
                # 
                # ADD IMAGE FROM EVERY 10TH FRAME
                if capturing:
                    img = gray[y:y+w, x:x+h]
                    face_imgs.append(img)
                    face_labels.append(user)
                    # if we've captured all frames, reset variables
                    # and increment user number for next training session. 
                    if captured_frames >= num_frames:
                        capturing = False
                        captured_frames = 0
                        print("\n\n\n")
                        print("Lemme know if you want to capture more faces?")
                        #next_user_label += 1
                    else:
                        captured_frames += 1
                        print("Captured {} of {} frames".format(captured_frames, 
                                num_frames))
                    
                # add bounding box to color image
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
                # ADD MESSAGE TO TRAINING IMAGE
                topLeftCornerOfText = (x-20,y)
                bottomLeftCornerOfText = (x-40, y+h+30)
                if not capturing:
                    msg = new_user_msg
                else: 
                    msg = capturing_msg
                cv2.putText(frame, msg, 
                                topLeftCornerOfText, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)
        
        cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        '''
        new_height = int(frame_height/2)
        new_width = int(frame_width/2)
        resize = cv2.resize(frame, (new_width, new_height))
        '''
        resize = frame
        #print("image {} resized".format(i))
        
        cv2.imshow('Training', resize)

        keypress = cv2.waitKey(1)
        if ((keypress & 0xFF == ord('y')) and (capturing == False)):
            """       
            user_label = input("enter user number in terminal")
            user_name = input("enter name in terminal")
            face_label_names.append(user_name)
            #face_labels.append(user)
            """
            print("Capturing images for a known user")    
            print("Please enter the number corresponding to user")
            print("\n")
        
            for key in userID_dict.keys():
                print(key + ": " + userID_dict[key])
         
            user = input()
            print("Capturing images of {}".format(userID_dict[user]))
        
            capturing = True
            captured_frames = 0
            print("Adding New User")
        
        elif keypress & 0xFF == ord('q'):
            break
        else: 
            pass
            
    print("New User Added")   
    cv2.destroyWindow("Training")
    cv2.waitKey(1)
    return face_imgs, face_labels




'''************************************************************************
track faces in real time video. If a user is recognized an update signal is 
sent to a server. 
************************************************************************'''
def track_faces(cap, face_detector, face_recognizers, userID_dict, 
                                        monitorID, url, log_file, 
                                        size_threshold, classification_confidence):
    
     # CREATE LISTS FOR IMAGES AND LABELS
    
    #classificationConfidence = 75
    
    # create video stream from camera 1
    #cap = cv2.VideoCapture(1)
    
    # create display window
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    
    # EXTRACT INFORMATION ABOUT VIDEO INPUT
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    
    
    # CODE FOR WRITING LABEL ONTO IMAGE
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    # WILL ADD BOTTOM CORNER IN LOOP
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    
    
    # INITIALIZE TO UNKNOWN USER
    unknownUser = "Unknown User"
    newUserMsg =  "Looks like you're new! Would you like to add your profile? Press y for Yes"        
      
    # initialize user
    user = 0  
    label = unknownUser
    prev_user = None
    current_users = []
    
    
    # capture and display video feed
    while(True):
        #capture frame_by_frame
        ret, frame = cap.read()
        #operate on the frame
        # converting to grayscale for recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          # find faces in image
        #print("detecting face")
        
        faces = face_detector.detectMultiScale(gray)
        
        #print("faces detected")
        # get bounding box for each detected face
        # check for number of faces first? 
        if len(faces) > 0:
            current_users = []
            for (x,y,w,h) in faces:
                if ((w > size_threshold) & (h > size_threshold)):
                
                    # add bounding box to color and resize image
                    img = cv2.resize(gray[y:y+w, x:x+h],(200,200))
                    
            
                    #print("Classifying face")
                    for key in face_recognizers.keys():
                        label_num, confidence = face_recognizers[key].predict(img)
                        label_num = str(label_num)
                        print("{}   {}  with confidence of {}".format(key, 
                                               userID_dict[label_num], confidence))
                        
                        '''
                        print("{0}   {1}  with confidence of {1:2.2f}".format(key, 
                                               userID_dict[label_num], confidence))
                        '''
                        label_num = str(label_num)
                    
                    print("\n \n")
                    '''
                    
                    if ((label_num in userID_dict.keys()) and 
                                    (confidence < classification_confidence)):
                        print("{0}  with confidence of {1:2.2f}".format(userID_dict[label_num], confidence))
                        current_users.append(userID_dict[label_num])
                        
                        
                    else:
                        pass
                        #label = unknownUser
                    '''
                        
            if len(current_users) != 0:
                if prev_user in current_users: 
                    label = prev_user
                else: 
                    label = current_users[0]
                    user_change_alert(label, monitorID, url, log_file, frame)
                    prev_user = label
            else:
                label = prev_user
                 
                    
            # IF FACE IS RECOGNIZED, FRAME IT IN SOME COLOR, 
            # ELSE, FRAME IT IN WHITE
            #topLeftCornerOfText = (x,y)
            #bottomLeftCornerOfText = (x, y+h+30)
            if label == unknownUser:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
                msg = newUserMsg
                topLeftCornerOfText = (0 + 20, frame_height - 50)
                bottomLeftCornerOfText = (0+ 20, frame_height - 30)
        
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
                msg = label
                topLeftCornerOfText = (x,y)
                bottomLeftCornerOfText = (x, y+h+30)
            
            cv2.putText(frame,msg, 
                        topLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
        
            
        # convert BGR image to RGB for plotting
        cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_height = int(frame_height/2)
        new_width = int(frame_width/2)
        resize = cv2.resize(frame, (new_width, new_height)) 
        cv2.imshow('Tracking', resize)
        keypress = cv2.waitKey(1)
        if keypress & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


'''************************************************************************
takes list of face labels, image names and saves them to a .csv file for 
use in training
************************************************************************'''
def save_training_labels(face_labels, img_count, file_name = 'userID_training_data.csv'):
    #if os.path.isfile(file_name):
    if img_count == 0:
        f = open(file_name, 'w')
        writer = csv.writer(f)
        writer.writerow(('Image Name', 'Label'))
    else: 
        f = open(file_name, 'a')
        writer = csv.writer(f)
        
    for count in range(len(face_labels)):
        img_name = 'img_' + '{}'.format(img_count) + '.png'  
        writer.writerow((img_name, face_labels[count]))
        img_count += 1
        
    f.close()
    return img_count
    print("labels saved")


'''************************************************************************
save training images. 
************************************************************************'''
# NAME AND SAVE IMAGES
def save_training_images(face_imgs, img_count):
    
    print("face images = {}".format(len(face_imgs)))    
    for img in face_imgs:
        img_name = 'img_' + '{}'.format(img_count) + '.png'
        cv2.imwrite('images/' + img_name, img)
        #img_names.append(img_name)
        img_count += 1
    print("images saved")
    
'''************************************************************************
called by track faces when a new user is recognized. sends a message to a 
server at the given url with information about the new user. 
************************************************************************'''
def user_change_alert(userID, monitorID, url, log_file, frame):
    """
    files ={'myImage':('myImage.jpg', open('images/Rob1.jpg','rb'), 'image/jpg')
            
            }
    """   
    print("posting Request")
    print(userID)
    #r = requests.post(url,files=files)
    print("Request posted")
    #print(r)
    img_file_name = time.strftime("log/%m%d%y_%H%M%S_"+ userID + ".png")
    cv2.imwrite(img_file_name, frame)
    logging.info("New User: {}".format(userID))
    
    
    return True




def main():

    
    # PUT A TRY HERE 
    #vars = str(sys.argv)
    script, var1, cam = sys.argv
    print(var1)
    
    # logging configuration
    logging.basicConfig(format = '%(asctime)s %(levelname)s:%(message)s', 
                                level=logging.DEBUG, filename = 'log/userID_log.log')
    
    userID_keys = 'training_data/userID_key.csv'
    userID_data = 'training_data/userID_training_data.csv'  
    userID_models = 'model/' 
    log_file = 'log/log_file.txt'
    userID_dict = {}
    
    
    
    url = "http://127.0.0.1:3000"
    monitorID = 0
    
    cam = int(cam)

    # size threshold for a face in number of pixels
    size_threshold = 75
    classification_confidence = 60
    
    

    if ((var1 == '-capture') or (var1 == '-track')):
        log_msg = "Creating video stream"
        print(log_msg)
        logging.info(log_msg)
         # create video stream from camera 1
        cap = cv2.VideoCapture(cam)
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920.0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080.0)

        log_msg = "Importing face detector"
        print(log_msg)
        logging.info(log_msg)
        # IMPORT FACE DETECTOR AND RECOGNIZER
        log_msg = "Importing Haarcascades face detector"
        print(log_msg)
        logging.info(log_msg)
        face_detector = import_face_detector()
        
        log_msg = "Importing User ID Info"
        print(log_msg)
        logging.info(log_msg)
        try: 
            with open(userID_keys, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter = ' ', quotechar='|')
                for row in reader: 
                    user_label, user_name = row[0].split(',')
                    userID_dict[user_label] = user_name
        except OSError as e: 
            print("Error loading User ID info from {}, {}".format(userID_keys, e))
             
        
    if var1 == '-capture':
        
        log_msg = "Called in '-capture' Mode"
        print(log_msg)
        logging.info(log_msg)
        face_imgs = []
        face_img_names = []
        face_labels = []
        face_label_names = []
        img_count = 0
        
        if os.path.isfile(userID_data):
            with open(userID_data, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                next(reader, None)
                for row in reader:
                    img_count += 1
            log_msg = "Adding to Existing User Data"
            print(log_msg)
            logging.info(log_msg)
        else:
            log_msg = "Previous data does not exist. Creating new user data"
            print(log_msg)
            logging.info(log_msg)
            
        # call capture faces with the video capture, face detector and 
        # lists of images, image names, user_id numbers and user _id names
        
        log_msg = "Capturing faces"
        print(log_msg)
        logging.info(log_msg)
        face_imgs, face_labels = capture_faces(cap, face_detector, 
                            face_imgs, face_labels, face_label_names, 
                            userID_dict, size_threshold)
        
        cap.release()
        
        
        save_training_images(face_imgs, img_count)
        img_count = save_training_labels(face_labels, img_count, userID_data)
        
        
        
        print("Aaaaaand done.")
        
        
    elif var1 == '-train':
        
        log_msg = "Called in '-train' Mode"
        print(log_msg)
        logging.info(log_msg)
        
        
         
        face_imgs, face_labels = create_training_data(userID_data)
        if face_imgs == None:
            print("There was a problem importing and creating traininig data")
            print("Stopping application.")
            sys.exit(0)
        

 
        LBPHF_face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        eigen_face_recognizer = cv2.face.EigenFaceRecognizer_create()
        fisher_face_recognizer = cv2.face.FisherFaceRecognizer_create()

        face_recognizer_models = [LBPHF_face_recognizer, eigen_face_recognizer, 
                            fisher_face_recognizer]
        face_recognizer_keys = ['LBPHF', 'Eigen', 'Fisher']
        face_recognizers = dict(zip(face_recognizer_keys,face_recognizer_models)) 
        face_recognizers = train_recognizers(face_recognizers, face_imgs, 
                                            face_labels, userID_models)
        
        
        
    elif var1 == '-track':
        # IMPORT PRE-TRAINED RECOGNIZERS HERE
        log_msg = "Importing face recognizer"
        print(log_msg)
        logging.info(log_msg)

        LBPHF_face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        eigen_face_recognizer = cv2.face.EigenFaceRecognizer_create()
        fisher_face_recognizer = cv2.face.FisherFaceRecognizer_create()

        face_recognizer_models = [LBPHF_face_recognizer, eigen_face_recognizer, 
                            fisher_face_recognizer]
        face_recognizer_keys = ['LBPHF', 'Eigen', 'Fisher']
        face_recognizers = dict(zip(face_recognizer_keys,face_recognizer_models)) 


        face_recognizers = import_face_recognizer(userID_models, face_recognizers)
        
        
        track_faces(cap, face_detector, face_recognizers, userID_dict, monitorID, 
                                        url, log_file, size_threshold, classification_confidence)
    
    
    
    else:
        print("""user_identification must be called with either -capture -train \n \ 
                or -track to indicate what function you'd like to perform""")
                 
    
    print("*****************************************")
    print("***************** fin *******************")
    print("*****************************************")
    

if __name__ == "__main__":
    main()
    
