import os               # This library is used to get access to your Operating System
import numpy as np      # This library is used to manipulate & claculate with Vectors & Arrays
import cv2              # This library is used in this program to Recognize faces & get access to your Webcam.
import face_recognition # This library is used to Face Image Reading & all the manipulation required to recognaize faces.
import pandas as pd     # This library is used to Read, Save & Manipulate Dataset.
from os import walk     # This library returns a generator, that creates a tuple of values

mypath = './known_faces/'
known_faces = []
for (dirpath, dirnames, filenames) in walk(mypath):
    known_faces.extend(filenames)
    break

image_list = []    # Make an empty List

for i in range(len(known_faces)):
    im = face_recognition.load_image_file('./known_faces/' + known_faces[i])
    image_list.append(im)

for i in range(len(known_faces)):
    known_faces[i] = known_faces[i].replace('.jpg','')
    known_faces[i] = known_faces[i].replace('.JPG','')

known_face_encoding_dictionary = []    # Make an empty List

print("\nPlease wait..." + "\n\tTraining Dataset is being loaded...\n")
print("\nSL no. \t\t Image_name \t Image_pixel_shape\n")

for i in range(len(known_faces)):
    image_name, image = known_faces[i], image_list[i]  # Converting List string into variable
    print(i+1, 'of', len(known_faces), '\t ', image_name, '\t ', image.shape)
    face_encoded = face_recognition.face_encodings(image)[0]
    known_face_encoding_dictionary.append(face_encoded)

Saved_Dataset = {
                    'Known_Faces_Name': known_faces,
                    'Known_Faces_Encoding_Dictionary': known_face_encoding_dictionary
                }

data_frame = pd.DataFrame(Saved_Dataset)
data_frame.to_csv ('Trained_Dataset.csv', index = False, header = True)


# print (data_frame)
print(Saved_Dataset['Known_Faces_Encoding_Dictionary'])

Trained_data = pd.read_csv('Trained_Dataset.csv')

known_faces_read = Trained_data['Known_Faces_Name']
known_face_encoding_dictionary_read = Trained_data['Known_Faces_Encoding_Dictionary']
# If you don't 'Squeeze' them, the variables will also read Datatypes & more info.
# But we need v

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read();  #capture frame by frame of video
    rgb_frame = frame[:, :, ::-1];  #converting the frame from OpenCV's BGR format to the RGB format

    #finds the face locations and encodings in each frame
    face_locations = face_recognition.face_locations(rgb_frame);
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations);

    #loops through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        #checks if the face is a match for known faces
        matches = face_recognition.compare_faces(known_face_encoding_dictionary, face_encoding);
        #if not, labelled as Unknown
        name = 'Unknown';

        # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. The distance tells you how similar the faces are
        face_distances = face_recognition.face_distance(known_face_encoding_dictionary, face_encoding);
        best_match = np.argmin(face_distances);

        if matches[best_match]:
            name = known_faces[best_match];

        #draws a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2);

        #draws a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED);
        font = cv2.FONT_HERSHEY_DUPLEX;
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1);

    #displays the webcam video on screen
    cv2.imshow('Video', frame);

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

video_capture.release();
cv2.destroyAllWindows();
