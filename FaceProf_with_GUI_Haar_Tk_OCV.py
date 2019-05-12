"""

# -*- coding: utf-8 -*-
Created on          07 May 2019 at 6:53 PM
Author:             Arvind Sachidev Chilkoor  
Created using:      PyCharm
Name of Project:    Using Tkinter and OpenCV to detect the facial profile using Webcam

Description:  
This script is a demonstration of combining Tkinter and OpenCV, where the USER can view/detect individual
features of the face such as full face, eyes, mouth and nose.
The program uses Haar Cascades the machine learning classifiers in this case.
The program is based within a single CLASS, i.e. FaceProfileApp.
The program also records and saves the video as a .avi file for future viewing.
"""

import tkinter
import cv2



class FaceProfileApp:

    def __init__(self, window, title, videoSource=0):
        """
        __init__ method declaration for FaceProfileApp.
        :param window:
        :param title:
        :param videoSource:
        """
        # Constructor declaration.
        self.window = window
        self.window.title(title)
        self.videoSource = videoSource

        # To indicate the choices to the user
        self.label = tkinter.Label(window, text=" SELECT THE FACIAL FEATURE THAT YOU WANT THE WEBCAM TO DETECT FROM THE"
                                                " CHOICES BELOW ", fg='green', font=('Arial Bold', 12))
        self.label.pack(anchor=tkinter.N)

        # Tkinter declaration for the GUI Button - Face
        self.Face_button = tkinter.Button(window, text="For Face Only", width=50, command=self.command_face)
        self.Face_button.pack(anchor=tkinter.CENTER, expand=True)

        # Tkinter declaration for the GUI Button - Eyes
        self.Eyes_button = tkinter.Button(window, text="For Eyes Only", width=50, command=self.command_eyes)
        self.Eyes_button.pack(anchor=tkinter.CENTER, expand=True)

        # Tkinter declaration for the GUI Button - Nose
        self.Nose_button = tkinter.Button(window, text="For Nose Only", width=50, command=self.command_nose)
        self.Nose_button.pack(anchor=tkinter.CENTER, expand=True)

        # Tkinter declaration for the GUI Button - Mouth
        self.Mouth_button = tkinter.Button(window, text="For Mouth Only", width=50, command=self.command_mouth)
        self.Mouth_button.pack(anchor=tkinter.CENTER, expand=True)

        # To indicate that pressing the 'q' on the keyboard will quit/exit the webcam feed.
        self.label = tkinter.Label(window, text = " PRESS Q TO QUIT WEBCAM FEED AND CLOSE THE WINDOW ", fg = 'red',
                                   font = ('Arial Bold', 12))
        self.label.pack(anchor = tkinter.N)

        # To keep window in loop until the USER closes/exits
        self.window.mainloop()

    def command_face(self):
        """
        Method for Face Detection
        :return:
        """

        # Haar Cascade Classifier for face
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # OpenCv declaration for default camera = 0 i.e. Webcam
        cap = cv2.VideoCapture(0)

        # Defines the type of the CODEC
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Declaration for saving the video, at 20 FPS and 640,480 is the frame size
        out = cv2.VideoWriter('Detect_Face.avi', fourcc, 20.0, (640, 480))

        while True:
            # While Loop condition for continuous detection

            # ret is 'return' which is boolean value if the video frame is read correctly.
            # img is the video/camera image or file...
            ret, img = cap.read()

            # converts the color image to grayscale.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detects objects of different sizes in the input image. The detected objects are returned as a
            # list of rectangles.
            # 1.3 is the scale factor, 5 is the no of neighbouring pixels to scan
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # For Loop to continuously detect the face profile
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Saves the video
            out.write(img)

            # Display the Webcam feed with profile detection
            cv2.imshow('img', img)

            # Introducing a delay to make sure that Webcam feed is not cut abruptly, or too quickly
            key = cv2.waitKey(1)

            # Condition for quitting the Webcam feed which is 'q' from keyboard
            if key & 0xFF == ord('q'):
                break

        # Exits from recording mode
        out.release()

        # Closes Webcam and releases it for subsequent recordings
        cap.release()

        # Destroys all windows, practically to close the application.
        cv2.destroyAllWindows()



    def command_eyes(self):
        """
        Method for eyes detection
        :return:
        """
        # Haar Cascade Classifier for eye
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        # OpenCv declaration for default camera = 0 i.e. Webcam
        cap = cv2.VideoCapture(0)

        # Defines the type of the CODEC
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Defines the type of CODEC

        # Declaration for saving the video, at 20 FPS and 640,480 is the frame size
        out = cv2.VideoWriter('Detect_eyes.avi', fourcc, 20.0, (640, 480))

        while True:
            # While Loop condition for continuous detection

            # ret is 'return' which is boolean value if the video frame is read correctly.
            # img is the video/camera image or file...
            ret, img = cap.read()

            # Converts the color image to grayscale.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detects objects of different sizes in the input image. The detected objects are returned as a
            # list of rectangles.
            # 1.3 is the scale factor, 5 is the no of neighbouring pixels to scan
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

            # For Loop to continuously detect the eye profile
            for (x, y, w, h) in eyes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Saves the video
            out.write(img)

            # Display the Webcam feed with profile detection
            cv2.imshow('img', img)

            # Introducing a delay to make sure that Webcam feed is not cut abruptly, or too quickly
            key = cv2.waitKey(1)

            # Condition for quitting the Webcam feed which is 'q' from keyboard
            if key & 0xFF == ord('q'):
                break

        # Exits from recording mode
        out.release()

        # Display the Webcam feed with profile detection
        cap.release()

        # Destroys all windows, practically to close the application.
        cv2.destroyAllWindows()


    def command_nose(self):
        """
        Method for Nose detection
        :return:
        """

        # Haar Cascade Classifier for nose
        nose_cascade = cv2.CascadeClassifier('Nariz.xml')

        # OpenCv declaration for default camera = 0 i.e. Webcam
        cap = cv2.VideoCapture(0)

        # Defines the type of the CODEC
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Defines the type of CODEC

        # Declaration for saving the video, at 20 FPS and 640,480 is the frame size
        out = cv2.VideoWriter('Detect_nose.avi', fourcc, 20.0, (640, 480))

        while True:
            # While Loop condition for continuous detection

            # ret is 'return' which is boolean value if the video frame is read correctly.
            # img is the video/camera image or file...
            ret, img = cap.read()

            # converts the color image to grayscale.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detects objects of different sizes in the input image. The detected objects are returned as a
            # list of rectangles.
            # 1.3 is the scale factor, 5 is the no of neighbouring pixels to scan
            nose = nose_cascade.detectMultiScale(gray, 1.3, 5)

            # For Loop to continuously detect the nose profile
            for (x, y, w, h) in nose:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Saves the video
            out.write(img)

            # Display the Webcam feed with profile detection
            cv2.imshow('img', img)

            # Introducing a delay to make sure that Webcam feed is not cut abruptly, or too quickly
            key = cv2.waitKey(1)

            # Condition for quitting the Webcam feed which is 'q' from keyboard
            if key & 0xFF == ord('q'):
                break

        # Exits from recording mode
        out.release()

        # Display the Webcam feed with profile detection
        cap.release()

        # Destroys all windows, practically to close the application.
        cv2.destroyAllWindows()


    def command_mouth(self):
        """
        Method for Mouth detection
        :return:
        """

        # Haar Cascade Classifier for mouth
        mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

        # OpenCv declaration for default camera = 0 i.e. Webcam
        cap = cv2.VideoCapture(0)

        # Defines the type of the CODEC
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Defines the type of CODEC

        # Declaration for saving the video, at 20 FPS and 640,480 is the frame size
        out = cv2.VideoWriter('Detect_mouth.avi', fourcc, 20.0, (640, 480))

        while True:
            # While Loop condition for continuous detection

            # ret is 'return' which is boolean value if the video frame is read correctly.
            # img is the video/camera image or file...
            ret, img = cap.read()

            # converts the color image to grayscale.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detects objects of different sizes in the input image. The detected objects are returned as a
            # list of rectangles.
            # 1.3 is the scale factor, 5 is the no of neighbouring pixels to scan
            mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)

            # For Loop to continuously detect the mouth profile
            for (x, y, w, h) in mouth:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Saves the video
            out.write(img)

            # Display the Webcam feed with profile detection
            cv2.imshow('img', img)

            # Introducing a delay to make sure that Webcam feed is not cut abruptly, or too quickly
            key = cv2.waitKey(1)

            # Condition for quitting the Webcam feed which is 'q' from keyboard
            if key & 0xFF == ord('q'):
                break

        # Exits from recording mode
        out.release()

        # Display the Webcam feed with profile detection
        cap.release()

        # Destroys all windows, practically to close the application.
        cv2.destroyAllWindows()


# Calling the FaceProfileApp Method
FaceProfileApp(tkinter.Tk(),"FACE FEATURE DETECTION USING HAAR CASCADES, TKINTER AND OPENCV")
