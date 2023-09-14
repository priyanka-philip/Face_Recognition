import cv2
import os
import numpy as np
from keras.models import model_from_json
import pathlib
import pyttsx3  # Import the pyttsx3 library
# Step 1: Capture and Store Reference Images
def capture_reference_images(person_name, num_images=600):
    cap = cv2.VideoCapture(0)
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(str(cascade_path))
    #face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # Create a folder for the person's reference images
    if not os.path.exists(person_name):
        os.makedirs(person_name)

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=1)

        for (x, y, w, h) in num_faces:
            face = gray_frame[y:y + h, x:x + w]
            # Save the captured face image
            img_path = os.path.join(person_name, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_path, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Capture Reference Images', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Step 2: Train the Face Recognition Model
def train_face_recognition():
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_images = []
    labels = []

    image_size = (100, 100)  # Set a consistent image size for training

    for label, person_name in enumerate(os.listdir('.')):
        if os.path.isdir(person_name):
            for img_file in os.listdir(person_name):
                img_path = os.path.join(person_name, img_file)
                face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if face is not None and face.size > 0:  # Check if the image is valid
                    face = cv2.resize(face, image_size)  # Resize the image
                    face_images.append(face)
                    labels.append(label)

    face_recognizer.train(face_images, np.array(labels))
    face_recognizer.save("model/eigenface_model.xml")

engine = pyttsx3.init()
# Step 3: Recognize and Display Name on Frame
def recognize_and_display():
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer = cv2.face_EigenFaceRecognizer.create()
    face_recognizer.read("model/eigenface_model.xml")
    # Load the emotion detection model
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    json_file = open('model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights("model/emotion_model.h5")
    face_images = []
    labels = []
    image_size = (100, 100)
    cap = cv2.VideoCapture(0)
    person_recognized = 0
    person_name=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
        face_detector = cv2.CascadeClassifier(str(cascade_path))
        #face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=1)
        #print(num_faces)

        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            resized_roi = cv2.resize(roi_gray_frame, image_size)
            # Perform face recognition
            label, confidence = face_recognizer.predict(resized_roi)
            print("Label:", label, "Confidence:", confidence)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if confidence < 1500:
                person_name = os.listdir('.')[label]
                cv2.putText(frame, person_name, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)

                # Resize the ROI for emotion detection
                resized_roi = cv2.resize(roi_gray_frame, (48, 48))
                cropped_img = np.expand_dims(np.expand_dims(resized_roi, -1), 0)
                # Perform emotion detection
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 5, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
                if person_recognized==0:
                    # Announce the name of the recognized person
                    engine.say(f"Hi Hello This one is {person_name} and emotion is {emotion_dict[maxindex]}")
                    engine.runAndWait()
                person_recognized = True  # Set the flag to True
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Uncomment and call the functions to perform the respective steps
#unknown_name = input("Enter the name for the unknown person: ")
#capture_reference_images(unknown_name, num_images=600)
#train_face_recognition()
recognize_and_display()
