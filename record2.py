import cv2
import threading
import face_recognition
import numpy as np
from telegram.ext import Updater, CommandHandler
from telegram import Update
from telegram.ext import CallbackContext
from concurrent.futures import ThreadPoolExecutor



video_count = 0
images = []
motion = False
video_end = 0

###############################################

updater = Updater(token='1067685030:AAGuuS7kNXShEwQKaG57g6aLi82M6UluJMQ', use_context=True)

dispatcher = updater.dispatcher

def start(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

updater.start_polling()

###############################################

executor = ThreadPoolExecutor(10)

def motion_detector():
    global motion, video_end, images, video_count
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    cap = cv2.VideoCapture("rtsp://adminmarco:51xMMmEps3I8DdersdTjAMcyGc9Heo@marco23.dynv6.net:36478/stream2")
    high_quality = False

    cv2.startWindowThread()

    frame_count = 0
    previous_frame = None

    while True:
        # 1. Load image; convert to RGB
        ret, frame = cap.read()
        if not ret:
            continue

        print("video end = ", video_end)
        if video_end > 0:
            video_end -= 1
            images.append(frame)
            if video_end == 0:
                print("Video ended")
                motion = False

                height, width, layers = frame.shape

                video = cv2.VideoWriter("videos/video_" + str(video_count) + ".mp4", fourcc, 15, (width, height))
                video_count += 1
                for image in images:
                    video.write(image)

                cv2.destroyAllWindows()
                video.release()
                print("Video saved")

                th = threading.Thread(target=check_faces, args=images.copy())
                th.start()
                images = []
                print("Thread started")

                cap = cv2.VideoCapture(
                    "rtsp://adminmarco:51xMMmEps3I8DdersdTjAMcyGc9Heo@marco23.dynv6.net:36478/stream2")
                high_quality = False
                previous_frame = None

                ret, frame = cap.read()
                if not ret:
                    continue

        frame_count += 1

        if high_quality:
            crop_frame = frame[70:1080, 0:1920]
        else:
            crop_frame = frame[25:720, 0:1080]

        img_brg = crop_frame
        img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

        if (frame_count % 2) == 0:
            # 2. Prepare image; grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            # 3. Set previous frame and continue if there is None
            if previous_frame is None:
                # First frame; there is no previous one yet
                print("None, skip")
                previous_frame = prepared_frame
                continue

            # calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # 5. Only take different areas that are different enough (>20 / 255)
            thresh_frame = cv2.threshold(src=diff_frame, thresh=40, maxval=255, type=cv2.THRESH_BINARY)[1]

            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 60:
                    # too small: skip!
                    continue
                video_end = 30 * 5
                print("Motion")
                if not motion:
                    print("Record start 1")
                    motion = True
                    if not high_quality:
                        cap = cv2.VideoCapture(
                            "rtsp://adminmarco:51xMMmEps3I8DdersdTjAMcyGc9Heo@marco23.dynv6.net:36478/stream1")
                        high_quality = True
                        previous_frame = None

            cv2.imshow('VIDEO', thresh_frame)
            cv2.waitKey(1)

def check_faces(images):
    global video_count
    print("Thread started 2")
    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file("Marco.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding
    ]
    known_face_names = [
        "Marco",
    ]

    # Initialize some variables
    # face_locations = []
    # face_encodings = []
    # face_names = []
    process_this_frame = True
    face_names = set()
    for frame in images:
        print("Analyzing image")
        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

                print("Face = ", name)
                #process_this_frame = not process_this_frame

        updater.bot.send_video(229856560, video=open('videos/video_' + str(video_count) + '.mp4', 'rb'),
                               supports_streaming=True)
        message = "Faces: ", ''.join(face_names)
        updater.bot.send_message(229856560, text=message)



if __name__ == '__main__':
    motion_detector()
