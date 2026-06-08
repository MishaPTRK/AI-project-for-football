import cv2
import time
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from TeamAssigner import TeamAssigner

model = YOLO("yolo11s.pt")

cap = cv2.VideoCapture("test.mp4")
count_frame = 0
FPS = 0

time_start = time.time()
prev_time = time.time()

pause = False
grayscale = False
Canny = False
detection = False
tracking  = False
countinueValue = True

if cap.isOpened() == False:
    print("Error opening video stream or file")

ret, frame = cap.read()
Resolution = np.shape(frame)


team_assigner = TeamAssigner()
team_assigner_ready = False

TEAM_COLORS = {
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 255),
}

def build_player_detections(boxes):
    player_detections = {}
    for box in boxes:
        if box.id is None:
            continue
        cls = int(box.cls[0].item())
        if cls != 0:
            continue
        s = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(s[0]), int(s[1]), int(s[2]), int(s[3])
        ID = int(box.id[0].item())
        player_detections[ID] = {"bbox": [x1, y1, x2, y2]}
    return player_detections




def StartTracking(frame):
    global team_assigner, team_assigner_ready

    results = model.track(frame, persist=True, tracker="bytetrack.yaml",classes=[0],conf=0.2,verbose=False)
    boxes = results[0].boxes

    if not team_assigner_ready:
        player_detections = build_player_detections(boxes)
        if len(player_detections) >= 2:
            team_assigner.assign_team_color(frame, player_detections)
            team_assigner_ready = True

    for box in boxes:
        if box.id is None:
            continue

        s = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(s[0]), int(s[1]), int(s[2]), int(s[3])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        ID = int(box.id[0].item())


        if cls == 0:
            color = (0, 255, 255)  # default before calibration
            if team_assigner_ready:
                try:
                    team_id = team_assigner.GetPlayerTeam(frame, [x1, y1, x2, y2], ID)
                    color = TEAM_COLORS.get(team_id, (0, 255, 255))
                except Exception as e:
                    print(f"Team assign error: {e}")

            player_crop = frame[y1:y2, x1:x2]
            cv2.putText(frame, f"ID: {ID} {conf:.2f}", (x1 - 25, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), thickness=2)

            foot_x = (x1 + x2) // 2
            foot_y = y2
            scale = x2-x1

            cv2.ellipse(
                frame,
                (foot_x, foot_y),
                (scale, 12),
                0,
                320,
                590,
                color,
                3
            )

    return frame



def draw_overlay(frame):
    cv2.putText(frame, f"Frame : {count_frame}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
    cv2.putText(frame, f"FPS : {FPS:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
    cv2.putText(frame, f"Res : {Resolution[0]}x{Resolution[1]}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                thickness=2)
    cv2.putText(frame, f"Time : {(current_time - time_start):.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                thickness=2)
    cv2.circle(frame, (Resolution[1] // 2, Resolution[0] // 2), 3, (255, 0, 0), thickness=3)
    return frame

def KeyFunctions(pause, Canny, grayscale,tracking,countinueValue):
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        pause = not pause


    elif key == ord('e'):
        Canny = not Canny

    elif key == ord('t'):
        tracking = not tracking

    elif key == ord('g'):
        grayscale = not grayscale

    elif key == ord('s'):
        cv2.imwrite(f"screenshots/frame{count_frame}.jpg", frame)
        print("saved")

    elif key == ord('q'):
        countinueValue = False
    return pause, Canny, grayscale,tracking, countinueValue

while countinueValue:
    if not pause:
        ret, frame = cap.read()
        if not ret:
            break


        if tracking:
            frame = StartTracking(frame)

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if Canny:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Canny(frame, 25,200)

        count_frame += 1
        current_time = time.time()

        if count_frame % 50 == 0:
            FPS = 1 / float(current_time - prev_time)

        prev_time = current_time
        frame = draw_overlay(frame)
    frame = cv2.resize(frame, (980, 540))
    cv2.imshow('frame',frame)

    pause, Canny, grayscale,tracking, countinueValue = KeyFunctions(pause, Canny, grayscale,tracking,countinueValue)

cap.release()
cv2.destroyAllWindows()