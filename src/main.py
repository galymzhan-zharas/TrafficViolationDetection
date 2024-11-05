import cv2
from ultralytics import YOLO

def main():
    video_path = 'data/vehicles_stop.mp4'
    cap = cv2.VideoCapture(video_path)

    while True: 
        ret, frame = cap.read()

        if not ret: 
            break
        cv2.imshow('frame', frame)

        if cv2.waitKey(1)==ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # main()
    pass