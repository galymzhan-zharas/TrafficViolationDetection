import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
# points = []
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:  # Once two points are selected, disable further selection
            cv2.setMouseCallback('Frame', lambda *args : None)

def main():
    video_path = 'data/vehicles_stop.mp4'
    yolov10_cars = YOLO('yolov10n.pt') # initialize model
    cap = cv2.VideoCapture(video_path)
    points = [[6, 1263], [1072, 1176]]
    line_start = np.array(points[0])
    line_end = np.array(points[1])
    dx, dy = line_end-line_start
    # cv2.namedWindow('cars and buses')
    # cv2.setMouseCallback('cars and buses', select_points)
    # bounding_box_annotator = sv.BoundingBoxAnnotator()
    # label_annotator = sv.LabelAnnotator()
    crossedCars_annotator = sv.BoundingBoxAnnotator(color=sv.Color(255,0,0))
    notCrossedCars_annotator = sv.BoundingBoxAnnotator(color=sv.Color(0,255,0))

    while True: 
        ret, frame = cap.read()
        # pass the frame to the YOLO model 
        if not ret: 
            break
        
        results = yolov10_cars(frame, conf=0.5, classes=[2,5])[0] # {'car': 2, 'bus':5}
        detections = sv.Detections.from_ultralytics(results)

        xywh = results.boxes.xywh.numpy()

        half_height = np.zeros((xywh.shape[0],2))
        half_height[:,1]=xywh[:,-1]/2

        bottom_centers = xywh[:,:2] + half_height

  
        # annotated_image = label_annotator.annotate(
        #                     scene=annotated_image, detections=detections)
        # if len(points)==2:
        #     print(points[0], points[1])
        #     cv2.line(frame, points[0], points[1], (255,0,0), 2)
        num = dy*bottom_centers[:, 0] - dx*bottom_centers[:, 1] + line_end[0]*line_start[1] - line_end[1]*line_start[0]
        denom = np.sqrt(dy**2 + dx**2)
        dist = num/denom

        crossed = detections[dist<0]
        not_crossed = detections[dist>=0]

        if len(crossed) > 0: 
            annotated_image = crossedCars_annotator.annotate(
                            scene=frame, detections=crossed)
        if len(not_crossed) > 0: 
            annotated_image = notCrossedCars_annotator.annotate(
                            scene=frame, detections=not_crossed)
        
        
        cv2.line(annotated_image, line_start, line_end, (255,0,0), 2)
        
        cv2.imshow('Cars and buses', annotated_image)
        # cv2.imshow('cars and buses', frame)
        if cv2.waitKey(1)==ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # pass