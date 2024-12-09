import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from paddleocr import PaddleOCR
import time


def add_text_with_background(image, text, position, font, font_scale, font_color, thickness, background_color):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    box_coords = ((position[0], position[1] + 10), (position[0] + text_width, position[1] - text_height - 10))
    cv2.rectangle(image, box_coords[0], box_coords[1], background_color, cv2.FILLED)
    cv2.putText(image, text, position, font, font_scale, font_color, thickness)



def main():
    
    cv2.namedWindow('Cars and buses', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cars and buses', 500, 800)

    yolov10_cars = YOLO('yolov10n.pt').to("cuda")
    yolov10_plate = YOLO('yolob8m.pt').to("cuda")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', res_algorithm="CRNN", show_log=False, det_algorithm="DB")

    video_path = 'vehicles_pass2.MOV'
    cap = cv2.VideoCapture(video_path)

    points = [[8, 1234], [717,1190]]
    line_color = (255,0,0)
    line_start = np.array(points[0])
    line_end = np.array(points[1])
    dx, dy = line_end-line_start
    # cv2.namedWindow('cars and buses')
    # cv2.setMouseCallback('cars and buses', select_points)
    # bounding_box_annotator = sv.BoundingBoxAnnotator()
    # label_annotator = sv.LabelAnnotator()
    crossedCars_annotator = sv.BoundingBoxAnnotator(color=sv.Color(255,0,0))
    notCrossedCars_annotator = sv.BoundingBoxAnnotator(color=sv.Color(0,255,0))
    plate_annotator = sv.BoundingBoxAnnotator(color=sv.Color(255,0,0))
    is_red = True
    while True: 
        ret, frame = cap.read()
        # pass the frame to the YOLO model 
        if not ret: 
            break
        
        results = yolov10_cars(frame, conf=0.5, classes=[2,5])[0] # {'car': 2, 'bus':5}
        detections = sv.Detections.from_ultralytics(results)
        annotated_image = frame.copy()

        if is_red:
            xywh = results.boxes.xywh.numpy()

            half_height = np.zeros((xywh.shape[0],2))
            half_height[:,1]=xywh[:,-1]/2
            half_width = np.zeros((xywh.shape[0],2))
            half_width[:,1]=xywh[:,-2]/2

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
                                scene=annotated_image, detections=crossed)
            if len(not_crossed) > 0: 
                annotated_image = notCrossedCars_annotator.annotate(
                                scene=annotated_image, detections=not_crossed)    

    

            roi = [frame[int(obj[1]):int(obj[3]), int(obj[0]):int(obj[2])] for obj in crossed.xyxy]

            for i, roi_img in enumerate(roi):
                plate_result = yolov10_plate(roi_img)

                if plate_result:
                    plate_detections = sv.Detections.from_ultralytics(plate_result[0]) 
                    annotated_roi = plate_annotator.annotate(scene=roi_img, detections=plate_detections)

                    roi_x1, roi_y1, roi_x2, roi_y2 = crossed.xyxy[i].astype(int)
                    annotated_image[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi


        else: 
            annotated_image = notCrossedCars_annotator.annotate(
                                scene=annotated_image, detections=detections)
        
        cv2.line(annotated_image, line_start, line_end, (255,0,0), 2)
        
        cv2.imshow('Cars and buses', annotated_image)
        # cv2.imshow('cars and buses', frame)
        if cv2.waitKey(1)==ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # detect()
    main()
    # pass