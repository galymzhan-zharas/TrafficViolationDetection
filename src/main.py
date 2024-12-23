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
    dx, dy = line_end - line_start

    crossedCars_annotator = sv.BoxAnnotator(color=sv.Color(255, 0, 0))
    notCrossedCars_annotator = sv.BoxAnnotator(color=sv.Color(0, 255, 0))
    plate_annotator = sv.BoxAnnotator(color=sv.Color(255, 0, 0))



    is_red = False
    start = time.time()
    text = ''
    frame_no = 0
    last = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # end = time.time()
        # if (is_red) and (end - start >= 18.0) or (not is_red) and (end - start >= 2): 
        #     is_red = not is_red
        #     start = time.time()

        # if frame_no - last <= 10: 
        #     is_red = False

        # elif frame_no - last > 10: 
        #     if (frame_no-last)-1 == 10: 
        #         last = frame_no-1
        #     is_red = True

        # elif frame_no - last == 100: 
        #     last = frame_no

        if frame_no < 200: 
            is_red = False
        else: 
            is_red = True
        

        annotated_image = frame.copy()
        if is_red: 
            car_results = yolov10_cars(frame, conf=0.5, classes=[2, 3, 5,7 ])[0]
            detections = sv.Detections.from_ultralytics(car_results)
            annotated_image = frame.copy()

            xywh = car_results.boxes.xywh.cpu().numpy()
            half_height = np.zeros((xywh.shape[0], 2))
            half_height[:, 1] = xywh[:, -1] / 2
            half_width = np.zeros((xywh.shape[0],2))
            half_width[:,1]=xywh[:,-2]/2

            bottom_centers = xywh[:, :2] + half_height

            num = dy * bottom_centers[:, 0] - dx * bottom_centers[:, 1] + line_end[0] * line_start[1] - line_end[1] * line_start[0]
            denom = np.sqrt(dy**2 + dx**2)
            dist = num/denom

            crossed = detections[dist<0]
            not_crossed = detections[dist>=0]

            if len(crossed) > 0:
                annotated_image = crossedCars_annotator.annotate(scene=annotated_image, detections=crossed)
            if len(not_crossed) > 0:
                annotated_image = notCrossedCars_annotator.annotate(scene=annotated_image, detections=not_crossed)

            roi = [frame[int(obj[1]):int(obj[3]), int(obj[0]):int(obj[2])] for obj in crossed.xyxy]

            for i, roi_img in enumerate(roi):
                plate_results = yolov10_plate(roi_img)[0]

                if plate_results:
                    x1 = int(plate_results.boxes.xyxy[0][0])
                    y1 = int(plate_results.boxes.xyxy[0][1])
                    x2 = int(plate_results.boxes.xyxy[0][2])
                    y2 = int(plate_results.boxes.xyxy[0][3])
                    print(x1,y1,x2,y2)
                    #detections = plate_results.xyxy[0]  
                        #x1, y1, x2, y2, conf, cls = plate_det[:6].int().tolist()
                    plate_img = roi_img[y1:y2, x1:x2]
                    plate_detections = sv.Detections.from_ultralytics(plate_results)
                    annotated_roi = plate_annotator.annotate(scene=roi_img, detections=plate_detections)

                    roi_x1, roi_y1, roi_x2, roi_y2 = crossed.xyxy[i].astype(int)
                    annotated_image[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi

                    ocr_result = ocr.ocr(plate_img)
                    
                    try: 
                        if ocr_result and ocr_result[0] and ocr_result[0][0] and ocr_result[0][0][-1]:
                            text = ocr_result[0][0][-1][0].upper()
                        else:
                            text = 'No text'
                    except Exception: 
                        print('Error with indexing')

                    plate_img_resized = cv2.resize(plate_img, (200, 100))  
                    annotated_image[0:100, 0:200] = plate_img_resized

            
                    # ocr_result = ocr.ocr(plate_img)
                    # if not ocr_result or not ocr_result[0]: 
                    #     text = 'No text detected'
                    # else:
                    #     ocr_result[0][0][-1][0].upper()

                    text_position = (210, 50)  
                    text_background_position = (200, 0, 400, 100) 

                    cv2.rectangle(annotated_image, (text_background_position[0], text_background_position[1]),
                                (text_background_position[2], text_background_position[3]), (255, 255, 255), -1)

                    font_scale = 0.9 * (text_background_position[2] - text_background_position[0]) / cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
                    text_thickness = 4

                    textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
                    textX = text_background_position[0] + (text_background_position[2] - text_background_position[0] - textSize[0]) / 2
                    textY = text_background_position[1] + (text_background_position[3] - text_background_position[1] + textSize[1]) / 2

                    cv2.putText(annotated_image, text, (int(textX), int(textY)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness)
                    #cv2.putText(annotated_image, text, (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7)
            line_color = (0,0,255)
            text = 'traffic light: red'
        else: 
            line_color = (255, 0, 0)
            text = 'traffic light: green'
        cv2.line(annotated_image, tuple(line_start), tuple(line_end), line_color, 2)
        cv2.putText(annotated_image, text, (frame.shape[1]-500, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, color=line_color, thickness=5)
        cv2.imshow('Cars and buses', annotated_image)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_no+=1
    print(i)
    end = time.time()
    print('time: ',end - start)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
