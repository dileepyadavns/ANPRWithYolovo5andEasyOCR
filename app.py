
#importing required libraries
import torch
import cv2
import numpy as np
import easyocr
import psycopg2 #pip install psycopg2 
import psycopg2.extras

conn = psycopg2.connect( #psycopg2 database adaptor for implementing python
         host="localhost",
        database="students",
       user='postgres',
      password='p@ssw0rd')

# DEFINING GLOBAL VARIABLE
EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
OCR_TH = 0.2
plates=[]

#Number plate detection:
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    print(results)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

# to plot the BBox and results
def plot_boxes(results, frame,classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    #looping through the each frame of detection 

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
        
            coords = [x1,y1,x2,y2]

            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)
            plates.append(plate_num)
            print(plates)            

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)



    return frame



#function to recognize license plate  with EasyOCR
def recognize_plate_easyocr(img, coords,reader,region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image


    ocr_result = reader.readtext(nplate)



    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

    if len(text) ==1:
        text = text[0].upper()
    return text


### to filter out wrong detections 

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


#function for loading model
def main(vid_path=None,vid_out = None):
    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model =  torch.hub.load('/home/neosoft/Downloads/numberplaterecognition-main/yolov5', 'custom', source ='local', path='best2.pt') ### The repo is stored locally

    classes = model.names ### class names in string format
    
    # for detection on video 
    if vid_path !=None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret  and frame_no %1 == 0:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


                frame = plot_boxes(results, frame,classes = classes)
                
                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    
                    break
                frame_no += 1
            else:
                print(f"[INFO] Clening up. . . ")
                ### releaseing the writer
                cap.release()
                out.release()
                ## closing all windows
                cv2.destroyAllWindows()
                break

 #calling fuction
main(vid_path="skodacrop.mp4",vid_out="vid_1.mp4") ### for custom video
# main(vid_path=0,vid_out="webcam_facemask_result.mp4") #### for webcam
print("list of all number plates extracted from each frame"+ str(plates))

result = list(filter(lambda x: x==str(x),plates)) #filtered empty item from all extractions

most_common = max(result, key = result.count) #most extracted number plate
print("Number plate extracted of car from the video: "+ str(most_common))

cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
s = "SELECT * FROM cars where vehicleNo =%s"

cur.execute(s, (most_common,))
res = cur.fetchall()
if len(res)>=1:
    print("Output after checking in database: "+"Car {} has granted permission".format(most_common))
else: 
    print("Output after checking in database:  "+"Car {} not granted permission".format(most_common) )    
