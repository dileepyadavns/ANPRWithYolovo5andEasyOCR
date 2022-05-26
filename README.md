#              Automatic Number Plate Recognition with YOLOV5 and EasyOCR

 A full-fledged system for authentication of the cars using their number plate which can have many real-world use cases for example automated garage opening.
# Objective:
To ensure that the access must be given to the users who have registered their car's number plate in the system.

# 1. DATA COLLECTION:
Downloaded high quality car images with clear and visible number plate
And cleaned if there is any image which is not clear and splitted the images to Train and 
Validation 

# 2. Labeling Images:
As i am using YOLOV5 algorithm i used labelimg tool for labeling images as yolo 
requires specific format to train images.
Bounding boxes: Bounding boxes are the most commonly used type of 
annotation in computer vision. Bounding boxes are rectangular boxes used to 
define the location of the target object. They can be determined by the 𝑥 and 𝑦 
axis coordinates in the upper-left corner and the 𝑥 and 𝑦 axis coordinates in the 
lower-right corner of the rectangle. Bounding boxes are generally used in object 
detection and localisation tasks.
# 3.Training the model:
Trained the dataset on YOLOv5 algorithm tried different versions like small medium and
Large versions.  After training the model downloaded the weights file i got, which can be 
used for future interface 
# 4.Loading the model:
Loaded the model using torch  and given the weights file which I have after training the  model on the dataset. 
# 5.Taking Frames from video:
Frame:
A frame is an image that forms a single instance of a video. A video consists of a lot of frames running per second (also known as frames per second). Using OpenCv2 taken frames from the video 

# 6.Passing each frame for number plate detection:
Each frame of the video is passed to yolov5 loaded model and number plate is detected for each frame
# 7.Passing Each frame with Bounding boxes to EasyOCR:
After passing the frame to YOLOV5 we get bounding boxes. We take the bounding boxes from  frame and we unpack the bounding boxes and will crop the image on bounding boxes and passing it to EasyOcr for getting the text out lof it
sample video: https://drive.google.com/file/d/1F7gMJXLnHzYAZa7SYfWOR-9cWWSg1afr/view?usp=sharing
# 8.Making list of all the extracted number plates:
Added all number plate details extracted from each frame to a list and took the maximum repeating number plate as the final predicted number plate.
# 9.Connecting to PostgreSQL:
Created a table on postgreSQL with a column containing number plate details . later connected the PostgreSQL to flask using Psycopg2 

# 10. Checking with database:
The extracted text then checked in database and printed the output based on the existence of number plate on database
