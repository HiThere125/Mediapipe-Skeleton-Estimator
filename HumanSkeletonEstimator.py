
#
#-----------------------------------------------------------DISCLAIMER------------------------------------------------------------------------
#
#   1:  This project uses OS library. Please read at the code before running it to AVOID running POTENTIALLY MALICIOUS code on your computer
#       -   Never trust an unknown programmer. ALWAYS look over their code before running it, even if a description/summary is provided
#       -   If you still cannot trust the code, make it yourself. There are planty of online guides to make something similar to this
#
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
#
#   The purpose of this program is to serve as an introduction to mediapipe as a machine learning library for human recognition in media
#
#   Libraries:
#   1:  Open-CV (aka cv2)
#   2:  OS
#   3:  Warnings
#   4:  Logging
#   5:  Mediapipe
#   6:  Threading
#   7:  Tkinter
#   8:  TkinterDND2
#   9:  Tensorflow
#
#   Notes:
#   1:  This program has only been tested with .mp4 files that has one human in frame
#       -   Using other file-types, videos with no human, or multiple humans in the frame have not been tested and accuracy has not been verified
#   2:  Threading is used to keep the tkinter window responsive while processing the input file
#       -   Despite this it is recommended to only process one file at a time to reduce CPU load
#   3:  OS, Warnings, Logging, and Tensorflow are imported only to reduce messages in the console when running the program. They are not necessary to run the program
#   4:  A Windows Environment Variable was created to satisfy a warning message
#
#   Future Plans:
#   1:  Introduce other forms of mediapipe processing to add more to the video
#       -   For example: Draw an outline around the person in the frame
#   2:  Reverse engineer this process to eventually create something similar
#   3:  Test how changing certain aspects of the video affect the accuracy of the program
#       -   For example: If how the accuracy is affected by changing the resolution of the video
#

import cv2, os, warnings, logging
import mediapipe as mp
from threading import *
from tkinter import *
from tkinterdnd2 import DND_FILES, TkinterDnD
import tensorflow as tf

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
# Suppress specific warnings from google.protobuf.symbol_database
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')
# Configure absl logging to suppress warnings and info messages
logging.getLogger('absl').setLevel('ERROR')


''' Returns various details about the video that is being upscaled: FPS, Total Frames, Duration, Resolution
    @Params:    video_path  |   String  |   File path of the video that is being upscaled
    @Returns:   fps         |   Integer |   Frames per second of the video file
                frame_count |   Integer |   Total frames in the video file
                duration    |   Integer |   Total length of the video file
                resolution  |   List    |   The size of the frames in the video file
                    [0]     |   Integer |   Width of the frame in pixels
                    [1]     |   Integer |   Height of the frame in pixels'''
def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(video.get(cv2.CAP_PROP_FPS)))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    duration = frame_count/fps
    resolution = [frame_width, frame_height]
    print(f"FPS:    {fps}\nFrame Count: {frame_count}\nDuration: {duration}\nResolution: {resolution}")
    return fps, frame_count, duration, resolution

''' Returns a video with a skeleton estimate drawn over each frame
    @Params:    input_path  |   String  |   Path to the Video file
                output_path |   String  |   Path to save the output video at
                fps         |   Int     |   Frames per second of the original video
                resolution  |   List    |   The size of the frames in the video file
                    [0]     |   Integer |   Width of the frame in pixels
                    [1]     |   Integer |   Height of the frame in pixels
    @Returns:   None'''
def pose_estimator(input_path, output_path, fps, frame_count, resolution):    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (resolution[0], resolution[1]))
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(RGB)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:  
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Processing done")

''' Returns the file path of the output video
    @Params:    input_path  |   String  |   Path to the input video
    @Returns:   output_path |   String  |   Path to save the new video at'''
def get_output_path(input_path):
    split_output_path = input_path.split("/")
    filename =  split_output_path[len(split_output_path)-1]
    new_filename = "Pose Estimated " + filename
    output_path = "\\".join(split_output_path[:len(split_output_path)-1]) + "\\" + new_filename
    return output_path

''' Processes the drag and drop input event, changing the label to reflect what was most recently dropped in the box
    @Params:    event       |           |   Drag and drop event
    @Returns:   None'''
def process_input(event):
    path = event.data.replace("{","").replace("}","")
    window.insert("end", path)
    l3.config(text = path)

''' Starts the estimation process by pulling the text from label l3 and using that as the video path
    @Params:    None
    @Returns:   None'''
def take_input():
    video = l3.cget("text")
    output = get_output_path(video)
    fps, frame_count, _, resolution = get_video_info(video)
    thread1 = Thread(target=pose_estimator, args=(video, output, fps, frame_count, resolution))
    thread1.start()

''' Handles Tkinter Drag-and-Drop window and serves as the GUI'''
if __name__ == "__main__":
    root = TkinterDnD.Tk()
    root.geometry("300x350")
 
    # Create labels
    l = Label(root, text = "Skeleton Estimator")
    l2 = Label(root, text = "Path to video:")
    l3 = Label(root, text = "")

    # Create buttons
    b1 = Button(root, text = "Estimate Pose", command = lambda:take_input())
    b2 = Button(root, text = "Exit",
                command = root.destroy)

    # Create Drag and Drop box
    window = Listbox(root)
    window.drop_target_register(DND_FILES)
    window.dnd_bind('<<Drop>>', process_input)
 
    # Pack all tkinter window components
    l.pack()   
    l2.pack()
    l3.pack()
    window.pack()
    b1.pack()
    b2.pack()
    
    # Run the tkinter window
    mainloop()