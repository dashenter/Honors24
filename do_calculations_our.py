# This code serves as a tool to analyze US videos. Make sure to run this first before starting the dashboard (running app3.py file)
# It creates the output video with the metrics: pennation angle, muscle thickness, and fascicle length as well as the corresponding graphs.
# It saves the output video so that it is connected with the dashboard.
# Start with analyzing the video from DL_Track_US_Example (calf_raise).
# Please make sure to read all the comments as they will help you with the correct setup of this code.

#Imports
import cv2
import numpy as np
from tensorflow.keras.models import load_model, model_from_json, Model
from tensorflow.keras.metrics import Metric
import random
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
import tensorflow as tf
import h5py
import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class IoU(Metric):
    def __init__(self, num_classes, target_class_ids, name='iou', **kwargs):
        super(IoU, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.target_class_ids = target_class_ids
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        iou = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        self.total_iou.assign_add(iou)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_iou / self.count

    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super(IoU, self).get_config()
        config.update({
            "num_classes": self.num_classes,
            "target_class_ids": self.target_class_ids
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# This part was added in the summer 2024 as the apo and fasc models were saved with a groups parameter, which is not supported in the current version of Keras for Conv2DTranspose.
# The code below (until def create_bar_plot, serves to extract the models without the specified parameter).
class CustomConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super(CustomConv2D, self).__init__(*args, **kwargs)


class CustomConv2DTranspose(Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        if 'groups' in config:
            del config['groups']
        return super().from_config(config)


# Define a dictionary to register custom objects
custom_objects = {
    'IoU': IoU,
    'CustomConv2D': CustomConv2D,
    'CustomConv2DTranspose': CustomConv2DTranspose,
    'Functional': Model
}


def load_model_without_groups(model_path):
    with h5py.File(model_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model configuration found in the file.')
        model_config = json.loads(model_config)
        for layer in model_config['config']['layers']:
            if 'config' in layer:
                layer['config'].pop('groups', None)

    model_json = json.dumps(model_config)
    model = model_from_json(model_json, custom_objects=custom_objects)
    model.load_weights(model_path)
    return model


# Function to create the bar graph of changing metrics
# def create_bar_plot(data, labels, title, max_y):
#     fig, ax = plt.subplots()
#     ax.bar(labels, data, color='#1e90ff')
#     ax.set_title(title)
#     ax.set_ylim(0, max_y)
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)  # Change this line
#     width, height = canvas.get_width_height()
#     buf = buf.reshape(height, width, 4)  # Update shape for ARGB
#     plt.close(fig)
#     return buf


def create_bar_plot(data, labels, title, max_y):
    fig, ax = plt.subplots()
    ax.bar(labels, data, color='#1e90ff')
    ax.set_title(title)
    ax.set_ylim(0, max_y)

    # Save the figure as an image instead of converting it to an array
    temp_filename = "bar_plot_temp.png"
    fig.savefig(temp_filename)
    plt.close(fig)

    # Load the image as a frame
    frame = cv2.imread(temp_filename)
    return frame

# Function to create the line graph of changing metrics
# def create_line_plot(data, labels, title, max_y):
#     fig, ax = plt.subplots()
#     ax.plot(labels, data, color='#1068b2')
#     ax.set_title(title)
#     ax.set_ylim(0, max_y)
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)  # Change this line
#     width, height = canvas.get_width_height()
#     buf = buf.reshape(height, width, 4)  # Update shape for ARGB
#     plt.close(fig)
#     return buf
def create_line_plot(data, labels, title, max_y):
    fig, ax = plt.subplots()
    ax.plot(labels, data, color='#1068b2')
    ax.set_title(title)
    ax.set_ylim(0, max_y)

    # Save the figure as an image instead of converting it to an array
    temp_filename = "line_plot_temp.png"
    fig.savefig(temp_filename)
    plt.close(fig)

    # Load the image as a frame
    frame = cv2.imread(temp_filename)
    return frame

def annotate_frame(frame, annotations, position=(50, 50), font_scale=1, color=(255, 255, 255)):
    for i, text in enumerate(annotations):
        cv2.putText(frame, text, (position[0], position[1] + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    return frame


# Since the apo and fasc models create a video in the AVI format, a conversion to MP4 format needs to be done.
# Make sure you have https://www.gyan.dev/ffmpeg/builds/ downloaded.
# Insert the path to the ffmpeg.exe file that should be located in the 'bin' directory of the uppermentioned file.
# At first, I tried dowloading https://ffmpeg.org/download.html, but there was no 'bin' directory, so I had to download the file mentioned above.
# Try downloading either one of those and locate the 'bin' directory.
def convert_avi_to_mp4(avi_path, mp4_path):
    ffmpeg_path = r"/Users/liischmidt/Documents/GitHub/Honors24/ffmpeg.exe"  # Update this path to the actual path of ffmpeg.exe
    command = [ffmpeg_path, '-i', avi_path, '-vcodec', 'libx264', '-acodec', 'aac', mp4_path]
    subprocess.run(command, check=True)

# Even after converting the videos into the MP4 format, they use the MP4 codec. It is necessary to have the AVC1 codec so the conversion is done in the snippet below.
# Insert the same path to the ffmpeg.exe file as the one above
def reencode_to_avc1(mp4_path, reencoded_mp4_path):
    ffmpeg_path = r"/Users/liischmidt/Documents/GitHub/Honors24/ffmpeg.exe"  # Update this path to the actual path of ffmpeg.exe
    command = [ffmpeg_path, '-i', mp4_path, '-vcodec', 'libx264', '-acodec', 'aac', reencoded_mp4_path]
    subprocess.run(command, check=True)


def doCalculationsVideo(input_video_path, output_video_path, bar_graph_video_path, line_graph_video_path, apo_model_path, fasc_model_path,
                        parameters, calib_dist=None, step=1, flip='no_flip', filter_fasc=False, update_interval=30):

    def compute_shortening_velocity(fascicle_lengths, fps):
        velocities = []
        contraction_active = False
        start_time, start_length = None, None

        for i in range(1, len(fascicle_lengths)):
            prev_length = fascicle_lengths[i - 1]
            curr_length = fascicle_lengths[i]
            time_elapsed = 1 / fps  # Time difference per frame

            if curr_length < prev_length:  # Fascicle shortening detected
                if not contraction_active:  # Start of contraction
                    start_time = i * time_elapsed
                    start_length = prev_length
                    contraction_active = True
            else:
                if contraction_active:  # End of contraction
                    end_time = (i - 1) * time_elapsed
                    end_length = prev_length
                    velocity = abs((end_length - start_length) / (end_time - start_time))  # Vf = ΔL / Δt
                    velocities.append(velocity)
                    contraction_active = False

        return round(sum(velocities) / len(velocities), 2) if velocities else 0.0
    

    # Loading the models without the specified parameter groups
    apo_model = load_model_without_groups(apo_model_path)
    fasc_model = load_model_without_groups(fasc_model_path)

    # Opening the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    if not os.path.exists(bar_graph_video_path) or os.path.getsize(bar_graph_video_path) == 0:
        print(f"Error: {bar_graph_video_path} is empty or does not exist!")
        return

    if not os.path.exists(line_graph_video_path) or os.path.getsize(line_graph_video_path) == 0:
        print(f"Error: {line_graph_video_path} is empty or does not exist!")
        return

    # Extracting Video Properties
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) #frames per second

    # Setting Up Video Writers (saving the processed frames into new video files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    bar_vid_out = cv2.VideoWriter(bar_graph_video_path, fourcc, fps, (640, 480))
    line_vid_out = cv2.VideoWriter(line_graph_video_path, fourcc, fps, (640, 480))

    # Initializing Variables
    frame_count = 0
    current_annotations = []
    stats = {
        'fascicle_length': [],
        'pennation_angle': [],
        'muscle_thickness': []
    }

    last_bar_plot_frame = None
    last_line_plot_frame = None
    max_values = {
        'fascicle_length': 15,
        'pennation_angle': 25,
        'muscle_thickness': 2
    }

    # Processing
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % update_interval == 0:
                # Preprocess frame for model input
                input_frame = cv2.resize(frame, (512, 512))
                input_frame = input_frame / 255.0
                input_frame = np.expand_dims(input_frame, axis=0)

                # Make predictions
                fasc_prediction = fasc_model.predict(input_frame)[0]
                apo_prediction = apo_model.predict(input_frame)[0]

                # Calculate metrics based on predictions
                fascicle_length = (np.sum(fasc_prediction > parameters['fasc_threshold']) * (calib_dist if calib_dist else 1)) / 100
                pennation_angle = np.mean(apo_prediction > parameters['apo_threshold']) * (parameters['max_pennation'] - parameters['min_pennation']) + parameters['min_pennation']
                muscle_thickness = np.sum(fasc_prediction > parameters['fasc_threshold']) * (calib_dist if calib_dist else 1) / width

                stats['fascicle_length'].append(fascicle_length)
                stats['pennation_angle'].append(pennation_angle)
                stats['muscle_thickness'].append(muscle_thickness)

                current_annotations = [
                    f"Fascicle Length: {fascicle_length:.2f} cm",
                    f"Pennation Angle: {pennation_angle:.2f} degrees",
                    f"Muscle Thickness: {muscle_thickness:.2f} cm"
                ]

                data = [fascicle_length, pennation_angle, muscle_thickness]
                labels = ['Fascicle Length', 'Pennation Angle', 'Muscle Thickness']
                title = f"Frame {frame_count}"
                max_y = max(max_values.values()) + 5
                last_bar_plot_frame = create_bar_plot(data, labels, title, max_y)
                last_line_plot_frame = create_line_plot(data, labels, title, max_y)

            annotated_frame = annotate_frame(frame, current_annotations)
            vid_out.write(annotated_frame)

            if last_bar_plot_frame is not None:
                bar_vid_out.write(last_bar_plot_frame)

            if last_line_plot_frame is not None:
                line_vid_out.write(last_line_plot_frame)

            frame_count += 1

    finally:
        cap.release()
        vid_out.release()
        bar_vid_out.release()
        line_vid_out.release()
        cv2.destroyAllWindows()

                # After processing all frames, compute averages
        if stats['fascicle_length']:
            avg_fascicle_length = round(sum(stats['fascicle_length']) / len(stats['fascicle_length']), 2)
            avg_velocity = compute_shortening_velocity(stats['fascicle_length'], fps)
        else:
            avg_fascicle_length = 0
            avg_velocity = 0.0

        if stats['pennation_angle']:
            avg_pennation_angle = round(sum(stats['pennation_angle']) / len(stats['pennation_angle']), 2)
        else:
            avg_pennation_angle = 0

        if stats['muscle_thickness']:
            avg_muscle_thickness = round(sum(stats['muscle_thickness']) / len(stats['muscle_thickness']), 2)
        else:
            avg_muscle_thickness = 0


        # Save to a JSON file
        results2 = {
            "fascicle_length": avg_fascicle_length,
            "pennation_angle": avg_pennation_angle,
            "muscle_thickness": avg_muscle_thickness,
            "shortening_velocity": avg_velocity

        }

        with open("static/metrics.json", "w") as json_file:
            json.dump(results2, json_file)

        # Re-encode the MP4 files to use the 'avc1' codec
        reencode_to_avc1(output_video_path, output_video_path.replace('.mp4', '_reencoded.mp4'))
        reencode_to_avc1(bar_graph_video_path, bar_graph_video_path.replace('.mp4', '_reencoded.mp4'))
        reencode_to_avc1(line_graph_video_path, line_graph_video_path.replace('.mp4', '_reencoded.mp4'))


# Define file paths and parameters
# Feel free to change them according to your layout.
# If you are changing one of the output paths, make sure to have them located in the static folder!
input_video_path = r"/Users/liischmidt/Documents/GitHub/Honors24/static/calf_raise.mp4"  #Path to the US video you want to analyze
output_video_path = r"/Users/liischmidt/Documents/GitHub/Honors24/static/calf_raise.mp4" #Path to the analyzed output video.
bar_graph_video_path = r"static/UM01_calfraise_1_VSCAN.mp4" #Path to the created bar graph.
line_graph_video_path = r"static/UM01_calfraise_1_VSCAN.mp4" #Path to the created line graph.
apo_model_path = r"/Users/liischmidt/Downloads/model-apo-VGG16-BCE-512.h5" #Path to the model.
fasc_model_path = r"/Users/liischmidt/Downloads/model-fasc-VGG16-BCE-512.h5" #Path to the model.


# Based on different ultrasound videos, the parameters can be adjusted.
# For the DL_Track_US_Example keep the parameters as they are
parameters = {
    'apo_threshold': 0.2,
    'fasc_threshold': 0.05,
    'fasc_cont_thresh': 40,
    'min_width': 60,
    'min_pennation': 10,
    'max_pennation': 40
}

results = doCalculationsVideo(input_video_path, output_video_path, bar_graph_video_path, line_graph_video_path,
                              apo_model_path, fasc_model_path, parameters, update_interval=30)
