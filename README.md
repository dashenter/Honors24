# USONO-2023
This project, developed by Health Track Honors students in 2024, in collaboration with Usono, focuses on enhancing ultrasonic imaging through machine learning models. It aims to provide advanced tools and algorithms for improved analysis and interpretation of ultrasound images.

# Features
Integration of [deepMTJ][1] and [Paul Rietsche DL_Track_US][2] models for advanced muscle and tissue analysis. Flask-based web application for accessibility and ease of use. Adaptability for both mobile and web platforms.

[1]: https://github.com/luuleitner/deepMTJ
[2]: https://github.com/PaulRitsche/DL_Track_US

# Contribution
As a project under active development, contributions from the medical and tech communities are welcome.

# License
GNU General Public License

For more details on the project, contributions, and usage, please refer to the provided documentation.

# Prerequisites

Python 3.x

Ensure you have the following Python libraries installed: numpy, pandas, opencv-python, matplotlib, dash, dash-core-components, dash-html-components, scipy, skimage, tensorflow, h5py, subprocess, scikit-image.

# Setup Instructions
## 1. Clone the Repository
Clone the repository to your local machine using:

`git clone https://github.com/.............git`

`cd your-repository-directory`


## 2. Download the Machine Learning Models
You need to download the machine learning models according to their respective documentation. 

The DL_Track_US package provides an easy to use graphical user interface (GUI) for deep learning based analysis of muscle architectural parameters from longitudinal ultrasonography images of human lower limb muscles. Please take a look at our [documentation][3] for more information (note that aggressive ad-blockers might break the visualization of the repository description as well as the online documentation).

Model 1 (US_Track_US): https://dltrack.readthedocs.io/en/latest/index.html

Model 1.2 (US_Track_US_Example): https://dltrack.readthedocs.io/en/latest/installation.html (refer to the section - Download the DL_Track_US executable). From here, download the model-fasc and model-apo files in h5 format.

Model 2: [Model 2 Documentation Link]

Ensure that the models are correctly downloaded.

[3]: https://dltrack.readthedocs.io/en/latest/index.html

## 3. Install Required Libraries
Install the required libraries using pip. Run the following command in your terminal:
`pip install -r requirements.txt`

## 4. Run the Analysis Script (do_our_calculations)
To analyze a video, you need to run the do_our_calculations.py script. 
This script will process the input video using the machine learning model (1.a) and output the analyzed video to the specified path.

Usage:
Feel free to update the paths based on where you saved and want to save the videos. Keep in mind that the output videos should 
be saved in the static folder. 

- input_video_path: Path to the input video you want to analyze. 

- output_video_path: Path where you want to save the analyzed video. 

- bar_graph_video_path: Path where you want to save the bar graph video.

- apo_model_path: Path to the APO model.

- fasc_model_path: Path to the FASC model.

Follow all the comments made in this file. 

Download:

- https://www.gyan.dev/ffmpeg/builds/

- https://ffmpeg.org/download.html

## 5. Running the Dashboard
After you have run the analysis script and generated the analyzed video, you can start the dashboard to view the output.

## 6. Start the Dashboard

The dashboard will be accessible at  http://127.0.0.1:5000. Open this URL in your web browser to view the dashboard.

## 7. Directory Structure
```
├── DL_Track_US  

├── DL_Track_US_Example      # Directory containing input video, apo and fasc models 

├── static                   # Directory containing all videos and pictures used in the dashboard

│   └── all videos and pictures     

├── do_our_calculations.py   # Script to run the video analysis

├── app3.py                  # Script to start the dashboard

├── requirements.txt         # List of required libraries

└── README.md                # This README file

├── templates                # Directory containing HTML templates

│   ├── index.html           # Main dashboard page

│   └── info.html            # Information page

└── .gitignore
```

## 8. Conclusion
This README has provided the necessary steps to set up and run the video analysis dashboard. 
Follow the instructions carefully to ensure that everything is set up correctly. 
If you encounter any issues, please refer to the documentation of the respective machine learning models or reach out for support. 
Enjoy analyzing your videos with our dashboard!
