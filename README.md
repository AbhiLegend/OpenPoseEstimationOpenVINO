## What the  app does

This project involves developing a Streamlit-based web application that integrates advanced technologies like Stable Diffusion, ControlNet, OpenPose, and OpenVINO for real-time human pose estimation. Users can upload images to the app, which then employs OpenPose for detecting human figures and their poses. The app further utilizes the combined capabilities of Stable Diffusion and ControlNet to modify or highlight these poses. The final output displays the original image alongside a version with pose annotations, offering an interactive and visual representation of human poses. This setup not only showcases the intersection of deep learning and image processing but also makes advanced pose estimation techniques accessible to users via a simple web interface.

## How to run the app
To run the Streamlit app described in the script, you would typically follow these steps:

Environment Setup:

Ensure you have Python installed on your computer. If not, you can download and install it from python.org.
It's recommended to use a virtual environment for Python projects. This keeps dependencies required by different projects separate by creating isolated environments for them. You can create a virtual environment using Python’s built-in module venv.
Install Required Libraries:

You need to install Streamlit and other dependencies mentioned in the script (torch, PIL, numpy, matplotlib, diffusers, etc.). This can be done using pip, Python’s package installer.
Run pip install streamlit torch pillow numpy matplotlib diffusers openvino in your command line interface.
Obtain the Script:

Ensure you have the script saved as a .py file on your computer.
Run the Streamlit App:

Open your command line interface (CLI) and navigate to the directory where your script is located.
Run the command streamlit run your_script_name.py, replacing your_script_name.py with the actual name of your script file.
Streamlit should automatically open the app in your default web browser. If it doesn’t, the CLI will provide a local URL that you can paste into your browser to access the app.
Using the App:

Once the app is running in your browser, you can interact with it according to its functionality. In your case, this would involve uploading images and viewing the pose estimation results.

## Open VINO integration

Conda installation

pip install openvino
