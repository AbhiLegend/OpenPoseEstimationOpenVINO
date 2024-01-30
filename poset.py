import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector
from pathlib import Path
import openvino as ov

# Function to visualize pose results
def visualize_pose_results(orig_img, skeleton_img):
    """
    Helper function for pose estimation results visualization

    Parameters:
       orig_img (Image.Image): original image
       skeleton_img (Image.Image): processed image with body keypoints
    Returns:
       fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
    """
    orig_img = orig_img.resize(skeleton_img.size)
    orig_title = "Original image"
    skeleton_title = "Pose"
    im_w, im_h = orig_img.size
    is_horizontal = im_h <= im_w
    figsize = (20, 10) if is_horizontal else (10, 20)
    fig, axs = plt.subplots(2 if is_horizontal else 1, 1 if is_horizontal else 2, figsize=figsize, sharex='all', sharey='all')
    fig.patch.set_facecolor('white')
    list_axes = list(axs.flat)
    for a in list_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.grid(False)
    list_axes[0].imshow(np.array(orig_img))
    list_axes[1].imshow(np.array(skeleton_img))
    list_axes[0].set_title(orig_title, fontsize=15)
    list_axes[1].set_title(skeleton_title, fontsize=15)
    fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00 , hspace=0.01 if is_horizontal else 0.1)
    fig.tight_layout()

    # Display the figure in Streamlit
    st.pyplot(fig)





# Function to clean up Torchscript cache
def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

def main():
    st.title('Stable Diffusion ControlNet Pipeline with OpenPose')

    OPENPOSE_OV_PATH = Path("openpose1.xml")

    # Model loading with progress
    with st.spinner('Loading ControlNet Model...'):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
        st.success('ControlNet Model Loaded')

    with st.spinner('Loading Stable Diffusion Pipeline...'):
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet
        )
        st.success('Stable Diffusion Pipeline Loaded')

    with st.spinner('Loading OpenposeDetector Model...'):
        pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        st.success('OpenposeDetector Model Loaded')

    # OpenVINO model conversion
    if not OPENPOSE_OV_PATH.exists():
        with st.spinner('Converting OpenPose to OpenVINO format...'):
            with torch.no_grad():
                ov_model = ov.convert_model(pose_estimator.body_estimation.model, example_input=torch.zeros([1, 3, 184, 136]), input=[[1,3,184,136]])
                ov.save_model(ov_model, OPENPOSE_OV_PATH)
                del ov_model
                cleanup_torchscript_cache()
            st.success('OpenPose successfully converted to IR')
    else:
        st.write(f"OpenPose will be loaded from {OPENPOSE_OV_PATH}")

    # Image upload and processing
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner('Processing image...'):
            img = Image.open(uploaded_file)
            pose = pose_estimator(img)
            fig = visualize_pose_results(img, pose)
            st.success('Image processed')

            # Visualize the results
            st.pyplot(fig)

    st.write("Pose estimation and visualization completed.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
