import matplotlib.pyplot as plt
import streamlit as st
import glob
from emitters_detectors import EmittersDetectors
import cv2
import numpy as np
from helpers import make_circle

st.write('# Tomograph Simulator')

files = glob.glob('../images/*')

filename = st.selectbox('Filename', files)

num_of_emitters = st.number_input('Number of emitters', 1, 150, 50)

alpha_angle = st.number_input('Angle', 1, 20, 1)

span = st.number_input('Span', 1, 180, 15)

full_rotation = st.checkbox("Full rotation (360 degrees)", True)

if not full_rotation:
    iterations = st.number_input('Number of iterations', 1, 100, 60)
else:
    iterations = int(360/alpha_angle)

show_all_iterations = st.checkbox("Show all iterations", True)

image = None
sinogram = None
reconstruction = None
last_set_iterations = iterations

if st.button('Run simulation', 'btn-1'):
    with st.spinner("Simulation is running..."):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        emitter = EmittersDetectors(num_of_emitters, alpha_angle, span, iterations, image)
        sinogram = emitter.create_sinogram()
        reconstruction = emitter._reverse_sinogram(sinogram)

if image is not None and sinogram is not None and reconstruction is not None:
    col1, col2, col3 = st.columns(3)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    col1.write('## Base image')
    col1.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.imshow(sinogram, cmap="gray")
    col2.write('## Sinogram')
    col2.pyplot(fig2)

    if not show_all_iterations:
        fig3, ax3 = plt.subplots()
        ax3.imshow(reconstruction, cmap="gray")
        col3.write('## Reconstruction Image')
        col3.pyplot(fig3)
    else:
        display_iteration = st.slider("Iteration no.", 1, last_set_iterations, 1)
        reconstruction_iteration = cv2.imread(f"../results/{display_iteration:03d}.jpg", cv2.IMREAD_GRAYSCALE)
        fig3, ax3 = plt.subplots()
        ax3.imshow(reconstruction_iteration, cmap="gray")
        col3.write('## Reconstruction Image')
        col3.pyplot(fig3)
