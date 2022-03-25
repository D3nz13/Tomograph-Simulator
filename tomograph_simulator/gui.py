from cProfile import label
import matplotlib.pyplot as plt
import streamlit as st
import glob
from emitters_detectors import EmittersDetectors
import cv2

st.write('# Tomograph Simulator')

form1 = st.form(key='main_data')

files = glob.glob('../images/*')
filename = form1.selectbox('Filename', files)
num_of_emitters = form1.number_input('Number of emitters', 1, 150, 50)
alpha_angle = form1.number_input('Angle', 1, 20, 1)
span = form1.number_input('Span', 1, 180, 15)
full_rotation = form1.checkbox("Full rotation (360 degrees)", True)

if not full_rotation:
    iterations = form1.number_input('Number of iterations', 1, 100, 60)
else:
    iterations = int(360/alpha_angle)

show_all_iterations = form1.checkbox("Show all iterations", True)

image = None
sinogram = None
reconstruction = None
last_set_iterations = iterations

submit_button1 = form1.form_submit_button(label='Run simulation')

if submit_button1:
    with st.spinner("Simulation is running..."):
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        emitter = EmittersDetectors(num_of_emitters, alpha_angle, span, iterations, image)
        sinogram = emitter.create_sinogram()
        reconstruction = emitter._reverse_sinogram(sinogram)
        st.session_state['image'] = image
        st.session_state['sinogram'] = sinogram
        st.session_state['reconstruction'] = reconstruction
elif 'image' in st.session_state and 'sinogram' in st.session_state and 'reconstruction' in st.session_state:
    image = st.session_state['image']
    sinogram = st.session_state['sinogram']
    reconstruction = st.session_state['reconstruction']

if image is not None and sinogram is not None and reconstruction is not None:
    col1, col2 = st.columns(2)

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
        st.write('## Reconstruction Image')
        st.pyplot(fig3)
    else:
        form2 = st.form(key="Iteration form")
        display_iteration = form2.slider("Iteration no.", 1, last_set_iterations, 1)
        submit_button2 = form2.form_submit_button(label="Change itteration")
        reconstruction_iteration = cv2.imread(f"../results/{display_iteration:03d}.jpg", cv2.IMREAD_GRAYSCALE)
        st.write('## Reconstruction Image')
        st.image(reconstruction_iteration, caption=f"Iteration {display_iteration}")
