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

num_of_emitters = st.number_input('Number of emitters', 1, 100, 50)

alpha_angle = st.number_input('Angle', 1, 360, 6)

span = st.number_input('Span', 1, 100, 30)

iterations = st.number_input('Number of iterations', 1, 100, 60)


if st.button('Run simulation', 'btn-1'):
    sinogram = None
    with st.spinner("Simulation is running..."):
        emitter = EmittersDetectors(num_of_emitters, alpha_angle, span, iterations, image=cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
        sinogram = emitter.create_sinogram()

    col1, col2 = st.columns(2)

    fig, ax = plt.subplots()
    ax.imshow(sinogram, cmap="gray")
    col1.write('## Sinogram')
    col1.pyplot(fig)
    
    fft_sinogram = np.fft.fft(sinogram)
    fig2, ax2 = plt.subplots()
    ax2.imshow(fft_sinogram.real, cmap="gray")
    col2.write('## Fourier Sinogram')
    col2.pyplot(fig2)
    
    circle = make_circle(fft_sinogram, alpha_angle)
    fig3, ax3 = plt.subplots()
    ax3.imshow(circle.real, cmap="gray")
    st.write('## Circle')
    st.pyplot(fig3)
    # diameter = len(fft_sinogram[0])
    # circle_angle = len(fft_sinogram)
    # xx, yy = np.mgrid[:diameter, :diameter]
    # circle = (xx - diameter//2)**2 + (yy - diameter//2)**2
    # rotated = rotate(circle, angle=180//circle_angle, reshape=False)
