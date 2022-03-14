import matplotlib.pyplot as plt
import streamlit as st
import glob
from emitters_detectors import EmittersDetectors
import cv2
import numpy as np
from scipy.ndimage import rotate

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
    fig, ax = plt.subplots()
    ax.imshow(sinogram, cmap="gray")
    st.write('## Sinogram')
    st.pyplot(fig)
    
    fft_sinogram = np.fft.fft(sinogram)
    fig2, ax2 = plt.subplots()
    ax2.imshow(fft_sinogram.real, cmap="gray")
    st.write('## Fourier Transform Sinogram')
    st.pyplot(fig2)
    
    diameter = len(fft_sinogram[0])
    circle_angle = len(fft_sinogram)
    xx, yy = np.mgrid[:diameter, :diameter]
    circle = (xx - diameter//2)**2 + (yy - diameter//2)**2
    fig3, ax3 = plt.subplots()
    rotated = rotate(circle, angle=180//circle_angle, reshape=False)
    ax3.imshow(rotated, cmap="gray")
    st.write('## Circle')
    st.pyplot(fig3)
