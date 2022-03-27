import matplotlib.pyplot as plt
import streamlit as st
import glob
from emitters_detectors import EmittersDetectors
import cv2
import datetime
from helpers import save_as_dicom, read_dicom

st.write('# Tomograph Simulator')

st.sidebar.write("## Tomograph Control Panel")
form1 = st.sidebar.form(key='main_data')

files = glob.glob('../images/*')
filename = form1.selectbox('Filename', files)
num_of_emitters = form1.number_input('Number of emitters', 1, 150, 50)
alpha_angle = form1.number_input('Angle', 1, 20, 1)
span = form1.number_input('Span', 1, 180, 15)
filtered = form1.checkbox("Filter", False)
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
        emitter = EmittersDetectors(num_of_emitters, alpha_angle, span, iterations, image, filtered)
        sinogram = emitter.create_sinogram()
        reconstruction = emitter.reverse_sinogram(sinogram)
        st.session_state['image'] = image
        st.session_state['sinogram'] = sinogram
        st.session_state['reconstruction'] = reconstruction
elif 'image' in st.session_state and 'sinogram' in st.session_state and 'reconstruction' in st.session_state:
    image = st.session_state['image']
    sinogram = st.session_state['sinogram']
    reconstruction = st.session_state['reconstruction']

if image is not None and sinogram is not None and reconstruction is not None:
    st.write("## Results")
    with st.expander("Click to expand"):

        col1, col2 = st.columns(2)

        col1.write('## Base image')
        col1.image(image)

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
        st.write('## Reconstruction iterations')
        form2 = st.form(key="Iteration form")
        display_iteration = form2.slider("Iteration no.", 1, last_set_iterations, 1)
        submit_button2 = form2.form_submit_button(label="Change iteration")
        reconstruction_iteration = cv2.imread(f"../results/{display_iteration:03d}.jpg", cv2.IMREAD_GRAYSCALE)
        st.write('## Reconstruction Image')
        st.image(reconstruction_iteration, caption=f"Iteration {display_iteration}")

    st.write("# Patient Form")
    with st.form(key="Patient Form"):
        pt_id = st.text_input(label="ID")
        fs_name = st.text_input(label="First Name")
        ls_name = st.text_input(label="Last Name")
        sex = st.radio("Sex", ('Male', 'Female'))
        birthdate = st.date_input(label="Birth Date", value=datetime.date.today(), min_value=datetime.date(1900, 1, 1))
        image_comments = st.text_area(label="Image comments")
        examination_date = st.date_input(label="Examination Date", value=datetime.date.today())
        dicom_filename = st.text_input(label="Filename")
        submit_patient = st.form_submit_button(label="Save as DICOM")

        if submit_patient:
            patient_data = {
                "PatientID": pt_id,
                "PatientName": f"{ls_name} {fs_name}",
                "PatientBirthDate": birthdate,
                "PatientSex": sex,
                "ImageComments": image_comments,
                "ExaminationDate": examination_date
            }
            save_as_dicom(f"{dicom_filename}.dicom", reconstruction, patient_data)

st.sidebar.write("## Load DICOM file")

patient_read = None

with st.sidebar.form(key="DICOM read form"):
    file = st.file_uploader(label="Pick dicom file to load", type=['dicom'], accept_multiple_files=False)
    submit_read = st.form_submit_button(label="Load data")
    if submit_read and file is not None:
        file_data = file.getvalue()
        with open(f"../dicom_files/{file.name}", "wb") as f:
            f.write(file_data)
        patient_read = f"../dicom_files/{file.name}"

if patient_read:
    ds, patient_image = read_dicom(patient_read)
    filename = patient_read.split("/")[-1]
    st.write(f"##  Patient data from *{filename}*")
    st.write(f"#### Patient ID: *{ds.PatientID}*")
    st.write(f"#### Patient Name: *{ds.PatientName}*")
    st.write(f"#### Patient Birth Date: *{ds.PatientBirthDate}*")
    st.write(f"#### Patient Sex: *{ds.PatientSex}*")
    if "ExaminationDate" in ds:
        st.write(f"#### Examination Date: *{ds.ExaminationDate}*")
    st.image(patient_image, "Patient Tomograph Image")
    st.write(f"#### Image notes: *{ds.ImageComments}*")
