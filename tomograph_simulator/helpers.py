import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
from pydicom.datadict import add_dict_entry

from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

def read_pixels_on_a_line(img: np.ndarray, line: np.ndarray) -> np.ndarray:
    return np.array([img[point_y, point_x] for point_x, point_y in line])


def create_kernel(length: int) -> np.ndarray:
    right = [1]
    for i in range(1, length):
        if i%2 == 0:
            right.append(0)
        else:
            right.append(-4/(np.pi**2*i**2))
    
    left = right[1::][::-1]
    left.extend(right)

    return np.array(left)

def convert_image_to_ubyte(img):
    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))

def save_as_dicom(file_name, img, patient_data):
    img_converted = convert_image_to_ubyte(img)
    
    # Populate required values for file meta information
    add_dict_entry(0x10021001, "DA", "ExaminationDate", "Examination Date")

    meta = Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, preamble=b"\0" * 128)
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    
    ds.PatientName = patient_data["PatientName"]
    ds.PatientID = patient_data["PatientID"]
    ds.PatientBirthDate = patient_data["PatientBirthDate"]
    ds.PatientSex = patient_data["PatientSex"]
    ds.ImageComments = patient_data["ImageComments"]
    ds.ExaminationDate = patient_data["ExaminationDate"]
    

    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SamplesPerPixel = 1
    ds.HighBit = 7

    ds.ImagesInAcquisition = 1
    ds.InstanceNumber = 1

    ds.Rows, ds.Columns = img_converted.shape

    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    ds.PixelData = img_converted.tobytes()
    print(ds.dir())
    ds.save_as(f"../dicom_files/{file_name}", write_like_original=False)

def read_dicom(filepath):
    ds = pydicom.dcmread(filepath)
    image = ds.pixel_array
    return ds, image