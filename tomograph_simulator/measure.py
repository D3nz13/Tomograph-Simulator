import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
import pandas as pd

from emitters_detectors import EmittersDetectors

if __name__ == "__main__":
    img = cv2.imread("../images/SADDLE_PE.JPG", cv2.IMREAD_GRAYSCALE)

    detectors_num_default = 180
    iterations_num_default = 180
    span_default = 180

    detectors_num = np.arange(90, 721, 90)
    iterations_num = np.arange(90, 721, 90)
    spans_num = np.arange(45, 271, 45)
    print("Creating an empty dataframe..")
    results = pd.DataFrame(data=[], columns=["Detectors", "Iterations", "Span", "RMSE", "Group"])
    idx = 0

    print("Detectors..")
    for det_num in detectors_num:
        print(f"Current settings: detectors = {det_num}")
        tomograph = EmittersDetectors(n=det_num, alpha=360/iterations_num_default, span=span_default, iterations=iterations_num_default, image=img, filtered=True)
        sinogram = tomograph.create_sinogram()
        reconstruction = tomograph.reverse_sinogram(sinogram)

        rmse = mean_squared_error(img, reconstruction, squared=False)

        results.loc[idx] = [det_num, iterations_num_default, span_default, rmse, "Detectors"]
        idx += 1
        results.to_csv("results.csv", index=False)

    print("Iterations..")
    for iter_num in iterations_num:
        print(f"Current settings: iterations = {iter_num}")
        tomograph = EmittersDetectors(n=detectors_num_default, alpha=360/iter_num, span=span_default, iterations=iter_num, image=img, filtered=True)
        sinogram = tomograph.create_sinogram()
        reconstruction = tomograph.reverse_sinogram(sinogram)

        rmse = mean_squared_error(img, reconstruction, squared=False)

        results.loc[idx] = [detectors_num_default, iter_num, span_default, rmse, "Iterations"]
        idx += 1
        results.to_csv("results.csv", index=False)

    print("Spans..")
    for span_num in spans_num:
        print(f"Current settings: span = {span_num}")
        tomograph = EmittersDetectors(n=detectors_num_default, alpha=360/iterations_num_default, span=span_num, iterations=iterations_num_default, image=img, filtered=True)
        sinogram = tomograph.create_sinogram()
        reconstruction = tomograph.reverse_sinogram(sinogram)

        rmse = mean_squared_error(img, reconstruction, squared=False)

        results.loc[idx] = [detectors_num_default, iterations_num_default, span_num, rmse, "Span"]
        idx += 1
        results.to_csv("results.csv", index=False)
    print("Finished !")