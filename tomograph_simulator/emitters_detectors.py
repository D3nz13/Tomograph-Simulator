import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from bresenham import bresenham
from helpers import read_pixels_on_a_line, create_kernel


class EmittersDetectors:
    def __init__(self, n: int, alpha: float, span: float, iterations: int, image: np.ndarray, filtered: bool=False) -> None:
        """
        Args:
            n (int): number of emitters/detectors
            alpha (float): the angle difference between each rotation (in degrees)
            span (float): the range within the emitters/detectors are positioned (in degrees)
            iterations (int): number of rotations
            img (np.array): the image, should be grayscale
            filtered (bool): whether to apply sinogram filtering or not
        """

        if n <= 0 or not(isinstance(n, int)):
            raise Exception("The number of emitters/detectors should be a positive integer")
        if len(image.shape) != 2:
            raise Exception("The image should be grayscale")
        if iterations <= 0 or not(isinstance(iterations, int)):
            raise Exception("The number of iterations should be a positive integer")


        self._img = image
        self._num = n
        self._iterations = iterations
        self._height, self._width = self._img.shape
        self._alpha = alpha
        self._span = span
        self._filtered = filtered
        self._radius = max(self._height//2, self._width//2)
        self._angles = np.arange(0, self._span, self._span/self._num)

        self._emitters, self._detectors = self._initialize_positions_ellipse()
    

    def _initialize_positions_ellipse(self) -> tuple:
        """Initializes emitters and detectors positions. They are formed into an elliptical shape

        Returns:
            tuple: emitters and detectors positions
        """
        middle_h = int(self._height/2)
        middle_w = int(self._width/2)


        emitters_x = ((middle_w * np.cos(np.radians(self._angles))) + middle_w).astype(int)
        emitters_y = ((middle_h * np.sin(np.radians(self._angles))) + middle_h).astype(int)

        emitters_x = np.clip(emitters_x, 0, self._width - 1)
        emitters_y = np.clip(emitters_y, 0, self._height - 1)

        detectors_x = (self._width - emitters_x)[::-1]
        detectors_y = (self._height - emitters_y)[::-1]

        detectors_x = np.clip(detectors_x, 0, self._width - 1)
        detectors_y = np.clip(detectors_y, 0, self._height - 1)

        return np.stack((emitters_x, emitters_y), axis=-1), np.stack((detectors_x, detectors_y), axis=-1)
    

    def _update_positions(self) -> None:
        """Updates emitters/detectors positions (rotates them by alpha)
        """
        self._angles += self._alpha

        self._emitters, self._detectors = self._initialize_positions_ellipse()


    def _apply_bresenham(self) -> list:
        """Applies bresenham algorithm for each corresponding pair of emitters and detectors.

        Returns:
            list: a list containing points that make up a line between the corresponding pairs
        """
        pairs = zip(self._emitters, self._detectors)

        lines = []
        for e_pos, d_pos in pairs:
            line = bresenham(e_pos, d_pos)

            line[:, 0] = np.clip(line[:, 0], 0, self._width - 1)
            line[:, 1] = np.clip(line[:, 1], 0, self._height - 1)
            lines.append(line)

        return lines
    

    def _calculate_avg_pixels_for_a_line(self, line: np.ndarray) -> float:
        """Calculates average value of pixels along a line

        Args:
            line (np.array): array containing coordinates of line points

        Returns:
            float: average pixel value
        """
        return np.mean(read_pixels_on_a_line(self._img, line))
    

    def _create_sinogram_row(self) -> list:
        """Creates a single row for sinogram

        Returns:
            list: list containing average values of all lines between each emitter-detector pair
        """
        lines = self._apply_bresenham()

        return [self._calculate_avg_pixels_for_a_line(line) for line in lines]
    

    def create_sinogram(self) -> np.ndarray:
        """Creates normalized sinogram

        Returns:
            np.array: sinogram
        """
        sinogram_rows = []

        for _ in range(self._iterations):
            sinogram_rows.append(self._create_sinogram_row())
            self._update_positions()
        
        sinogram = np.array(sinogram_rows)

        if self._filtered:
            return self._filter_sinogram(sinogram)
        return sinogram
    

    def _filter_sinogram(self, sinogram: np.ndarray) -> np.ndarray:
        filtered_sinogram = np.zeros_like(sinogram)
        rows = sinogram.shape[0]
 
        kernel = create_kernel(length=int(self._num/20)+2)
        
        for i in range(rows):
            filtered_sinogram[i, :] = np.convolve(sinogram[i, :], kernel, mode='same')
        
        return filtered_sinogram
    

    def reverse_sinogram(self, sinogram: np.ndarray) -> np.ndarray:
        """Create image reconstruction from a sinogram.

        Args:
            sinogram (np.ndarray): a sinogram that will be used to reconstruct the image

        Returns:
            np.ndarray: the reconstructed image
        """
        parent_path = "../results"

        if not os.path.isdir(parent_path):  # creating a directory for saving the results
            os.mkdir(parent_path)
        
        for file in os.listdir(parent_path):  # clearing the directory
            os.remove(f"{parent_path}/{file}")

        result = np.zeros_like(self._img, dtype=float)

        self._emitters, self._detectors = self._initialize_positions_ellipse()

        for iteration in range(self._iterations):
            for idx, (emitter, detector) in enumerate(zip(self._emitters, self._detectors)):
                line = bresenham(emitter, detector)

                for pos_x, pos_y in line:
                    result[pos_y, pos_x] += sinogram[iteration, idx]

            normalized_result = 255*(result - np.percentile(result, 5))/(np.percentile(result, 95) - np.percentile(result, 5))
            normalized_result = np.clip(normalized_result, 0, 255)
            cv2.imwrite(f"{parent_path}/{iteration+1:03d}.jpg", normalized_result)

            self._update_positions()

        normalized_result = 255*(result - np.percentile(result, 5))/(np.percentile(result, 95) - np.percentile(result, 5))

        return np.clip(normalized_result, 0, 255)



if __name__ == "__main__":
    sample_file_path = "../images/SADDLE_PE.JPG"
    emitter = EmittersDetectors(n=180, alpha=2, span=180, iterations=int(360/2), image=cv2.imread(sample_file_path, cv2.IMREAD_GRAYSCALE), filtered=True)
    sinogram = emitter.create_sinogram()
    reconstruction = emitter.reverse_sinogram(sinogram)

    plt.subplot(1, 2, 1)
    plt.imshow(sinogram, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstruction, cmap='gray')
    plt.show()
    