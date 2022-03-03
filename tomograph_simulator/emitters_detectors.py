import numpy as np
import matplotlib.pyplot as plt


class EmittersDetectors:
    def __init__(self, n: int, alpha: float, img_shape: tuple, ellipse = True) -> None:
        """
        Args:
            n (int): number of emitters/detectors
            alpha (float): the angle between another emitters/detectors (in degrees)
            img_shape (tuple): the image shape (the image should be grayscale)
        """

        if n <= 0 or not(isinstance(n, int)):
            raise Exception("The number of emitters/detectors should be a positive integer")

        (h, w) = img_shape
        self._num = n
        self._height = h
        self._width = w
        self._alpha = alpha
        self._radius = max(self._height//2, self._width//2)
        
        if ellipse:
            self._emitters, self._detectors = self._initialize_positions_ellipse()
        else:
            self._emitters, self._detectors = self._initialize_positions_clipped_circle()
    

    def _initialize_positions_ellipse(self) -> tuple:
        """Initializes emitters and detectors positions. They are formed into an elliptical shape

        Returns:
            tuple: emitters and detectors positions
        """
        middle_h = int(self._height/2)
        middle_w = int(self._width/2)

        angles = np.arange(0, self._num*self._alpha, self._alpha)

        emitters_x = ((middle_w * np.cos(np.radians(angles))) + middle_w).astype(int)
        emitters_y = ((middle_h * np.sin(np.radians(angles))) + middle_h).astype(int)

        emitters_x = np.clip(emitters_x, 0, self._width - 1)
        emitters_y = np.clip(emitters_y, 0, self._height - 1)

        detectors_x = self._width - emitters_x
        detectors_y = self._height - emitters_y

        return np.stack((emitters_x, emitters_y), axis=-1), np.stack((detectors_x, detectors_y), axis=-1)


    def _initialize_positions_clipped_circle(self) -> tuple:
        """Initializes emitters and detectors positions. They are formed into an circular shape,
        if it caused exceeding the image size, the position will be clipped to fit into.

        Returns:
            tuple: emitters and detectors positions
        """
        middle_h = int(self._height/2)
        middle_w = int(self._width/2)

        angles = np.arange(0, self._num*self._alpha, self._alpha)

        emitters_x = ((self._radius * np.cos(np.radians(angles))) + middle_w).astype(int)
        emitters_y = ((self._radius * np.sin(np.radians(angles))) + middle_h).astype(int)

        emitters_x = np.clip(emitters_x, 0, self._width - 1)
        emitters_y = np.clip(emitters_y, 0, self._height - 1)

        detectors_x = self._width - emitters_x
        detectors_y = self._height - emitters_y

        return np.stack((emitters_x, emitters_y), axis=-1), np.stack((detectors_x, detectors_y), axis=-1)


if __name__ == "__main__":
    emitter = EmittersDetectors(72, 5, (100, 200))
    e_x, e_y = emitter._emitters[:, 0], emitter._emitters[:, 1]
    d_x, d_y = emitter._detectors[:, 0], emitter._detectors[:, 1]
    plt.scatter(e_x, e_y)
    plt.xlim([0, 200])
    plt.ylim([0, 200])
    plt.scatter(d_x, d_y)
    plt.show()
    