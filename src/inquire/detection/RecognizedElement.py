import numpy
import cv2
from inquire.detection.ButtonState import ButtonState
from inquire.detection.TemplateMatcher import TemplateMatcher
from inquire.helper.Comparator import Comparator


class RecognizedElement:
    """
    Abstract class for the recognized Elements like Text and Symbols of the GUI
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self, image: numpy.ndarray,
                 rectangle: tuple = (0, 0, 0, 0),
                 confidence: float | None = None,
                 colours: list | None = None,
                 state_colours: dict | None = None,
                 state: ButtonState | None = None,
                 binary_image: numpy.ndarray | None = None,
                 render_bin: int = 0):
        self.confidence = confidence
        self.colours = colours
        self.state_colours: dict[ButtonState, list[list[tuple[int, int, int]]]] = state_colours
        self.state = state
        self.rectangle = rectangle
        self.image = image
        if binary_image is not None:
            self.binary_image = binary_image
        else:
            if render_bin == 1:
                self.binary_image = TemplateMatcher.generate_binary_image(image)
            elif render_bin == 4:
                self.binary_image = TemplateMatcher.generate_binary_image_4bit(image)
            else:
                self.binary_image = None

    def set_state_colours_dic(self, states: dict[ButtonState, list[list[tuple[int, int, int]]]]):
        self.state_colours = states

    def get_button_state(self) -> None | ButtonState:
        if self.state is None:
            self.find_button_state()
        return self.state

    def find_button_state(self) -> bool:
        if self.state_colours is None:
            return False
        if self.state_colours:
            self.analyse_colours_of_img(10)
            for key, value in self.state_colours.items():
                for colours in value:
                    found = 0
                    for colour1 in colours:
                        for colour2 in self.colours:
                            if Comparator.is_similar_colour(colour2, colour1):
                                found += 1
                                break
                    if len(colours) == found:
                        self.state = key
                        return True

    def get_center_point(self):
        """
        Returns the center point (x,y) coordinates of the Element
        :return: (x,y) coordinates
        :rtype: tuple[int,int]
        """
        return self.rectangle[0] + int(self.rectangle[2] / 2), self.rectangle[1] + int(self.rectangle[3] / 2)

    def get_width_and_height(self):
        """
        Returns the Size of the Element
        :return: width,height
        :rtype: tuple[int,int]
        """
        return self.rectangle[2], self.rectangle[3]

    def analyse_colours_of_img(self, t: int = 3):
        """
        Extracts from the bgr image the most used rgb colours
        :param t: Amount of colours to be stored
        :type t: int
        :return: Colour list [(r,g,b),...]
        :rtype: list[int,int,int]
        """
        n = 10
        img = self.image.copy()
        # flatten 3D Array to 2D Array
        pixels = img.reshape((-1, 3)).astype(numpy.float32)

        # k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(pixels, n, None, criteria, 10, flags)

        # Retrieve Colours
        colours = centers.astype(numpy.uint8)
        freq = numpy.unique(labels, return_counts=True)[1]
        sorted_freq_idx = numpy.argsort(freq)[::-1]
        self.colours = colours[sorted_freq_idx]
        # switch from bgr to rgb
        # self.colours[:, [2, 0]] = self.colours[:, [0, 2]]
        self.colours = self.colours[:t]
        return self.colours
