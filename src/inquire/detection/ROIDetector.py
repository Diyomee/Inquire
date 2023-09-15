import cv2
import numpy
from inquire.helper.Comparator import Comparator
from inquire.helper.Utilities import Utilities


class ROIDetector:
    """
    Abstract class for the detection of region of interest.
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self):
        self._img = None
        self._gray_img = None
        self._binary_img = None
        self._dilated_img = None
        self._block_segments = []
        self.draw_ocr = False

    def set_all_4_images(self,
                         img: numpy.ndarray,
                         gray: numpy.ndarray,
                         bin_img: numpy.ndarray,
                         dilated: numpy.ndarray) -> None:
        """
        Sets all 4 images
        :param img: bgr image
        :type img: numpy.ndarray
        :param gray: gray image
        :type gray: numpy.ndarray
        :param bin_img: binary image
        :type bin_img: numpy.ndarray
        :param dilated: dilated binary image
        :type dilated: numpy.ndarray
        :return: None
        :rtype: None
        """
        self._img = img
        self._gray_img = gray
        self._binary_img = bin_img
        self._dilated_img = dilated

    def set_image(self, img: numpy.ndarray) -> None:
        """
        save the given rgb image in a local variable for further works.
        Generates a binary and gray image from the given image.
        :param img: rgb image
        :type img: numpy.ndarray
        :return: None
        :rtype: None
        """
        self._img = img.copy()
        self._gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        # self.__binary_img = cv2.adaptiveThreshold(self.__gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        #                                           11, 4)
        self._binary_img = cv2.Laplacian(self._gray_img.copy(), cv2.CV_64F)
        self._binary_img[abs(self._binary_img) > 0.0] = 255.0
        self._binary_img = numpy.uint8(numpy.absolute(self._binary_img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._dilated_img = self._binary_img.copy()
        self._dilated_img = cv2.bitwise_not(self._dilated_img)
        self._dilated_img = cv2.morphologyEx(self._binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    def get_images(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """
        Returns a reference to all 4 saved images
        :return: 4 images
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        return self._img, self._gray_img, self._binary_img, self._dilated_img

    @staticmethod
    def generate_dilated_image(img):
        dil = cv2.Laplacian(img, cv2.CV_64F)
        dil[abs(dil) > 0.0] = 255.0
        dil = numpy.uint8(numpy.absolute(dil))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dil = cv2.bitwise_not(dil)
        dil = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel, iterations=2)
        return dil

    def show_image(self):
        """
        Opens four opencv windows to show the current image
        :return: None
        :rtype: None
        """
        cv2.imshow("img", self._img)
        cv2.imshow("gray", self._gray_img)
        cv2.imshow("bin", self._binary_img)
        cv2.imshow("dil", self._dilated_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    @staticmethod
    def merge_overlapping_rectangles(rectangles):
        # TODO 1.6
        to_merge = {}
        for j in range(len(rectangles) - 1):
            merge = set()
            for k in range(j + 1, len(rectangles)):
                if Comparator.intersect_rectangles(rectangles[k], rectangles[j]):
                    merge.add(j)
                    merge.add(k)
            if merge:
                to_merge[j] = merge
        skip = set()
        temp = []
        for j in range(len(rectangles)):
            if j in to_merge:
                rec = [0, 0, 0, 0]
                for k in to_merge[j]:
                    rec = Utilities.merge_rectangles(rec, rectangles[k])
            else:
                rec = rectangles[j]
            temp.append(rec)
        return temp
