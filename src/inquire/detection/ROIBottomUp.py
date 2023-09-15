import cv2
import numpy
from inquire.detection.ROIDetector import ROIDetector


class ROIBottomUp(ROIDetector):
    """
    TODO a lot
    Class to manage the topdown approach for the region of interest detection
    """

    def __init__(self):
        super().__init__()

    def contours(self):
        img = self._img.copy()
        c = self._dilated_img.astype(numpy.int32)
        contours, hierarchy = cv2.findContours(c, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("gray", self._dilated_img)
        cv2.imshow("flood", img)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pass