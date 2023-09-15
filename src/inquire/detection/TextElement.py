import logging

import cv2
import numpy

from inquire.detection.ButtonState import ButtonState
from inquire.detection.RecognizedElement import RecognizedElement

try:
    from robot.api.logger import info

    robot_logger = True
except ImportError:
    robot_logger = False
    logging.basicConfig(level=logging.INFO)


class TextElement(RecognizedElement):
    """
    TextElement Class. It's used to store the information about the found Text, like it's state,...
    """

    def __init__(self, text: str | None,
                 image: numpy.ndarray,
                 rectangle: tuple = (0, 0, 0, 0),
                 confidence: float | None = None,
                 colours: list | None = None,
                 state: ButtonState | None = None,
                 binary_image: numpy.ndarray | None = None,
                 render_bin: int = 0):
        super().__init__(image=image,
                         rectangle=rectangle,
                         confidence=confidence,
                         colours=colours,
                         state=state,
                         binary_image=binary_image,
                         render_bin=render_bin)
        self.text = text

    def __str__(self):
        if self.rectangle != (0, 0, 0, 0):
            return f'({self.text}, conf:{self.confidence}, rec:{self.rectangle})'
        else:
            return f'({self.text}, conf:{self.confidence})'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def concatenate_horizontal(img_list: list[numpy.ndarray]) -> numpy.ndarray:
        h_min = min(img.shape[0] for img in img_list)
        im_list_resize = []
        for img in img_list:
            if img.size == 0:
                if robot_logger:
                    info(f'textelement merging encountered empty fields', html=True)
                else:
                    logging.info(f'textelement merging encountered empty fields')
                continue
            width = int(img.shape[1] * h_min / img.shape[0])
            resized_img = cv2.resize(img, (width, h_min), interpolation=cv2.INTER_CUBIC)
            im_list_resize.append(resized_img)
        # im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min),
        #                              interpolation=cv2.INTER_CUBIC)
        #                   for img in img_list]
        concat = cv2.hconcat(im_list_resize)
        return concat

    def do_they_align_horizontally(self, element: 'TextElement') -> bool:
        if (element.rectangle[1] + element.rectangle[3] < self.rectangle[1]) or (
                element.rectangle[1] > self.rectangle[1] + self.rectangle[3]):
            return False
        else:
            return True

    def do_they_align_vertically(self, element: 'TextElement') -> bool:
        if (element.rectangle[0] + element.rectangle[2] < self.rectangle[0]) or (
                element.rectangle[0] > self.rectangle[0] + self.rectangle[2]):
            return False
        else:
            return True

    def append_text_element(self, element: 'TextElement') -> None:
        """
        TODO img, rectangle
        Append the text and confidence level of the given text element to the current one. (Merges)
        :param element: Another TextElement to append
        :type element: TextElement
        :return:
        :rtype: None
        """
        if element.text is not None:
            self.text += " " + element.text
        if element.confidence is not None:
            self.confidence = (self.confidence + element.confidence) / 2
        self.rectangle = (min(self.rectangle[0], element.rectangle[0]),
                          min(self.rectangle[1], element.rectangle[1]),
                          self.rectangle[2] + element.rectangle[2],
                          self.rectangle[3] + element.rectangle[3])
        self.image = self.concatenate_horizontal([self.image, element.image])
