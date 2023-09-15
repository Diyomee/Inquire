import copy
import queue

import numpy

from inquire.ocr.PaddleOCR import PaddleOCR
from inquire.detection.TextElement import TextElement
from inquire.helper.Comparator import Comparator


class Found(Exception):
    pass


class TextRecognizer:
    """
    Wrapper class for the different ocr methods
    1: PaddleOCR
    2: Custom Trained PaddleOCR
    3: Tesseract
    4: Easy OCR
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self, ocr_selection=1):
        self.ocr = None
        self.text_states = None
        self.ocr_selection = ocr_selection
        match ocr_selection:
            case 1:
                # paddleocr
                self.ocr = PaddleOCR()
            case 2:
                # paddleocr_trained
                raise NotImplementedError
            case 3:
                # Tesseract
                raise NotImplementedError
            case 4:
                # easy ocr
                raise NotImplementedError
            case _:
                raise NotImplementedError

    @staticmethod
    def align_text_to_lines(dict_t: dict[str, list[TextElement]]) -> dict[str, list[TextElement]]:
        n_dict_t = {}
        elements = []
        # Creates a list from all found TextElements
        for key, values in dict_t.items():
            for value in values:
                elements.append(value)
        if not elements:
            return n_dict_t
        # Sorts the list of Elements by their height
        elements = sorted(elements, key=Comparator.height_key_of_recognized_elements, reverse=True)
        visited = set()
        current_index = 0
        current_element = elements[0]
        align = []
        horizontal_candidates = queue.Queue()
        # creates lists of TextElements which have a common horizontal overlapping region
        while len(visited) < len(elements):
            selected = False
            for index, element in enumerate(elements):
                if index in visited:
                    continue
                else:
                    if not selected:
                        if align:
                            horizontal_candidates.put(align)
                            align = []
                        visited.add(index)
                        current_element = element
                        align.append(element)
                        selected = True
                    else:
                        if current_element.do_they_align_horizontally(element):
                            visited.add(index)
                            align.append(element)
        if align:
            horizontal_candidates.put(align)
        unique_horizontals = []
        # Splits the sets of vertically overlapping TextElements into multiple
        # Example: [1,2,3,4] and 2,3 are overlapping vertically in constructs [1,2,4] and [1,3,4]
        while horizontal_candidates.qsize() != 0:
            elements: list[TextElement] = horizontal_candidates.get()
            try:
                for i in range(len(elements) - 1):
                    for j in range(i + 1, len(elements)):
                        if elements[i].do_they_align_vertically(elements[j]):
                            e1 = copy.deepcopy(elements)
                            e1.pop(i)
                            e2 = copy.deepcopy(elements)
                            e2.pop(j)
                            horizontal_candidates.put(e1)
                            horizontal_candidates.put(e2)
                            raise Found
                elements = sorted(elements, key=Comparator.x_coord_key_of_recognized_elements, reverse=False)
                unique_horizontals.append(elements)
            except Found:
                horizontal_candidates.task_done()
            else:
                horizontal_candidates.task_done()
        # Merges the Found TextElements into new ones and stores them in a dictionary
        for unique in unique_horizontals:
            if unique:
                current_element = copy.deepcopy(unique[0])
                # [1:] removes the first element, since its already used above
                for index, element in enumerate(unique[1:], start=2):
                    current_element.append_text_element(element)
                if current_element.text in n_dict_t:
                    n_dict_t[current_element.text].append(current_element)
                else:
                    n_dict_t[current_element.text] = [current_element]
        # returns a dictionary of lines of text
        return n_dict_t

    def recognize_text(self, colour_img: numpy.ndarray, gray: numpy.ndarray,
                       binary_img: numpy.ndarray,
                       xx: int,
                       yy: int) -> dict[str, list[TextElement]]:
        """
        Uses the via self.ocr_selection specified ocr to find all text fields, and their bounding
        rectangles. This information is then wrapped in the class TextElement and stored in the
        returned dictionary
        :param colour_img:
        :type colour_img:
        :param gray: gray-scale img of the Region with the Text
        :type gray: numpy.ndarray
        :param binary_img: binary img og the region with the text
        :type binary_img: numpy.ndarray
        :param xx: x coordinate of the given image in the application window
        :type xx: int
        :param yy: y coordinate of the given image in the application window
        :type yy: int
        :return: dictionary of found text elements
        :rtype: dic[str,list[TextElement]]
        """
        ans = {}
        match self.ocr_selection:
            case w if w in {1, 2}:
                found_text = self.ocr.ocr_image(gray, binary_img, xx, yy)
                if found_text is not None:
                    for i in found_text:
                        rect = list(i[:4])
                        rect[0] -= xx
                        rect[1] -= yy
                        colour_img_2 = colour_img.copy()
                        colour_img_2 = colour_img_2[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                        text = TextElement(text=i[4],
                                           image=colour_img_2,
                                           binary_image=binary_img,
                                           rectangle=tuple(i[:4]),
                                           confidence=i[5])
                        text.set_state_colours_dic(self.text_states)
                        if i[4] in ans:
                            ans[i[4]].append(text)
                        else:
                            ans[i[4]] = [text]
                return ans
            case 3:
                raise NotImplementedError
            case 4:
                raise NotImplementedError
            case _:
                raise NotImplementedError
