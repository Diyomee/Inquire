from threading import Event
import paddleocr
import pathlib
import cv2
import numpy

# Path To THIS FILE
CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
DET_MODEL_DIR = CURRENT_PATH.parent.parent.parent.joinpath("rsc", "data", "ocr", "en_PP-OCRv3_det_infer").resolve()
if not DET_MODEL_DIR.is_dir():
    raise FileNotFoundError(str(DET_MODEL_DIR))
REC_MODEL_DIR = CURRENT_PATH.parent.parent.parent.joinpath("rsc", "data", "ocr", "en_PP-OCRv3_rec_infer").resolve()
if not DET_MODEL_DIR.is_dir():
    raise FileNotFoundError(str(DET_MODEL_DIR))
CLS_MODEL_DIR = CURRENT_PATH.parent.parent.parent.joinpath("rsc", "data", "ocr",
                                                           "ch_ppocr_mobile_v2.0_cls_infer").resolve()
if not CLS_MODEL_DIR.is_dir():
    raise FileNotFoundError(str(CLS_MODEL_DIR))


# TODO Metaklasse für Paddle und Tesseract für paperanalyse!
class PaddleOCR:
    """
    Class used for text detection and recognition. it uses a PP-OCRv3 from Paddle Paddle as a DB Model.
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self):
        self.fin_ocr = Event()
        self.threshold = 0.0
        self.draw_ocr = False
        self._ocr = paddleocr.PaddleOCR(det_algorithm="DB", lang="en", det=True, rec=True, cls=False,
                                        use_space_char=True,
                                        det_db_unclip_ratio=1.8, det_db_thresh=0.2, det_db_box_thresh=0.5,
                                        precision="int8",
                                        det_model_dir=str(DET_MODEL_DIR), rec_model_dir=str(REC_MODEL_DIR),
                                        structure_version="PP-StructureV2", drop_score=0.5,
                                        cls_model_dir=str(CLS_MODEL_DIR), show_log=False)

    def reset(self):
        self.fin_ocr.clear()

    def analyse(self, img_t) -> list[tuple[int, int, int, int, str, float]]:
        self.reset()
        results = self._ocr.ocr(img_t, cls=False)
        # results has at least length 1
        ans = []
        if results:
            # results first entry has at least length 1
            if results[0]:
                for result in results[0]:
                    if result[1][1] >= self.threshold:
                        ans.append((int(result[0][0][0]),
                                    int(result[0][0][1]),
                                    int(result[0][2][0] - result[0][0][0]),
                                    int(result[0][2][1] - result[0][0][1]),
                                    str(result[1][0]),
                                    float(result[1][1])))
        self.fin_ocr.set()
        return ans

    def ocr_image(self, gray: numpy.ndarray, bin_img: numpy.ndarray, xx: int, yy: int) -> numpy.ndarray:
        """
        Operates Paddle ocr on the given region of the image
        TODO
        :return: numpy array of found text (x,y,w,h,text,conf)
        :rtype: numpy.ndarray
        """
        bina = bin_img.copy()
        bina = cv2.bitwise_not(bina)
        founds_g = self.analyse(gray)
        founds_b = self.analyse(bina)
        if self.draw_ocr:
            for (x, y, w, h, val, con) in founds_g:
                gray = cv2.rectangle(gray, (x, y), (x + w, y + h), 125, 5)
                gray = cv2.putText(gray, val, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 192, 3)
            cv2.imshow("gray", gray)
            for (x, y, w, h, val, con) in founds_b:
                bina = cv2.rectangle(bina, (x, y), (x + w, y + h), 125, 5)
                bina = cv2.putText(bina, val, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 192, 3)
            cv2.imshow("bin", bina)
            cv2.waitKey()
            cv2.destroyAllWindows()
        if founds_b or founds_g:
            found_text = self.vote_on_found_text([founds_b, founds_g])
            found_text[:, 0] = found_text[:, 0] + xx
            found_text[:, 1] = found_text[:, 1] + yy
        else:
            found_text = None
        return found_text

    def vote_on_found_text(self, voters: list[list[tuple[int, int, int, int, str, float]]]) -> numpy.ndarray:
        r = self.convert_list_of_analyse_data_to_numpy_array(voters)
        r = self.non_max_suppression_for_text(r)
        return r

    @staticmethod
    def convert_list_of_analyse_data_to_numpy_array(to_convert: list[list[tuple[int, int, int, int, str, float]]]):
        """
        Converts a list of multiple analysis results into one numpy
        :param to_convert:
        :type to_convert:
        :return:
        :rtype:
        """
        temp = []
        for j in to_convert:
            temp.extend(j)
        ans = numpy.array(temp, dtype=object)
        ans[:, 2] = ans[:, 0] + ans[:, 2] - 1
        ans[:, 3] = ans[:, 1] + ans[:, 3] - 1
        return ans

    @staticmethod
    def non_max_suppression_for_text(boxes: numpy.ndarray, threshold: float = 0.3) -> numpy.ndarray | None:
        """
        Finds the rectangle with the largest overlapping area with hin all other overlapping rectangles
        Added functionality to not lose the text data and confidence
        :param boxes: list of rectangles with their top left and bottom right corner. NOT width or height!
        :type boxes: numpy.ndarray
        :param threshold: Overlapping area in %
        :type threshold: float
        :return: list of rectangles
        :rtype: numpy.ndarray
        :raises TypeError
        """
        # No Rectangles to Merge
        if len(boxes) == 0:
            return None
        if boxes.dtype.kind != "O":
            raise TypeError("The dtype of boxes needs to be O")
        picked = []
        # x
        left = boxes[:, 0]
        # y
        top = boxes[:, 1]
        # x2
        right = boxes[:, 2]
        # y2
        bottom = boxes[:, 3]
        # found string
        text = boxes[:, 4]
        # confidence
        confidence = boxes[:, 5]

        area = (right - left + 1) * (bottom - top + 1)
        indexes = numpy.argsort(area)
        while len(indexes) > 0:
            current_index = indexes[len(indexes) - 1]
            picked.append(current_index)
            suppressed = [len(indexes) - 1]
            for position in range(len(indexes) - 1, -1, -1):
                other_index = indexes[position]
                new_left = max(left[current_index], left[other_index])
                new_top = max(top[current_index], top[other_index])
                new_right = min(right[current_index], right[other_index])
                new_bottom = min(bottom[current_index], bottom[other_index])
                width = max(0, new_right - new_left + 1)
                height = max(0, new_bottom - new_top + 1)
                if area[other_index] != 0:
                    overlap_area = float(width * height) / area[other_index]
                    if overlap_area > threshold:
                        if confidence[current_index] < confidence[other_index]:
                            text[current_index] = text[other_index]
                            confidence[current_index] = confidence[other_index]
                        else:
                            text[other_index] = text[current_index]
                            confidence[other_index] = confidence[current_index]
                        suppressed.append(position)
            indexes = numpy.delete(indexes, suppressed)

        boxes[:, 2] = boxes[:, 2] + 1 - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + 1 - boxes[:, 1]
        return boxes[picked]


if __name__ == '__main__':
    pass
