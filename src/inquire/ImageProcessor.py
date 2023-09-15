from time import perf_counter_ns
import cv2
import numpy
import pathlib
from inquire.detection.SymbolRecogniser import SymbolRecognizer
from inquire.ocr.TextRecognizer import TextRecognizer
from inquire.detection.ROITopDown import ROITopDown
from inquire.detection.TemplateMatcher import TemplateMatcher
from inquire.exceptions.ImageNotFoundException import ImageNotFoundException
from inquire.loggers.ImageCache import ImageCache
from inquire.loggers.ImageHashing import ImageHashing
from inquire.helper.XMLParser import XMLParser

try:
    from robot.api.logger import info

    robot_logger = True
except ImportError:
    robot_logger = False
    import logging

    logging.basicConfig(level=logging.INFO)


class ImageProcessor:
    # if not robot_logger:
    #     logger = logging.getLogger(__name__)
    #     logger.setLevel(logging.INFO)
    def __init__(self):
        self.ocr = TextRecognizer(1)
        self.top_down = ROITopDown()
        self.symbol_recognizer = SymbolRecognizer()
        self.bottom_up = None
        self.print_img = False
        self.capture_region = {}

    def load(self, image_dir: str | pathlib.Path,
             xml_path: str | pathlib.Path | None = None,
             xml_path_2: str | pathlib.Path | None = None,
             colour_depths: int = 16):
        """
        Loads the Symbols of the symbols recognizer, and the states possible states for specific elements
        :param xml_path_2:
        :type xml_path_2:
        :param image_dir:
        :type image_dir:
        :param xml_path:
        :type xml_path:
        :param colour_depths:
        :type colour_depths:
        :return:
        :rtype:
        """
        dic = XMLParser.parse_state_information(xml_path_2)
        if "Text" in dic:
            self.ocr.text_states = dic["Text"]
        self.symbol_recognizer.symbol_states = dic
        self.symbol_recognizer.load_symbols(image_dir, xml_path, colour_depths)

    @staticmethod
    def find_capture_region(image: numpy.ndarray, template: numpy.ndarray) -> tuple[dict[str, int], float]:
        """
        Uses Template Matching to Find the Capture region and returns the region as dictionary and the confidence value
        :param image: BGR Screenshot of the whole screen
        :type image: numpy.ndarray
        :param template: BGR Screenshot of the GUI start Screen
        :type template: numpy.ndarray
        :return: capture region dictionary with key top, left, width, height. And confidence
        :rtype: tuple[dict[str,int],float]
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = TemplateMatcher.match_one(image, template)
        if result is None:
            raise ImageNotFoundException("Capture Region!")
        ans = {"top": result[1], "left": result[0], "width": result[2], "height": result[3]}
        return ans, result[4]

    @staticmethod
    def __save_rois(rois, img):
        img_t = img.copy()
        for key, value in rois.items():
            rect_region = value[0]
            # img_t = cv2.rectangle(img_t, (rect_region[0], rect_region[1]), (rect_region[0] + rect_region[2], rect_region[1] + rect_region[3]), (0,255,0), 2)
            for rect in value[1:]:
                img_t = cv2.rectangle(img_t, (rect[0] + rect_region[0], rect[1] + rect_region[1]),
                                      (rect_region[0] + rect[0] + rect[2], rect[1] + rect[3] + rect_region[1]),
                                      (0, 255, 0), 2)

        hash_t = ImageHashing.marr_hildreth_hashing(img_t)
        hash_t += ".png"
        CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
        dir = CURRENT_PATH.parent.parent.parent.joinpath("Daten", "roi", ).resolve().joinpath(hash_t)
        cv2.imwrite(str(dir), img_t)

    def analyse_image(self, image: numpy.ndarray, cache: ImageCache):
        regional_dictionary = {}
        fullscreen_dictionary = {"Text": {}, "Symbol": {}, "Textline": {}}
        img_mod_start = perf_counter_ns()
        self.top_down.set_image(image)
        img, gray, bin_img, diluted = self.top_down.get_images()
        if self.print_img:
            hash_t = ImageHashing.marr_hildreth_hashing(img)
            im1 = "colour" + hash_t + ".png"
            im2 = "gray" + hash_t + ".png"
            im3 = "binaer" + hash_t + ".png"
            im4 = "diluted" + hash_t + ".png"
            CURRENT_PATH = pathlib.Path(__file__).parent.resolve()

            dir = CURRENT_PATH.parent.parent.parent.joinpath("Daten", "roi", ).resolve().joinpath(im1)
            cv2.imwrite(str(dir), img)
            dir = CURRENT_PATH.parent.parent.parent.joinpath("Daten", "roi", ).resolve().joinpath(im2)
            cv2.imwrite(str(dir), gray)
            dir = CURRENT_PATH.parent.parent.parent.joinpath("Daten", "roi", ).resolve().joinpath(im3)
            cv2.imwrite(str(dir), bin_img)
            dir = CURRENT_PATH.parent.parent.parent.joinpath("Daten", "roi", ).resolve().joinpath(im4)
            cv2.imwrite(str(dir), diluted)
        if self.bottom_up is not None:
            self.bottom_up.set_all_4_images(img, gray, bin_img, diluted)
        regions = self.top_down.flood_filling_algorithm()
        rois = self.top_down.block_analysis(regions)
        img_mod_end = perf_counter_ns()
        img_mod_value = abs(img_mod_end - img_mod_start)
        img_text_start = perf_counter_ns()
        img_text_end = perf_counter_ns()
        img_text_value = 0
        img_symb_start = perf_counter_ns()
        img_symb_end = perf_counter_ns()
        img_symb_value = 0
        self.__save_rois(rois, img)

        # TODO add bottom up ROIs! and vote/maximum suppression
        for key, value in rois.items():
            rect_region = value[0]
            # todo fix cache ROI
            # generate reference to the images of the Region of Interest
            gray_roi = gray[rect_region[1]:rect_region[1] + rect_region[3],
                       rect_region[0]:rect_region[0] + rect_region[2]]
            binary_roi = bin_img[rect_region[1]:rect_region[1] + rect_region[3],
                         rect_region[0]:rect_region[0] + rect_region[2]]
            img_roi = img[rect_region[1]:rect_region[1] + rect_region[3],
                      rect_region[0]:rect_region[0] + rect_region[2]]
            # roi_hash = ImageHashing.marr_hildreth_hashing(img_roi.copy())
            # if cache.has_hash(roi_hash):
            #     str_t, d3, d4 = cache.get_element(roi_hash)
            #     regional_dictionary[key] = d3
            #     info(f'used roi cash {key}', html=True)
            #     texts = d3["Text"]
            #     symbols = d3["Symbol"]
            # else:
            # Analyse Text
            img_text_start = perf_counter_ns()
            texts = self.ocr.recognize_text(img_roi, gray_roi, binary_roi.copy(), rect_region[0], rect_region[1])
            img_text_end = perf_counter_ns()
            img_text_value += abs(img_text_end - img_text_start)
            # Analyse Symbols
            img_symb_start = perf_counter_ns()
            symbols = self.symbol_recognizer.identify_symbols(img_roi, binary_roi, value, rect_region[0],
                                                              rect_region[1])
            img_symb_end = perf_counter_ns()
            img_symb_value += abs(img_symb_end - img_symb_start)
            d3 = {"Text": texts, "Symbol": symbols}
            regional_dictionary[key] = d3
            # cache.add_element(roi_hash, "", d3, {}, img_roi)

            # TODO generate a text line dictionary. in which the text of each line stand. idea start with the
            #  smallest one
            # Adds the found values of the region to the global dictionary
            for text_key, text_value in texts.items():
                # TODO build a x-y map of rectangles, so that each rectangle knows its nearest neighbor.
                if text_key in fullscreen_dictionary["Text"]:
                    fullscreen_dictionary["Text"][text_key].extend(text_value)
                else:
                    fullscreen_dictionary["Text"][text_key] = text_value
            for symbol_key, symbol_value in symbols.items():
                # TODO build a x-y map of rectangles, so that each rectangle knows its nearest neighbor.
                if symbol_key in fullscreen_dictionary["Symbol"]:
                    fullscreen_dictionary["Symbol"][symbol_key].extend(symbol_value)
                else:
                    fullscreen_dictionary["Symbol"][symbol_key] = symbol_value

        fullscreen_dictionary["Textline"] = self.ocr.align_text_to_lines(fullscreen_dictionary["Text"])
        if robot_logger:
            info(
                f'Performance: Image editing: {img_mod_value / 1000000} ms; Textrecognition: {img_text_value / 1000000} ms; Symbolrecognition: {img_symb_value / 1000000}ms',
                html=True)
        else:
            logging.info(
                f'Performance: Image editing: {img_mod_value / 1000000} ms; Textrecognition: {img_text_value / 1000000} ms; Symbolrecognition: {img_symb_value / 1000000}ms')
        return fullscreen_dictionary, regional_dictionary


if __name__ == '__main__':
    pass
