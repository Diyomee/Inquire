import numpy
import cv2
import pathlib

from inquire.helper.XMLParser import XMLParser
from inquire.detection.Symbol import Symbol
from inquire.detection.TemplateMatcher import TemplateMatcher
from inquire.detection.ButtonState import ButtonState
from inquire.helper.Comparator import Comparator

try:
    from robot.api.logger import warn

    robot_logger = True
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    robot_logger = False


class SymbolRecognizer:
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self):
        self._symbols: dict[str, Symbol] = {}
        self.symbol_states: dict[str, dict[ButtonState, list[list[tuple[int, int, int]]]]] = {}
        self.threshold = 0.85
        # close cross conf 87

    def identify_symbols(self, img: numpy.ndarray,
                         bin_img: numpy.ndarray,
                         rectangles: list[tuple[int, int, int, int]],
                         xx: int, yy: int) -> dict[str, list[Symbol]]:
        """
        Tries to find for each region specified in rectangles the symbol through which it is represented,
        with a confidence higher than 92%
        :param img: The BGR-Image off the Screen containing multiple Elements that have to be recognized
        :type img: numpy.ndarray
        :param bin_img: The Binary-Image off the Screen containing multiple Elements that have to be recognized
        :type bin_img: numpy.ndarray
        :param rectangles: List of Region of Interest that have to be identified
        :type rectangles: list[tuple[int,int,int,int]]
        :param xx: x Coordinate of the given image, in the application window
        :type xx: int
        :param yy: y Coordinate of the given image, in the application window
        :type yy: int
        :return: Dictionary of found symbols. Since multiple symbols of the same kind could be detected,
        they are stored in a list
        :rtype: dict[str, list[Symbol]]
        """
        ans = {}
        ymax = img.shape[0]
        xmax = img.shape[1]
        for rect in rectangles[1:]:
            (x, y, w, h) = rect
            sym_img = img[y:y + h, x:x + w]
            sym_bin_img = bin_img[max(0, y - 5):min(ymax, y + h + 5), max(0, x - 5):min(xmax, x + w + 5)]
            x += xx
            y += yy
            symbol = self.identify_symbol(sym_img, sym_bin_img, (x, y, w, h))
            if symbol is not None:
                if symbol.symbol in ans:
                    ans[symbol.symbol].append(symbol)
                else:
                    ans[symbol.symbol] = [symbol]
                if len(ans[symbol.symbol]) > 1:
                    ans[symbol.symbol] = sorted(ans[symbol.symbol],
                                                key=Comparator.position_key_of_recognized_elements_position,
                                                reverse=False)
        return ans

    def identify_symbol(self, img: numpy.ndarray, bin_img: numpy.ndarray, rectangle: tuple):
        """
        Tries to recognize the symbol in the given image
        :param bin_img: Binary Image, for the recognition process
        :type bin_img: numpy.ndarray
        :param img: BGR image, to store the found symbol for the robot framework
        :type img: numpy.ndarray
        :param rectangle: Bounding rectangle (x,y,width,height) of the given image in the application window
        :type rectangle: tuple
        :return: Returns the found Symbol or None
        :rtype: Symbol | None
        """
        conf = 0.0
        name = ""
        path = ""
        position = [0, 0, 0, 0]
        for symbol in self._symbols.values():
            match = TemplateMatcher.match_one(image=bin_img, template=symbol.binary_image, mask=symbol.mask)
            if match is None:
                continue
            else:
                if conf < match[4]:
                    conf = match[4]
                    name = symbol.symbol
                    path = symbol.symbol_path
                    position[0] = match[0] + rectangle[0]
                    position[1] = match[1] + rectangle[1]
                    position[2] = match[2]
                    position[3] = match[3]
        if conf < self.threshold:
            return None
        else:
            s = Symbol(symbol=name,
                       symbol_path=path,
                       image=img,
                       binary_image=bin_img,
                       rectangle=tuple(position),
                       confidence=conf)
            s.set_state_colours_dic(self._symbols[name].state_colours)
            return s

    def __set_symbol(self, img_path: pathlib.Path, colour_depths: int) -> None:
        """
        Adds the new symbol locate at the path to the local dictionary for storage
        It should only be used in combination with load_symbols
        :param img_path: path to the symbol image
        :type img_path: pathlib.Path
        :param colour_depths: colour depths for the rendering of the binary image (4,16)
        :type colour_depths: int
        :return: None
        :rtype: None
        """
        img = cv2.imread(str(img_path))
        match colour_depths:
            case 4:
                bin_img = TemplateMatcher.generate_binary_image_4bit(img)
            case _:
                bin_img = TemplateMatcher.generate_binary_image(img)
        mask = self.generate_mask(bin_img)
        self._symbols[img_path.stem] = Symbol(symbol=img_path.stem,
                                              symbol_path=str(img_path),
                                              image=img,
                                              binary_image=bin_img,
                                              mask=mask)

    @staticmethod
    def generate_mask(img: numpy.ndarray) -> numpy.ndarray:
        """
        Generates a binary mask from a given gray scaled/binary image
        :param img: gray scaled | binary image
        :type img: numpy.ndarray
        :return: binary mask
        :rtype: numpy.ndarray
        """
        canny_out = img.copy()
        mask_contour, _ = cv2.findContours(image=canny_out, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        mask_contour = numpy.concatenate(mask_contour)
        mask_t = numpy.zeros((canny_out.shape[0], canny_out.shape[1], 1), dtype=numpy.uint8)
        hull_list = []
        hull = cv2.convexHull(mask_contour)
        hull_list.append(hull)
        mask_t = cv2.drawContours(mask_t, hull_list, -1, 255, cv2.FILLED)
        return mask_t

    def load_symbols(self, image_dir: str | pathlib.Path,
                     xml_path: str | pathlib.Path | None = None,
                     colour_depths: int = 16) -> None:
        """
        Loads all .png files from image_dir and its subdirectories as symbols, and generates their binary image.
        The xml paths specifies which symbols shall or shall not be loaded. It can work as an ignore list,
        or a selection list.
        :param image_dir: Path to the directory where the symbols are located
        :type image_dir: str | pathlib.Path
        :param xml_path: Path to the xml file which specifies which images shall be used as symbols
        :type xml_path: str | pathlib.Path | None
        :param colour_depths: rendering information for the binary image of the symbol. 4 | 16 for the colour depths
        :type colour_depths: int
        :return: None
        :rtype: None
        """
        match image_dir:
            case w if isinstance(w, str):
                img_path = pathlib.Path(image_dir)
                if not img_path.is_dir():
                    raise FileExistsError(f'Image Directory {image_dir} does not exist')
            case w if isinstance(w, pathlib.PurePath):
                img_path = image_dir
                if not image_dir.is_dir():
                    raise FileExistsError(f'Image Directory {image_dir} does not exist')
            case _:
                raise TypeError('image_dir needs to be a string or a pathlib path!')
        match xml_path:
            case w if w is None:
                pos_paths = set()
                neg_paths = set()
            case w if isinstance(w, str):
                xml_path = pathlib.Path(xml_path)
                # relative paths are returned
                pos_paths, neg_paths = XMLParser.parse_symbol_selection_xml(xml_path)
            case w if isinstance(w, pathlib.PurePath):
                pos_paths, neg_paths = XMLParser.parse_symbol_selection_xml(xml_path)
            case _:
                raise TypeError('xml_path has to be None, str, or a pathlib path')
        if pos_paths:
            for i in pos_paths:
                i = xml_path.parent.joinpath(i)
                if i.is_file():
                    self.__set_symbol(i, colour_depths)
                else:
                    if robot_logger:
                        warn(f'could not load symbol {i}')
                    else:
                        logging.warning(f'could not load symbol {i}')
        else:
            paths = img_path.glob('**/*.png')
            if neg_paths:
                neg_pathlib_paths = set()
                for np in neg_paths:
                    neg_pathlib_paths.add(pathlib.Path(np))
                for i in paths:
                    if i.relative_to(img_path.parent) in neg_pathlib_paths:
                        continue
                    if i.is_file():
                        self.__set_symbol(i, colour_depths)
                    else:
                        if robot_logger:
                            warn(f'could not load symbol {i}')
                        else:
                            logging.warning(f'could not load symbol {i}')
            else:
                for i in paths:
                    if i.is_file():
                        self.__set_symbol(i, colour_depths)
                    else:
                        if robot_logger:
                            warn(f'could not load symbol {i}')
                        else:
                            logging.warning(f'could not load symbol {i}')
        for key, value in self.symbol_states.items():
            if key in self._symbols:
                self._symbols[key].set_state_colours_dic(value)

    @staticmethod
    def img_is_an_image(img: numpy.ndarray):
        """
        Returns True if the given image resulted from an image (photographic)
        :param img: the given image
        :type img: numpy.ndarray
        :return: True is it's and image
        :rtype: bool
        :raises TypeError
        """
        if img.shape[2] == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:
            gray_img = img
        else:
            raise TypeError("Img is neither BGR nor Grayscale")
        gray_img = cv2.Laplacian(gray_img, cv2.CV_64F)
        gray_img[abs(gray_img) > 0.0] = 255.0
        gray_img = numpy.uint8(numpy.absolute(gray_img))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray_img = cv2.bitwise_not(gray_img)
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        wp = cv2.countNonZero(gray_img)
        t = gray_img.size
        return wp / t > 0.9


if __name__ == '__main__':
    pass