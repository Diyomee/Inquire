import pathlib
import time
import cv2
import Levenshtein
from time import perf_counter_ns
from inquire.ImageProcessor import ImageProcessor
from inquire.loggers.ImageLogger import ImageLogger
from inquire.loggers.VideoLogger import VideoLogger
from inquire.loggers.ImageCache import ImageCache
from inquire.detection.ButtonState import ButtonState
from inquire.exceptions.ImageNotFoundException import ImageNotFoundException
import mouse

try:
    from robot.api.logger import info

    robot_logger = True
except ImportError:
    import logging

    robot_logger = False
    logging.basicConfig(level=logging.WARNING)


class Inquire:
    """
    Inquire main api, and wrapper of all functionalities
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = True
    __version__ = '0.1'

    def __init__(self):
        self.start = perf_counter_ns()
        self.image_processor = ImageProcessor()
        self.image_logger = ImageLogger()
        self.video_logger = VideoLogger()
        self.cache = ImageCache()
        self.root_directory = None
        self.ocr_model_path = None
        self.symbol_xml_path = None
        self.state_xml_path = None
        self.images_path = None
        self.results_path = None
        self.last_dom = {}
        self.last_regional_dom = {}
        self.last_filename = ""
        self.region = {}

    def start_recording_video(self):
        self.video_logger.start_process_for_video_recoding()

    def stop_recording_video(self):
        self.video_logger.stop_recording()
        self.cache.save_cache()

    def set_file_name(self, suite_name: str, test_name: str):
        """
        Sets the names for the saved png documentation
        :param suite_name:
        :type suite_name:
        :param test_name:
        :type test_name:
        :return:
        :rtype:
        """
        self.image_logger.set_filename_to(suite_name, test_name)
        self.video_logger.set_filename_to(suite_name, test_name)

    def load_necessary_data(self, root_directory: str,
                            ocr_model_path: str = "rsc//data//ocr",
                            symbol_xml_path: str | None = "rsc//Symbols.xml",
                            state_xml_path: str | None = "rsc//States.xml",
                            images_path: str = "rsc//images",
                            result_path: str = "results",
                            init_screen_path: str = "rsc//init.png",
                            colour_depths: int = 16):
        """
        Loads all Data necessary for this module. This includes the paths to specific directories, and the cache.
        TODO
        :param state_xml_path:
        :type state_xml_path:
        :param result_path:
        :type result_path:
        :param init_screen_path:
        :type init_screen_path:
        :param colour_depths:
        :type colour_depths:
        :param root_directory:
        :type root_directory:
        :param ocr_model_path:
        :type ocr_model_path:
        :param symbol_xml_path:
        :type symbol_xml_path:
        :param images_path:
        :type images_path:
        :return:
        :rtype:
        """
        self.root_directory = pathlib.Path(root_directory)
        if not self.root_directory.is_dir():
            raise FileExistsError(f'Root directory does not exist {self.root_directory}')
        self.ocr_model_path = self.root_directory.joinpath(ocr_model_path).resolve()
        if not self.ocr_model_path.is_dir():
            raise FileExistsError(f'OCR model directory does not exist {self.ocr_model_path}')
        if symbol_xml_path is not None:
            self.symbol_xml_path = self.root_directory.joinpath(symbol_xml_path).resolve()
            if not self.symbol_xml_path.is_file():
                raise FileExistsError(f'Specification file for Symbols,xml does not exist {self.symbol_xml_path}')
        if state_xml_path is not None:
            self.state_xml_path = self.root_directory.joinpath(state_xml_path).resolve()
            if not self.state_xml_path.is_file():
                raise FileExistsError(f'Specification file for Symbols,xml does not exist {self.state_xml_path}')
        init_screen_path = self.root_directory.joinpath(init_screen_path).resolve()
        if not init_screen_path.is_file():
            raise FileExistsError(f'Initialization screenshot foes not exist: {init_screen_path}')
        self.images_path = self.root_directory.joinpath(images_path).resolve()
        if not self.images_path.is_dir():
            raise FileExistsError(f'Image Directory does not exist {self.images_path}')
        self.results_path = self.root_directory.joinpath(result_path).resolve()
        if not self.results_path.is_dir():
            self.results_path.mkdir(parents=True)
        cache_path = self.root_directory.joinpath("rsc", "data", "cache").resolve()
        if not self.results_path.is_dir():
            cache_path.mkdir(parents=True, exist_ok=True)

        self.image_processor.load(self.images_path, self.symbol_xml_path, self.state_xml_path, colour_depths)

        self.image_logger.set_output_directory(str(self.results_path))
        self.video_logger.set_output_directory(str(self.results_path))

        image = self.image_logger.screenshot_all_monitors()
        template = cv2.imread(str(init_screen_path))
        region, conf = self.image_processor.find_capture_region(image, template)
        self.region = region
        self.image_logger.set_capture_region(region)
        self.video_logger.set_capture_region(region)
        self.cache.set_storage_path(cache_path)
        self.cache.load_cache()

    def does_symbol_occure_in_x_seconds(self, symbol: str, timeframe: int = 10, count: int = 1, dom=False) -> bool:
        """
        Checks whether a symbol appears in x seconds. In the case that the symbol can occur multiple times on the display,
        you have to specify which you want to verify. The first, second,third,.... When you don't want to take a new
        screenshot, and work with the information of the last one, you have to set dom to True.
        :param symbol: Name of the symbol you want to find. It has to be the same as the file name!
        :type symbol: str
        :param timeframe: Timeout for looking for the image
        :type timeframe: int
        :param count: xth Symbol you want to verify
        :type count: int
        :param dom: Disables the taking of a new screenshot, and uses the previous dom!
        :type dom: bool
        :return: True if visible, else an exception is raised!
        :rtype: bool
        :raises ImageNotFoundException
        """
        start = time.time()
        now = time.time()
        while (now - start) < timeframe:
            try:
                self.is_symbol_visible(symbol, count, dom)
            except:
                pass
            else:
                return True
            now = time.time()
        raise ImageNotFoundException(f'{count + 1}st symbol: {symbol} did not appeared in {timeframe} seconds')

    def does_symbol_disappear_in_x_seconds(self, symbol: str, timeframe: int = 10, count: int = 1, dom=False) -> bool:
        """
        Checks whether a symbol disappears in x seconds. In the case that the symbol can occur multiple times on the display,
        you have to specify which you want to verify. The first, second,third,.... When you don't want to take a new
        screenshot, and work with the information of the last one, you have to set dom to True.
        :param symbol: Name of the symbol you want to find. It has to be the same as the file name!
        :type symbol: str
        :param timeframe: Timeout for looking for the image
        :type timeframe: int
        :param count: xth Symbol you want to verify
        :type count: int
        :param dom: Disables the taking of a new screenshot, and uses the previous dom!
        :type dom: bool
        :return: True if visible, else an exception is raised!
        :rtype: bool
        :raises ImageNotFoundException
        """
        start = time.time()
        now = time.time()
        while (now - start) < timeframe:
            try:
                self.is_symbol_visible(symbol, count, dom)
            except:
                return True
            now = time.time()
        raise ImageNotFoundException(f'{count + 1}st symbol: {symbol} did not disappear in {timeframe} seconds')

    def is_symbol_visible(self, symbol: str, count: int = 1, dom=False):
        """
        Checks whether a symbol a visible or not. In the case that the symbol can occur multiple times on the display,
        you have to specify which you want to verify. The first, second,third,.... When you don't want to take a new
        screenshot, and work with the information of the last one, you have to set dom to True.
        :param symbol: Name of the symbol you want to find. It has to be the same as the file name!
        :type symbol: str
        :param count: xth Symbol you want to verify
        :type count: int
        :param dom: Disables the taking of a new screenshot, and uses the previous dom!
        :type dom: bool
        :return: True if visible, else an exception is raised!
        :rtype: bool
        :raises ImageNotFoundException
        """
        count -= 1
        if not dom:
            hash_t, dir_t, rel, screenshot = self.image_logger.make_a_screenshot()
            name = pathlib.Path(dir_t).joinpath(rel).resolve()
            if self.cache.has_hash(hash_t):
                if robot_logger:
                    info("cache used", html=True)
                else:
                    logging.info("cache used")
                name, d1, d2 = self.cache.get_element(hash_t)
            else:
                d1, d2 = self.image_processor.analyse_image(screenshot, self.cache)
                self.cache.add_element(hash_t, name, d1, d2)
            self.last_dom = d1
            self.last_regional_dom = d2
            self.last_filename = rel
            if robot_logger:
                info(f'found elements: {d1}')
            else:
                logging.info(f'found elements: {d1}')
        if symbol in self.last_dom["Symbol"]:
            symbols = self.last_dom["Symbol"][symbol]
            if count < len(symbols):
                if robot_logger:
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                return True
            else:
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                raise ImageNotFoundException(f'{count + 1}st symbol: {symbol}')
        else:
            if robot_logger:
                info(self.last_dom, html=True)
                info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
            raise ImageNotFoundException(f'{count + 1}st symbol: {symbol}')

    def is_similiar_text_displayed(self, text: str, count: int = 1, error_margin_abs=-1,
                                   error_margin_percent: float = 0.33, dom=False):
        """

        :param text:
        :type text:
        :param count:
        :type count:
        :param error_margin_abs: Number of changes
        :type error_margin_abs:
        :param error_margin_percent: percent of text that are allowed to be changed. error_margin_abs needs to be negative
        :type error_margin_percent: float
        :param dom:
        :type dom:
        :return:
        :rtype:
        """
        if error_margin_abs <= -1:
            error_margin_abs = round(len(text) * error_margin_percent)
        count -= 1
        if not dom:
            hash_t, dir_t, rel, screenshot = self.image_logger.make_a_screenshot()
            name = pathlib.Path(dir_t).joinpath(rel).resolve()
            if self.cache.has_hash(hash_t):
                if robot_logger:
                    info("cache used", html=True)
                else:
                    logging.info("cache used")
                name, d1, d2 = self.cache.get_element(hash_t)
            else:
                d1, d2 = self.image_processor.analyse_image(screenshot, self.cache)
                self.cache.add_element(hash_t, name, d1, d2)
            self.last_dom = d1
            self.last_regional_dom = d2
            self.last_filename = rel
        min_v = None
        min_k = None
        min_t = 1000
        for key, value in self.last_dom["Text"].items():
            d = Levenshtein.distance(text, key)
            if d < min_t:
                min_v = value
                min_k = key
                min_t = d
        if min_v is None or min_k is None:
            if robot_logger:
                info(self.last_dom, html=True)
                info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
            raise ImageNotFoundException(f'{count + 1}st text: {text}')
        if min_t <= error_margin_abs:
            if count < len(min_v):
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                return min_k, min_v[count]
            else:
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                raise ImageNotFoundException(f'{count + 1}st text: {text}')
        else:
            if robot_logger:
                info(self.last_dom, html=True)
                info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
            raise ImageNotFoundException(f'{count + 1}st text: {text}')

    def is_text_displayed(self, text: str, count: int = 1, dom=False):
        """
        Checks whether a Text a visible or not. In the case that the text can occur multiple times on the display,
        you have to specify which you want to verify. The first, second,third,.... When you don't want to take a new
        screenshot, and work with the information of the last one, you have to set dom to True.
        TODO check lines
        :param text: Text you want to find. It has to be the same as the file name!
        :type text: str
        :param count: xth Symbol you want to verify
        :type count: int
        :param dom: Disables the taking of a new screenshot, and uses the previous dom!
        :type dom: bool
        :return: True if visible, else an exception is raised!
        :rtype: bool
        :raises ImageNotFoundException
        """
        count -= 1
        if not dom:
            hash_t, dir_t, rel, screenshot = self.image_logger.make_a_screenshot()
            name = pathlib.Path(dir_t).joinpath(rel).resolve()
            if self.cache.has_hash(hash_t):
                if robot_logger:
                    info("cache used", html=True)
                else:
                    logging.info("cache used")
                name, d1, d2 = self.cache.get_element(hash_t)
            else:
                d1, d2 = self.image_processor.analyse_image(screenshot, self.cache)
                self.cache.add_element(hash_t, name, d1, d2)
            self.last_dom = d1
            self.last_regional_dom = d2
            self.last_filename = rel
        if text in self.last_dom["Text"]:
            texts = self.last_dom["Text"][text]
            if count < len(texts):
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                return True
            else:
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                raise ImageNotFoundException(f'{count + 1}st text: {text}')
        else:
            if robot_logger:
                info(self.last_dom, html=True)
                info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
            raise ImageNotFoundException(f'{count + 1}st text: {text}')

    def is_similiar_textline_displayed(self, textline: str, count: int = 1, error_margin_abs=-1,
                                       error_margin_percent: float = 0.33, dom=False):
        """

        :param textline:
        :type textline:
        :param count:
        :type count:
        :param error_margin_abs: Number of changes
        :type error_margin_abs:
        :param error_margin_percent: percent of text that are allowed to be changed. error_margin_abs needs to be negative
        :type error_margin_percent: float
        :param dom:
        :type dom:
        :return:
        :rtype:
        """
        if error_margin_abs <= -1:
            error_margin_abs = round(len(textline) * error_margin_percent)
        count -= 1
        if not dom:
            hash_t, dir_t, rel, screenshot = self.image_logger.make_a_screenshot()
            name = pathlib.Path(dir_t).joinpath(rel).resolve()
            if self.cache.has_hash(hash_t):
                if robot_logger:
                    info("cache used", html=True)
                else:
                    logging.info("cache used")
                name, d1, d2 = self.cache.get_element(hash_t)
            else:
                d1, d2 = self.image_processor.analyse_image(screenshot, self.cache)
                self.cache.add_element(hash_t, name, d1, d2)
            self.last_dom = d1
            self.last_regional_dom = d2
            self.last_filename = rel
        min_v = None
        min_k = None
        min_t = 1000
        for key, value in self.last_dom["Textline"].items():
            d = Levenshtein.distance(textline, key)
            if d < min_t:
                min_v = value
                min_k = key
                min_t = d
        if min_v is None or min_k is None:
            if robot_logger:
                info(self.last_dom, html=True)
                info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
            raise ImageNotFoundException(f'{count + 1}st text: {textline}')
        if min_t <= error_margin_abs:
            if count < len(min_v):
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                return min_k, min_v[count]
            else:
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                raise ImageNotFoundException(f'{count + 1}st text: {textline}')
        else:
            if robot_logger:
                info(self.last_dom, html=True)
                info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
            raise ImageNotFoundException(f'{count + 1}st text: {textline}')

    def is_textline_displayed(self, textline: str, count: int = 1, dom=False):
        """
        Checks whether a Text a visible or not. In the case that the text can occur multiple times on the display,
        you have to specify which you want to verify. The first, second,third,.... When you don't want to take a new
        screenshot, and work with the information of the last one, you have to set dom to True.
        TODO check lines
        :param textline: Text you want to find. It has to be the same as the file name!
        :type textline: str
        :param count: xth Symbol you want to verify
        :type count: int
        :param dom: Disables the taking of a new screenshot, and uses the previous dom!
        :type dom: bool
        :return: True if visible, else an exception is raised!
        :rtype: bool
        :raises ImageNotFoundException
        """
        count -= 1
        if not dom:
            hash_t, dir_t, rel, screenshot = self.image_logger.make_a_screenshot()
            name = pathlib.Path(dir_t).joinpath(rel).resolve()
            if self.cache.has_hash(hash_t):
                if robot_logger:
                    info("cache used", html=True)
                else:
                    logging.info("cache used")
                name, d1, d2 = self.cache.get_element(hash_t)
            else:
                d1, d2 = self.image_processor.analyse_image(screenshot, self.cache)
                self.cache.add_element(hash_t, name, d1, d2)
            self.last_dom = d1
            self.last_regional_dom = d2
            self.last_filename = rel
        if textline in self.last_dom["Textline"]:
            textliness = self.last_dom["Textline"][textline]
            if count < len(textlines):
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                return True
            else:
                if robot_logger:
                    info(self.last_dom, html=True)
                    info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
                raise ImageNotFoundException(f'{count + 1}st text: {textline}')
        else:
            if robot_logger:
                info(self.last_dom, html=True)
                info(f'<a href="{self.last_filename}"><img src="{self.last_filename}"></a>', html=True)
            raise ImageNotFoundException(f'{count + 1}st text: {textline}')

    def click_on_symbol(self, symbol: str, count: int = 1, dom=False):
        try:
            self.is_symbol_visible(symbol, count, dom)
        except Exception:
            raise
        else:
            count -= 1
            coordinates = self.last_dom["Symbol"][symbol][count].get_center_point()
            if self.region:
                img_text_start = perf_counter_ns()
                mouse.move(coordinates[0] + self.region["left"], coordinates[1] + self.region["top"], True, 0.2)
                img_text_end = perf_counter_ns()
                img_text_value = abs(img_text_end - img_text_start) / 1000000
                if robot_logger:
                    info(f'mouseanimation: {img_text_value}')
                else:
                    logging.info(f'mouseanimation: {img_text_value}')
                mouse.press(mouse.LEFT)
                time.sleep(0.05)
                mouse.release(mouse.LEFT)
                time.sleep(0.08)
            else:
                raise ValueError("Capture Regions has to be set!!!")
        finally:
            time.sleep(0.1)

    def click_on_text(self, text: str, count: int = 1, dom=False, x_offset: int = 0, y_offset: int = 0):
        try:
            self.is_text_displayed(text, count, dom)
        except Exception:
            raise
        else:
            count -= 1
            coordinates = self.last_dom["Text"][text][count].get_center_point()
            if self.region:
                mouse.move(coordinates[0] + self.region["left"] + x_offset,
                           coordinates[1] + self.region["top"] + y_offset,
                           True, 0.2)
                mouse.press(mouse.LEFT)
                time.sleep(0.05)
                mouse.release(mouse.LEFT)
                time.sleep(0.08)
            else:
                raise ValueError("Capture Regions has to be set!!!")
        finally:
            time.sleep(0.1)

    def click_on_similliar_text(self, text: str, count: int = 1,
                                error_margin_abs=-1, error_margin_percent: float = 0.33, dom=False,
                                x_offset: int = 0, y_offset: int = 0):
        try:
            found_text, found_element = self.is_similiar_text_displayed(text, count, error_margin_abs,
                                                                        error_margin_percent, dom)
        except Exception:
            raise
        else:
            count -= 1
            coordinates = found_element.get_center_point()
            if self.region:
                mouse.move(coordinates[0] + self.region["left"] + x_offset,
                           coordinates[1] + self.region["top"] + y_offset,
                           True, 0.2)
                mouse.press(mouse.LEFT)
                time.sleep(0.05)
                mouse.release(mouse.LEFT)
                time.sleep(0.08)
            else:
                raise ValueError("Capture Regions has to be set!!!")

    def scroll_from_symbol_down_by_y(self, symbol: str, y_axis: int, count: int = 1, dom=False):
        try:
            self.is_symbol_visible(symbol, count, dom)
        except Exception:
            raise
        else:
            count -= 1
            coordinates = self.last_dom["Symbol"][symbol][count]
            t = abs(y_axis) / 89 * 0.4
            if self.region:
                mouse.move(coordinates[0] + self.region["left"], coordinates[1] + self.region["top"], True, 0.2)
                mouse.press(mouse.LEFT)
                mouse.move(coordinates[0] + self.region["left"], coordinates[1] + self.region["top"] - y_axis, True, t)
                mouse.release()
            else:
                raise ValueError("Capture Regions has to be set!!!")

    def scroll_from_text_down_by_y(self, text: str, y_axis: int, count: int = 1, dom=False):
        try:
            self.is_text_displayed(text, count, dom)
        except Exception:
            raise
        else:
            count -= 1
            coordinates = self.last_dom["Text"][text][count].get_center_point()
            t = abs(y_axis) / 89 * 0.4
            if self.region:
                mouse.move(coordinates[0] + self.region["left"], coordinates[1] + self.region["top"], True, 0.2)
                mouse.press(mouse.LEFT)
                mouse.move(coordinates[0] + self.region["left"], coordinates[1] + self.region["top"] - y_axis, True, t)
                mouse.release()
            else:
                raise ValueError("Capture Regions has to be set!!!")

    def click_on_element_next_to_the_displayed_text(self, text: str, orientation: str = "LEFT", count: int = 1,
                                                    dom=False):
        # TODO
        pass

    def click_on_element_next_to_the_symbol(self, symbol: str, orientation: str = "LEFT", count: int = 1,
                                            dom=False):
        # TODO
        pass

    def symbol_has_state(self, symbol: str, state: str, count=1, dom=False):
        try:
            self.is_symbol_visible(symbol, count, dom)
        except Exception:
            raise
        else:
            count -= 1
            symbol_state = self.last_dom["Symbol"][symbol][count].get_button_state()
            state = ButtonState[state]
            if symbol_state is state:
                return True
            else:
                raise AssertionError(f'given {state} is not found {symbol_state}')

    def text_has_state(self, text: str, state: str, count=1, dom=False):
        try:
            self.is_text_displayed(text, count, dom)
        except Exception:
            raise
        else:
            count -= 1
            text_state = self.last_dom["Text"][text][count].get_button_state()
            state = ButtonState[state]
            if text_state is state:
                return True
            else:
                raise AssertionError(f'given {state} is not found {text_state}')

    def similiar_text_has_state(self, text: str, state: str, error_margin_abs=-1, error_margin_percent: float = 0.33,
                                count=1, dom=False):
        try:
            found_text, found_element = self.is_similiar_text_displayed(text, count, error_margin_abs,
                                                                        error_margin_percent, dom)
        except Exception:
            raise
        else:
            count -= 1
            text_state = found_element.get_button_state()
            state = ButtonState[state]
            if text_state is state:
                return True
            else:
                raise AssertionError(f'given {state} is not found {text_state}')


if __name__ == '__main__':
    pass
