import os
import json
import cv2
from os import path, getcwd
import pathlib
import numpy
from mss import mss

from inquire.loggers.Logger import Logger
from inquire.loggers.ImageHashing import ImageHashing

try:
    from robot.api.logger import info

    robot_logger = True
except ImportError:
    robot_logger = False

ROOT = getcwd()
FORMAT = cv2.VideoWriter_fourcc(*'VP90')


class ImageLogger(Logger):
    """
    The ImageLogger class is responsible for taking, storing, and hashing of screenshots, which can be accessed over
    their hash.
    When a screenshot is taken, it returns all necessary information about the image.
    """

    def __init__(self):
        """
        todo pathlib
        Initializes the ImageLogger Class.
        """
        super().__init__()
        self.__screenshot_name = self._output_dir.joinpath("filler.png")
        self.__screenshot_dir = self._output_dir.joinpath("filler")
        self.__dictionary_path = self._output_dir.joinpath("image_map.json")
        self.__screenshot_dictionary = {}
        self.__screenshot_path_dictionary = {}
        self.__load_screenshots_dictionary_from_file()
        self.__counter = 0
        self.__sct = mss()
        self.__suite_name = ""
        self.__test_name = ""
        self.__filetype = ".png"

    def write_screenshots_dictionary_to_files(self) -> None:
        """
        TODO maybe with packle instead of json
        Saves the image cache to the file image_map.json in the given output directory of this ImageLoggerClass
        :return: None
        :rtype: None
        """
        with open(str(self.__dictionary_path), 'w') as f:
            json.dump(self.__screenshot_path_dictionary, f)
        return None

    def __load_screenshots_dictionary_from_file(self) -> bool:
        """
        TODO maybe with packle instead of json, pathlib
        Tries load the cache from the file image_map.json in the given output directory of this ImageLoggerClass,
        if it exists
        :return: True|False
        :rtype: bool
        :raises FileNotFoundError
        """
        if self.__dictionary_path.exists():
            with open(self.__dictionary_path) as f:
                self.__screenshot_path_dictionary = json.load(f)
        for hash_t, path_t in self.__screenshot_path_dictionary.items():
            p = self._output_dir.joinpath(path_t).resolve()
            if p.exists():
                img = cv2.imread(str(p))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.__screenshot_dictionary[hash_t] = (path_t, img)
            else:
                raise FileNotFoundError(str(p))
        return True

    def set_filename_to(self, suite_name: str, test_name: str = "", filetype: str = "png"):
        """
        Sets the filename and filetype to the given values for the next video recording
        TODO pathlib
        :param test_name: Name of the test case
        :type test_name: str
        :param suite_name: Name of the test suite
        :type suite_name: str
        :param filetype: type of the file. default webm
        :type filetype: str
        :return:None
        :rtype:
        """
        self.__counter = 0
        self.__suite_name = suite_name
        self.__test_name = test_name
        self.__filetype = filetype
        self.__screenshot_dir = self._output_dir.joinpath(f'{self.__suite_name}').resolve()
        self.__screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.__screenshot_name = self.__screenshot_dir.joinpath(
            f'{self.__test_name}_{self.__counter}.{self.__filetype}').resolve()
        if robot_logger:
            info(f'File path is {self.__screenshot_name}')

    def increment_file_name(self):
        """
        Increments the counter of the screenshot png files by one
        :return: None
        :rtype:
        """
        self.__counter += 1
        self.__screenshot_name = self.__screenshot_dir.joinpath(
            f'{self.__test_name}_{self.__counter}.{self.__filetype}').resolve()

    def screenshot_all_monitors(self) -> numpy.ndarray:
        """
        Takes a Screenshot of all monitors, saves it as display.png in the stored output directory, and loads it to
        memory
        :return: screenshot of the monitor
        :rtype: numpy.ndarray
        """
        self.set_filename_to("display")
        self.__sct.shot(mon=1, output=str(self.__screenshot_name))
        return cv2.imread(str(self.__screenshot_name))

    def make_a_screenshot(self) -> (str, str, numpy.ndarray):
        """
        takes a screenshot of the current capture region.
        saves the screenshot
        :return: Hash, Image-Path, Image
        :rtype: (str, str, numpy.ndarray)
        """
        self.increment_file_name()
        screenshot = numpy.array(self.__sct.grab(self._capture_region))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(self.__screenshot_name), screenshot)
        hash_t = ImageHashing.marr_hildreth_hashing(screenshot)
        relpath = self.__screenshot_name.relative_to(self._output_dir)
        self.__screenshot_dictionary[hash_t] = (str(relpath), screenshot)
        self.__screenshot_path_dictionary[hash_t] = str(relpath)
        # info(f'<a href="{self.__screenshot_name}"><img src="file:///{self.__screenshot_name}"></a>', html=True)
        return hash_t, str(self._output_dir), str(relpath), screenshot

    def get_image_from_dictionary_by_hash(self, hash_t: str) -> (str, numpy.ndarray):
        """
        Returns the image and its path of the given hash
        :param hash_t: hash of the image
        :type hash_t: str
        :return: path, image
        :rtype: (str, numpy.ndarray)
        """
        return self.__screenshot_dictionary[hash_t]
