import os
import cv2
import pathlib
from os import path, getcwd

ROOT = getcwd()
FORMAT = cv2.VideoWriter_fourcc(*'VP90')


class Logger:
    ROBOT_LIBRARY_SCOPE = "TEST SUITE"
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = True
    __version__ = '0.1'

    def get_capture_region(self) -> dict[str, int]:
        """
        Returns the current capture region as dictionary with keys top,left,width,height
        :return: capture region
        :rtype: dict[str,int]
        """
        return self._capture_region

    def __init__(self):
        """
        Constructor of the Superclass Logger for VideoLogger and Screenshotlogger.
        :param output: absolute directory path where the videos or images should be saved
        :type output: str
        """
        self._output_dir = pathlib.Path.home().joinpath("result").resolve()
        self._capture_region = {"top": 0, "left": 0, "width": 0, "height": 0, "mon": 1}

    def set_capture_region(self, capture_region: list | dict) -> None:
        """
        Sets the capture region for the video recording.
        :param capture_region: [top,left,width,height]
        :type capture_region: list
        :return: None
        :rtype: None
        :raises TypeError, IndexError, ValueError
        """
        if type(capture_region) is dict:
            self._capture_region = capture_region
        else:
            if type(capture_region) is not list:
                raise TypeError("Capture_region needs to be a list")
            if len(capture_region) != 4:
                raise IndexError("Capture_region has to be of length 4")
            for i in capture_region:
                if type(i) is not int:
                    raise TypeError("Capture_region values need to be int")
                if i < 0:
                    raise ValueError("Negative values are not allowed")
            self._capture_region["top"] = capture_region[0]
            self._capture_region["left"] = capture_region[1]
            self._capture_region["width"] = capture_region[2]
            self._capture_region["height"] = capture_region[3]
            self._capture_region["mon"] = 1

    def set_filename_to(self, suite_name: str, test_name: str = "", filetype: str = ""):
        """
        Sets the filename and filetype to the given values for the next logged record
        :param test_name: Name of the test case
        :type test_name: str
        :param suite_name: Name of the test suite
        :type suite_name: str
        :param filetype: type of the file.
        :type filetype: str
        :return:None
        :rtype:
        """
        raise NotImplementedError

    def set_output_directory(self, output: str):
        """
        Changes the current output directory to the new given one
        :param output: Absolute path to the new Folder from the old one
        :type output: str
        :return: None
        :rtype:
        """
        p = pathlib.Path(output)
        p.mkdir(parents=True, exist_ok=True)
        self._output_dir = p
