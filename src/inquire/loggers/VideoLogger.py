import time

import cv2
from os import path, getcwd, mkdir
import threading
import numpy
from mss import mss
import mouse
from inquire.loggers.Logger import Logger

try:
    from robot.api.logger import info

    robot_logger = True
except ImportError:
    import logging

    robot_logger = False
    logging.basicConfig(level=logging.WARNING)

ROOT = getcwd()
FORMAT = cv2.VideoWriter_fourcc(*'VP90')
FRAMERATE = 10.0
NANOSECONDS = 10 ** 9
RECORDING_DELAY = NANOSECONDS / FRAMERATE
MOUSE_X = [0, 4, 3, 7, 6, 2, 1, 0]
MOUSE_Y = [0, 1, 2, 6, 7, 3, 4, 0]


class VideoLogger(Logger):

    def __init__(self):
        """
        Constructor of the VideoLogger Class. It controls the recording of the screen, and saves the recorded
        video into the given output directory
        """
        super().__init__()
        self.__video_dir = self._output_dir
        self.__video_name = self._output_dir.joinpath("v1.webm").resolve()
        self.__recording_thread = None
        self.__is_recording = threading.Event()

    def set_filename_to(self, suite_name: str, test_name: str = "", filetype: str = "webm") -> None:
        """
        Sets the filename and filetype to the given values for the next video recording
        :param test_name: Name of the test case
        :type test_name: str
        :param suite_name: Name of the test suite
        :type suite_name: str
        :param filetype: type of the file. default webm
        :type filetype: str
        :return:None
        :rtype:None
        """
        self.__video_dir = self._output_dir.joinpath(f'{suite_name}').resolve()
        self.__video_dir.mkdir(parents=True, exist_ok=True)
        self.__video_name = self.__video_dir.joinpath(f'{test_name}.{filetype}').resolve()
        if robot_logger:
            info(f'File path is {self.__video_name}', html=True)

    def stop_recording(self) -> None:
        """
        Sends a clearing event to the recording thread for stopping the video recording, and waits for its completion.
        :return: None
        :rtype: None
        """
        if self.__is_recording.is_set():
            if self.__recording_thread is not None:
                self.__is_recording.clear()
                self.__recording_thread.join()
                relpath = self.__video_name.relative_to(self._output_dir)
                if robot_logger:
                    info(
                        f'<a href="{relpath}"><video width="800" controls><source src="{relpath}" type="video/webm"></video></a>',
                        html=True)
            else:
                self.__is_recording.clear()

    def start_process_for_video_recoding(self) -> None:
        """
        Start a new process for recording the current screen, and stops and waits a running recording thread
        :return: None
        :rtype: None
        """
        self.stop_recording()
        self.__recording_thread = threading.Thread(target=self.__record_video, daemon=True)
        self.__recording_thread.start()

    def print_cursor_onto_frame(self, frame):
        mouseX, mouseY = mouse.get_position()
        mouseX = min(max(0, mouseX - self._capture_region["left"]), frame.shape[1] - 1)
        mouseY = min(max(0, mouseY - self._capture_region["top"]), frame.shape[0] - 1)
        colour = 255 - frame[mouseY][mouseX]
        Xthis = [4 * x + mouseX for x in MOUSE_X]
        Ythis = [4 * y + mouseY for y in MOUSE_Y]
        points = list(zip(Xthis, Ythis))
        points = numpy.array(points, 'int32')
        try:
            cv2.fillPoly(frame, [points], color=(int(colour[0]), int(colour[1]), int(colour[2])))
        except Exception:
            raise Exception(f'{(int(colour[0]), int(colour[1]), int(colour[2]))}')
        return frame

    def __record_video(self) -> None:
        """
        Handles the recording of the video.
        It creates the video writer, takes screenshots, and appends them to the video according to the frame rate.
        upon receiving the clearing of the recoding event (__is_recording) it will stop the recording and saves the
        last frame.
        :return: None
        :rtype: None
        """
        self.__is_recording.set()
        sct = mss()
        self.__video_writer = cv2.VideoWriter(str(self.__video_name), FORMAT, FRAMERATE,
                                              (self._capture_region["width"], self._capture_region["height"]))
        while self.__is_recording.is_set():
            t_0 = time.perf_counter_ns()
            screenshot = numpy.array(sct.grab(self._capture_region))
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            screenshot = self.print_cursor_onto_frame(screenshot)
            self.__video_writer.write(screenshot)
            t_1 = time.perf_counter_ns()
            delta_t = RECORDING_DELAY - t_1 + t_0
            if delta_t > 0:
                time.sleep(delta_t / NANOSECONDS)
        screenshot = numpy.array(sct.grab(self._capture_region))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        self.__video_writer.write(screenshot)
