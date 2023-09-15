import cv2
import shutil
import pickle
import numpy

from inquire.loggers.ImageHashing import ImageHashing
from inquire.detection.Symbol import Symbol
from inquire.detection.TextElement import TextElement
import pathlib

try:
    from robot.api.logger import warn

    robot_logger = True
except ImportError:
    import logging

    robot_logger = False
    logging.basicConfig(level=logging.WARNING)


class ImageCache:

    def __init__(self):
        self.cache = {}
        self.path = pathlib.Path()

    def __str__(self):
        return str(self.cache)

    def __repr__(self):
        return self.__str__()

    def set_storage_path(self, path_t: pathlib.PurePath | str, relative: bool = False):
        """
        Sets the storage path for the cache. if one was already set, it's possible to give a relative path as argument
        else it needs to be an absolute path!
        :param path_t: absolute path (relative if previous set) to the storage directory of the cache
        :type path_t: pathlib.PurePath | str
        :param relative: if you want to use a relative path
        :type relative: bool
        :return: None
        """
        if isinstance(path_t, pathlib.PurePath):
            if relative:
                self.path = pathlib.PurePath(self.path, path_t)
            else:
                self.path = path_t
        if isinstance(path_t, str):
            if relative:
                self.path = self.path.joinpath(path_t)
            else:
                self.path = pathlib.Path(path_t)
        self.path.mkdir(parents=True, exist_ok=True)

    def add_element(self, hash_t, path: str, d1: dict, d2: dict, region: numpy.ndarray | None = None):
        if hash_t in self.cache:
            if robot_logger:
                warn(f'{path} image was already stored in the cache and will be overwritten', html=True)
                warn(f'{self.cache[hash_t]} is replaced by {d1}, {d2}', html=True)
            else:
                logging.warning(
                    f'{path} image was already stored in the cache and will be overwritten. And cache values will be replaced!')
        p = self.path.joinpath(f'{hash_t}.png').resolve()
        if region is None:
            shutil.copy(path, str(p))
        else:
            cv2.imwrite(str(p), region)
        self.cache[hash_t] = [str(p), d1, d2]

    def get_element(self, hast_t) -> list[str, dict, dict]:
        if hast_t in self.cache:
            return self.cache[hast_t]
        else:
            raise KeyError(f"Image Cache has not key {hast_t}")

    def has_hash(self, hash_t) -> bool:
        """
        Checks Whether a hash is already stored or not
        :param hash_t:
        :type hash_t:
        :return:
        :rtype:
        """
        return hash_t in self.cache

    def remove_element(self, hash_t):
        if hash_t in self.cache:
            del self.cache[hash_t]

    def clear_cache(self):
        self.cache.clear()

    def save_cache(self, filepath: str = "cache.pkl"):
        backup_file = self.path.joinpath(filepath).resolve()
        with open(str(backup_file), 'wb') as file:
            pickle.dump(self.cache, file)

    def load_cache(self, filepath: str = "cache.pkl"):
        backup_file = self.path.joinpath(filepath).resolve()
        if backup_file.is_file():
            with open(str(backup_file), 'rb') as file:
                cache = pickle.load(file)
                self.cache = cache
        else:
            self.cache = {}


if __name__ == '__main__':
    pass
