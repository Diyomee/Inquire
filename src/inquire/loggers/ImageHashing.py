import numpy
import cv2
from os import path

try:
    from robot.api.logger import info

    robot_logger = True
except ImportError:
    import logging

    robot_logger = False
    logging.basicConfig(level=logging.INFO)


class ImageHashing:
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self):
        pass

    @staticmethod
    def load_image_for_tests(directory: str, filepath: str) -> numpy.ndarray:
        """
        Loads the given image into memory for later use
        :param directory: Absolute Path to the directory
        :type directory:  str
        :param filepath: relative path to the image from the directory
        :type filepath: str
        :return: image
        :rtype: numpy.ndarray
        """
        img_path = path.join(path.normpath(directory), path.normpath(filepath))
        if path.exists(img_path):
            img = cv2.imread(img_path)
            return img
        else:
            raise FileNotFoundError(img_path)

    @staticmethod
    def _generate_hex_hash_string_from_binary_array(bin_arr: numpy.array) -> str:
        """
        Generates the Hash from the given Binary Array.
        :param bin_arr:  Binary Array to be hashed
        :type bin_arr: numpy:ndarray
        :return: Hex-String
        :rtype: str
        """
        bin_string = ''.join(str(bit) for bit in 1 * bin_arr)
        str_length = int(numpy.ceil(len(bin_string) / 4))
        hex_hash = '{:0>{width}x}'.format(int(bin_string, 2), width=str_length)
        return hex_hash

    @staticmethod
    def average_hashing(image: numpy.ndarray, hash_size: int = 8) -> str:
        """
        Calculates the Hash of an image using the average hashing method
        https://web.archive.org/web/20171112054354/https://www.safaribooksonline.com/blog/2013/11/26/image-hashing-with-python/
        :param image: RGB numpy.ndarray, with the image data
        :type image: numpy.ndarray
        :param hash_size: Size of the hash
        :type hash_size: int
        :return: hash string
        :rtype: str
        """
        # https://docs.opencv.org/3.4/db/d25/classcv_1_1img__hash_1_1AverageHash.html
        avg_hash = cv2.img_hash.AverageHash_create()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hash_arr = avg_hash.compute(image)
        aH = hex(int.from_bytes(hash_arr.tobytes(), byteorder='big', signed=False))
        return aH

        # image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # avg = numpy.mean(image)
        # bits = numpy.array([], dtype=bool)
        # for i in numpy.nditer(image):
        #     bits = numpy.append(bits, i > avg)
        # hex_hash = ImageHashing._generate_hex_hash_string_from_binary_array(bits)
        # return hex_hash

    @staticmethod
    def perceptual_hashing(image: numpy.ndarray, hash_size: int = 8, scale_factor=4) -> str:
        """
        Calculates the Hash of an image using the perceptual hashing method
        It's very robust towards changes. Up to 25% of the image can be altered. Expensive. high accuracy
        :param image:RGB numpy.ndarray, with the image data
        :type image: numpy.ndarray
        :param hash_size: Size of the hash
        :type hash_size: int
        :param scale_factor: Multiplier for the size for the dct algorithm
        :type scale_factor: int
        :return: hash string
        :rtype: str
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hash_arr = cv2.img_hash.pHash(image)
        hex_hash = hex(int.from_bytes(hash_arr.tobytes(), byteorder='big', signed=False))
        return hex_hash

    @staticmethod
    def difference_hashing(image: numpy.ndarray, hashSize: int = 8) -> str:
        """
        Calculates the Hash of an image using the difference hashing method.
        Nearly identical to average Hashing but with a better performance.
        http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
        :param image: RGB numpy.ndarray, with the image data
        :type image: numpy.ndarray
        :param hashSize:
        :type hashSize:
        :return:hash string
        :rtype: str
        """
        if image.size != 0:
            image = cv2.resize(image, (hashSize + 1, hashSize))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            diff = image[1:, :] > image[:-1, :]
            bits = numpy.array([], dtype=bool)
            for i in numpy.nditer(diff):
                bits = numpy.append(bits, i)
            hex_hash = ImageHashing._generate_hex_hash_string_from_binary_array(bits)
        else:
            if robot_logger:
                info(f'difference hashing img empty size 0', html=True, also_console=True)
            else:
                logging.info(f'difference hashing img empty size 0')
            hex_hash = "00000000"
        return hex_hash

    def wavelet_hashing(self):
        raise NotImplementedError

    def crop_resistant_hashing(self):
        raise NotImplementedError

    @staticmethod
    def radial_variance_hashing(image: numpy.ndarray) -> str:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        colour_hash = cv2.img_hash.radialVarianceHash(image)
        hex_hash = hex(int.from_bytes(colour_hash.tobytes(), byteorder='big', signed=False))
        return hex_hash

    @staticmethod
    def marr_hildreth_hashing(image: numpy.ndarray) -> str:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        colour_hash = cv2.img_hash.marrHildrethHash(image)
        hex_hash = hex(int.from_bytes(colour_hash.tobytes(), byteorder='big', signed=False))
        return hex_hash

    def img_hash_base(self):
        raise NotImplementedError

    @staticmethod
    def color_moment_hashing(image: numpy.ndarray) -> str:
        """
        Calculates the Colour-hash of the given image
        http://www.phash.org/docs/pubs/thesis_zauner.pdf
        :param image:RGB numpy.ndarray, with the image data
        :type image: numpy.ndarray
        :return: hash string
        :rtype: str
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        colour_hash = cv2.img_hash.colorMomentHash(image)
        hex_hash = hex(int.from_bytes(colour_hash.tobytes(), byteorder='big', signed=False))
        return hex_hash

    @staticmethod
    def block_mean_hashing(image: numpy.ndarray) -> str:
        """
        Calculates the Block-mean-hash of the given image
        Block mean value based image perceptual
        hashing. In Proceedings of the International Conference on Intelligent
        Information Hiding and Multimedia Multimedia Signal Processing
        https://phash.org/docs/pubs/thesis_zauner.pdf
        :param image:RGB numpy.ndarray, with the image data
        :type image: numpy.ndarray
        :return: hash string
        :rtype: str
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block_hash = cv2.img_hash.blockMeanHash(image, cv2.img_hash.BLOCK_MEAN_HASH_MODE_0)
        hex_hash = hex(int.from_bytes(block_hash.tobytes(), byteorder='big', signed=False))
        return hex_hash
