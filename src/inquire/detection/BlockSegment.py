import numpy


class BlockSegment:
    """
    Datastructure to save the information about each detected block from the topdown approach.
    To Access or Change any values use the member variables top, left, width, height, gray_block, colour_block,
    binary_block
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self, top: int, left: int, width: int, height: int, img: numpy.ndarray | None = None,
                 img_gray: numpy.ndarray | None = None,
                 img_bin: numpy.ndarray | None = None, offset_top: int = 0, offset_left: int = 0):
        """
        Constructor for a Block Segment to save one Block
        :param top: Top left corners y-Coordinate in the screenshot of the detected block
        :type top: int
        :param left: Top left corners x-Coordinate in the screenshot of the detected block
        :type left: int
        :param width: The width of the detected block
        :type width: int
        :param height: The height of the detected block
        :type height: int
        :param img: The cut RGB image from the above region of the screenshot
        :type img: Numpy.ndarray
        :param img_gray: The cut Gray image from the above region of the screenshot
        :type img_gray: Numpy.ndarray
        :param img_bin: The cut Binary image from the above region of the screenshot
        :type img_bin: Numpy.ndarray
        """
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.offset_top = offset_top
        self.offset_left = offset_left
        self.gray_block = img_gray
        self.colour_block = img
        self.binary_block = img_bin
        self.detected = {}

    def __str__(self):
        return f'x:{self.left},y:{self.top},w:{self.width},h:{self.height},ox:{self.offset_left},oy:{self.offset_top}'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            if self.top * self.left == other.top * other.left:
                return self.width * self.height < other.width * other.height
            else:
                return self.top * self.left < other.top * other.left
        else:
            raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other,
                          self.__class__) and self.top == other.top and self.left == other.left and self.width == other.width and self.height == other.height

    def __hash__(self):
        return hash((self.top, self.left, self.width, self.height))

    def get_dimensions_as_dictionary(self) -> dict:
        """
        Gets the dimension of the block segment for the screenshot as a dictionary with keys top,left,width,height
        :return: dictionary with keys top,left,width,height
        :rtype: dict
        """
        return {"top": self.top, "left": self.left, "width": self.width, "height": self.height}
