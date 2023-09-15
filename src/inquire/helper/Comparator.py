import numpy
import cv2


class Comparator:
    """
    Class to handle the comparison of multiple elements used in this project
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self):
        pass

    @staticmethod
    def ciede2000(lab_colour, expected_lab_colour):
        """
        Calculates the difference between to colours using the CIEDE2000 algorithm
        https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
        :param lab_colour:
        :type lab_colour:
        :param expected_lab_colour:
        :type expected_lab_colour:
        :return:
        :rtype:
        """
        # Convert LAB values to double precision
        L1, a1, b1 = map(float, lab_colour)
        L2, a2, b2 = map(float, expected_lab_colour)

        # Calculate dLp, dCp, and dHp
        dLp = L2 - L1
        C1 = numpy.sqrt(a1 ** 2 + b1 ** 2)
        C2 = numpy.sqrt(a2 ** 2 + b2 ** 2)
        Cbar = (C1 + C2) / 2
        G = 0.5 * (1 - numpy.sqrt((Cbar ** 7) / (Cbar ** 7 + 25 ** 7)))
        ap1 = a1 * (1 + G)
        ap2 = a2 * (1 + G)
        Cp1 = numpy.sqrt(ap1 ** 2 + b1 ** 2)
        Cp2 = numpy.sqrt(ap2 ** 2 + b2 ** 2)
        Cbarp = (Cp1 + Cp2) / 2
        dCp = Cp2 - Cp1
        hp1 = numpy.degrees(numpy.arctan2(b1, ap1))
        hp2 = numpy.degrees(numpy.arctan2(b2, ap2))
        dhp = numpy.where(numpy.abs(hp1 - hp2) <= 180, hp2 - hp1, (hp2 - hp1 + 360) % 360 - 180)
        dHp = 2 * numpy.sqrt(Cp1 * Cp2) * numpy.sin(numpy.radians(dhp) / 2)
        Lbarp = (L1 + L2) / 2
        T = 1 - 0.17 * numpy.cos(numpy.radians(hp1 - 30)) + 0.24 * numpy.cos(numpy.radians(2 * hp1)) + 0.32 * numpy.cos(
            numpy.radians(3 * hp1 + 6)) - 0.20 * numpy.cos(numpy.radians(4 * hp1 - 63))
        SL = 1 + ((0.015 * (Lbarp - 50) ** 2) / numpy.sqrt(20 + (Lbarp - 50) ** 2))
        SC = 1 + 0.045 * Cbarp
        SH = 1 + 0.015 * Cbarp * T
        RT = -2 * numpy.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25 ** 7)) * numpy.sin(
            numpy.radians(60 * numpy.exp(-((hp1 - 275) / 25) ** 2)))
        dE00 = numpy.sqrt(
            (dLp / (SL * 1.0)) ** 2 + (dCp / (SC * 1.0)) ** 2 + (dHp / (SH * 1.0)) ** 2 + RT * (dCp / (SC * 1.0)) * (
                    dHp / (SH * 1.0)))
        return dE00

    @staticmethod
    def is_similar_colour(bgr_colour, expected_bgr_colour, threshold=10) -> bool:
        """
        Analyses the two given colours and returns true if they look similar to a human eye
        TODO find optimal treshhold
        :param bgr_colour:
        :type bgr_colour:
        :param expected_bgr_colour:
        :type expected_bgr_colour:
        :param threshold:
        :type threshold:
        :return:
        :rtype:
        """
        rgb_array = numpy.array([[bgr_colour]], dtype=numpy.uint8)
        ex_rgb_array = numpy.array([[expected_bgr_colour]], dtype=numpy.uint8)
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2Lab)
        ex_lab = cv2.cvtColor(ex_rgb_array, cv2.COLOR_BGR2Lab)
        diff = Comparator.ciede2000(lab[0][0], ex_lab[0][0])
        return diff <= threshold

    @staticmethod
    def images_should_be_equal(arr1: numpy.ndarray, arr2: numpy.ndarray) -> bool:
        """
        Checks whether the two given images are equal with pixel comparison
        :param arr1: image1
        :type arr1: numpy.ndarray
        :param arr2: image2
        :type arr2: numpy.ndarray
        :return: True|False
        :rtype: bool
        """
        return (arr1 == arr2).all()

    @staticmethod
    def intersect_rectangles(rec1: tuple | list | numpy.ndarray, rec2: tuple | list | numpy.ndarray) -> bool:
        """
        Checks whether two rectangles intersect/overlap each other.
        :param rec1: (x,y,width,height)
        :type rec1: tuple | list | numpy.ndarray
        :param rec2: (x,y,width,height)
        :type rec2: tuple | list | numpy.ndarray
        :return: True if they intersect
        :rtype: bool
        :raises TypeError | IndexError
        """
        if type(rec1) is not tuple or type(rec2) is not tuple:
            if type(rec1) is not list or type(rec2) is not list:
                if type(rec1) is not numpy.ndarray or type(rec2) is not numpy.ndarray:
                    raise TypeError("Rectangles have to be tuples or list")
        if len(rec1) != 4 or len(rec2) != 4:
            raise IndexError("Rectangles have to be of length 4")
        x1_s = rec1[0]
        x1_e = rec1[0] + rec1[2] - 1
        y1_s = rec1[1]
        y1_e = rec1[1] + rec1[3] - 1
        x2_s = rec2[0]
        x2_e = rec2[0] + rec2[2] - 1
        y2_s = rec2[1]
        y2_e = rec2[1] + rec2[3] - 1
        if Comparator.intersect_interval((x1_s, x1_e), (x2_s, x2_e)):
            return Comparator.intersect_interval((y1_s, y1_e), (y2_s, y2_e))

    @staticmethod
    def intersect_interval(i1: tuple, i2: tuple) -> bool:
        """
        Checks whether the two intervals overlap
        :param i1: [min,max]
        :type i1: tuple
        :param i2: [min,max]
        :type i2: tuple
        :return: True if they overlap
        :rtype: bool
        :raises TypeError | IndexError
        """
        if type(i1) is not tuple or type(i2) is not tuple:
            raise TypeError("Intervals have to be tuples")
        if len(i1) != 2 or len(i2) != 2:
            raise IndexError("Intervals have to be of length 2")
        return i1[0] <= i2[0] <= i1[1] or i1[0] <= i2[1] <= i1[1] or i2[0] <= i1[0] <= i2[1] or i2[0] <= i1[1] <= i2[1]

    @staticmethod
    def position_key_of_recognized_elements_position(ele) -> int:
        """
        Function for sorting symbols and text elements by their top left corner. starting from the top left  line by line
        :param ele: Symbol or Text element
        :type ele: RecognizedElement
        :return: ye4+x
        :rtype: int
        """
        return ele.rectangle[1] * 10000 + ele.rectangle[0]

    @staticmethod
    def height_key_of_recognized_elements(ele) -> int:
        """
        Function for sorting symbols and text elements by their height .
        :param ele: Symbol or Text element
        :type ele: RecognizedElement
        :return: height
        :rtype: int
        """
        return ele.rectangle[3]

    @staticmethod
    def x_coord_key_of_recognized_elements(ele) -> int:
        """
        Function for sorting symbols and text elements by their x-coordinate.
        :param ele: Symbol or Text element
        :type ele: RecognizedElement
        :return: height
        :rtype: int
        """
        return ele.rectangle[0]

    @staticmethod
    def position_key_of_rectangles_tuple(rec1: tuple) -> int:
        """
        Function for sorting rectangles according to their position. starting from the top left  line by line
        :param rec1: rectangle with (x,y,width,height)
        :type rec1: tuple
        :return: ye4+x
        :rtype: int
        :raises TypeError | IndexError
        """
        if type(rec1) is not tuple:
            raise TypeError("Rectangles have to be tuples")
        if len(rec1) != 4:
            raise IndexError("Rectangles have to be of length 4")
        return rec1[1] * 10000 + rec1[0]

    @staticmethod
    def area_key_of_rectangles_tuple(rec1: tuple) -> int:
        """
        Function for sorting rectangles according to their area with the sorted() function, where this is used as key
        :param rec1: rectangle with (x,y,width,height)
        :type rec1: tuple
        :return: Area
        :rtype: int
        :raises TypeError | IndexError
        """
        if type(rec1) is not tuple:
            raise TypeError("Rectangles have to be tuples")
        if len(rec1) != 4:
            raise IndexError("Rectangles have to be of length 4")
        return rec1[2] * rec1[3]

    @staticmethod
    def euclid_distance_key_of_rectangles_tuple(rec1: tuple) -> int:
        """
        Function for sorting rectangles according to their euclidian distance with the sorted() function, where this is used as key
        :param rec1: rectangle with (x,y,width,height)
        :type rec1: tuple
        :return: distance
        :rtype: int
        :raises TypeError | IndexError
        """
        if type(rec1) is not tuple:
            raise TypeError("Rectangles have to be tuples")
        if len(rec1) != 4:
            raise IndexError("Rectangles have to be of length 4")
        return rec1[0] ** 2 + rec1[1] ** 2 + rec1[2] ** 2 + rec1[3] ** 2

    @staticmethod
    def is_interval_in(t1: tuple, t2: tuple) -> bool:
        """
        Checks whether the given interval t1 = (x0,x1) is within the interval t2 = (x2,x3)
        :param t1: Interval from x0 to x1
        :type t1: tuple
        :param t2: Interval from x2 to x3
        :type t2: tuple
        :return: true if t1 is a subset ob t2
        :rtype: bool
        :raises NotImplementedError
        """
        if len(t1) != 2 or len(t2) != 2:
            raise NotImplementedError
        return t2[0] <= t1[0] and t1[1] <= t2[1]


if __name__ == '__main__':
    # Test is similiar collour
    c = Comparator
    print(c.is_similar_colour((128, 255, 128), (128, 255, 0), 10))
    # Tests intersect_interval
    # i1 = (10, 20)
    # i = [(1, 9), (2, 10), (3, 11), (12, 19), (13, 20), (14, 21), (9, 12), (10, 13), (11, 14), (19, 29), (20, 30),
    #      (21, 31)]
    # ans = [False, True, True, True, True, True, True, True, True, True, True, False]
    # for j in range(len(i)):
    #     assert ans[j] == Comparator.intersect_interval(i1, i[j])
    #     assert ans[j] == Comparator.intersect_interval(i[j], i1)
    # # Tests intersect_rectangles
    # rec = (10, 30, 10, 20)
    # i = [(11, 31, 8, 18), (11, 30, 8, 18), (11, 29, 8, 18), (11, 32, 8, 18), (11, 33, 8, 18), (10, 31, 8, 18),
    #      (9, 31, 8, 18), (12, 31, 8, 18), (14, 31, 8, 18), (9, 31, 12, 18), (11, 29, 8, 22), (9, 29, 12, 22),
    #      (10, 30, 10, 20)]
    # for j in i:
    #     assert Comparator.intersect_rectangles(rec, j)
    #     assert Comparator.intersect_rectangles(j, rec)
    # i = [(10, 20, 10, 10), (10, 50, 10, 10), (0, 30, 10, 20), (20, 30, 10, 20)]
    # for j in i:
    #     assert not Comparator.intersect_rectangles(rec, j)
    #     assert not Comparator.intersect_rectangles(j, rec)
