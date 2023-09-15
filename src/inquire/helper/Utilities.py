class Utilities:
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self):
        pass

    @staticmethod
    def hex_rgb_to_bgr(hex_colour: str) -> tuple[int, int, int]:
        if len(hex_colour) != 6:
            raise IndexError("Hex string has to be of length 6")
        r = int(hex_colour[0:2], 16)
        g = int(hex_colour[2:4], 16)
        b = int(hex_colour[4:6], 16)
        return b, g, r

    @staticmethod
    def merge_rectangles(rec1: tuple | list, rec2: tuple | list, two_points=False) -> list[int, int, int, int]:
        """
        Merges the given rectangles to a new one. a rectangle with width and height of zero is ignored
        :param rec1: Rectangle given as (left,top,width,height) or (left,top,right,bottom)
        :type rec1: tuple | list
        :param rec2: Rectangle given as (left,top,width,height) or (left,top,right,bottom)
        :type rec2: tuple | list
        :param two_points: Must be true for rectangles in the form of (left,top,right,bottom)
        :type two_points: bool
        :return:
        :rtype: list[int,int,int,int]
        """
        if two_points:
            rec1_t = [rec1[0], rec1[1], rec1[2] - rec1[0], rec1[3] - rec1[1]]
            rec2_t = [rec2[0], rec2[1], rec2[2] - rec2[0], rec2[3] - rec2[1]]
        else:
            rec1_t = rec1
            rec2_t = rec2
        if rec1_t[2] == 0 and rec1_t[3] == 0:
            return rec2_t
        if rec2_t[2] == 0 and rec2_t[3] == 0:
            return rec1_t
        x_max = max(rec1_t[0] + rec1_t[2], rec2_t[0] + rec2_t[2])
        x = min(rec1_t[0], rec2_t[0])
        w = x_max - x
        y_max = max(rec1_t[1] + rec1_t[3], rec2_t[1] + rec2_t[3])
        y = min(rec1_t[1], rec2_t[1])
        h = y_max - y
        return [x, y, w, h]


if __name__ == '__main__':
    # test merge rectangles
    r1 = (10, 20, 30, 40)
    r2 = (5, 6, 0, 0)
    r = Utilities.merge_rectangles(r1, r2)
    assert r1 == r
    r = Utilities.merge_rectangles(r2, r1)
    assert r1 == r
    r2 = (15, 25, 40, 50)
    r = Utilities.merge_rectangles(r1, r2)
    assert (10, 20, 45, 55) == r
    r = Utilities.merge_rectangles(r2, r1)
    assert (10, 20, 45, 55) == r
