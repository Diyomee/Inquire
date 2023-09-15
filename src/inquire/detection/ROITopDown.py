import itertools
import math
import pathlib

import cv2
import numpy

from inquire.loggers.ImageHashing import ImageHashing
from inquire.ocr.PaddleOCR import PaddleOCR
from inquire.detection.ROIDetector import ROIDetector
from inquire.helper.Comparator import Comparator
from inquire.helper.SvgToPngGenerator import SvgToPngGenerator


class ROITopDown(ROIDetector):
    """
    TODO a lot
    Class to manage the topdown approach for the region of interest detection
    """

    def __init__(self):
        super().__init__()
        self.colours = []
        self.generate_colour_table()
        self.draw_rois = False
        self.draw_floodfill = False

    def generate_colour_table(self):
        for i, j, k in itertools.product(range(4), repeat=3):
            if i == 0 and j == 0 and k == 0:
                continue
            self.colours.append((i * 64, j * 64, k * 64))

    @staticmethod
    def cut_rectangle(b_rec: tuple, s_rec: tuple) -> list[tuple]:
        """
        vertically cuts the given rectangle b_rec depending on the position of the smaller rectangle s_rec.
        if not vertical cuts are possible it will cut horizontally.
        :param b_rec: bigger rectangle, which should be cut
        :type b_rec:
        :param s_rec:
        :type s_rec:
        :return:
        :rtype:
        """
        if type(b_rec) is not tuple or type(s_rec) is not tuple:
            raise TypeError("args have to be tuples")
        if len(b_rec) != 4 or len(s_rec) != 4:
            raise IndexError("has to of length 4")
        bx, by, bw, bh = b_rec
        sx, sy, sw, sh = s_rec
        # ----bx---sx-----sx+sw-----bx+bw------
        if bx <= sx <= sx + sw <= bx + bw:
            a = (bx, by, sx - bx, bh)
            c = (sx + sw, by, bw - sx + bx - sw, bh)
            # ----by----sy------sy+sh------by+bh------
            if by <= sy <= sy + sh <= by + bh:
                b = (sx, by, sw, sy - by)
                d = (sx, sy + sh, sw, bh - sy + by - sh)
                return [a, b, c, d]
            # ----sy----by----sy+sh-----by+bh
            elif sy <= by <= sy + sh <= by + bh:
                d = (sx, sy + sh, sw, bh - sh + by - sy)
                return [a, c, d]
            # ----by----sy-------by+bh-----sy+sh
            elif by <= sy <= by + bh <= sy + sh:
                b = (sx, by, sw, sy - by)
                return [a, b, c]
            # -----sy-----by-----by+bh----sy+sh
            elif sy <= by <= by + bh <= sy + sh:
                return [a, c]
            # -----sy-----sy+sh----by-----by+bh
            # -----by-----by+bh----sy-----sy+sh
            else:
                return [b_rec]
        # ------sx-----bx-----sx+sw------bx+bw------
        elif sx <= bx <= sx + sw <= bx + bw:
            c = (sx + sw, by, bw - sw + bx - sx, bh)
            # ----by----sy------sy+sh------by+bh------
            if by <= sy <= sy + sh <= by + bh:
                a = (bx, by, sw - bx + sx, sy - by)
                b = (bx, sy + sh, sw - bx + sx, bh - sh + by - sy)
                return [a, b, c]
            # ----sy----by----sy+sh-----by+bh
            elif sy <= by <= sy + sh <= by + bh:
                b = (bx, sy + sh, sw - bx + sx, bh - sh + by - sy)
                return [b, c]
            # ----by----sy-------by+bh-----sy+sh
            elif by <= sy <= by + bh <= sy + sh:
                a = (bx, by, sw - bx + sx, sy - by)
                return [a, c]
            # -----sy-----by-----by+bh----sy+sh
            else:
                return [c]
        # -------bx----sx-----bx+bw------sx+sw
        elif bx <= sx <= bx + bw <= sx + sw:
            a = (bx, by, sx - bx, bh)
            # ----by----sy------sy+sh------by+bh------
            if by <= sy <= sy + sh <= by + bh:
                b = (sx, by, bw - sx + bx, sy - by)
                c = (sx, sy + sh, bw - sx + bx, bh - sh + by - sy)
                return [a, b, c]
            # ----sy----by----sy+sh-----by+bh
            elif sy <= by <= sy + sh <= by + bh:
                c = (sx, sy + sh, bw - sx + bx, bh - sh + by - sy)
                return [a, c]
            # ----by----sy-------by+bh-----sy+sh
            elif by <= sy <= by + bh <= sy + sh:
                b = (sx, by, bw - sx + bx, sy - by)
                return [a, b]
            # -----sy-----by-----by+bh----sy+sh
            else:
                return [a]
        # ----sx----bx-----bx+bw-------sx+sw
        elif sx <= bx <= bx + bw <= sx + sw:
            # ----by----sy------sy+sh------by+bh------
            if by <= sy <= sy + sh <= by + bh:
                a = (bx, by, bw, sy - by)
                b = (bx, sy + sh, bw, bh - sh - sy + by)
                return [a, b]
            # ----sy----by----sy+sh-----by+bh
            elif sy <= by <= sy + sh <= by + bh:
                c = (bx, sy + sh, bw, bh - sh + by - sy)
                return [c]
            # ----by----sy-------by+bh-----sy+sh
            elif by <= sy <= by + bh <= sy + sh:
                a = (bx, by, bw, sy - by)
                return [a]
            else:
                return [b_rec]
        # ---bx----bx+bw-----sx------sx+sw
        # ---sx----sx+sw-----bx------bx+bw
        else:
            return [b_rec]

    @staticmethod
    def increase_rectangle(rect: tuple, img_shape):
        x, y, w, h = rect
        if x - 4 > 0:
            x -= 2
        else:
            x = 0
        if y - 4 > 0:
            y -= 2
        else:
            y = 0
        if w + 4 <= img_shape[1]:
            w += 4
        else:
            w = img_shape[1]
        if h + 4 <= img_shape[0]:
            h += 4
        else:
            h = img_shape[0]
        return x, y, w, h

    def flood_filling_algorithm(self) -> list[tuple]:
        """
        Utilizes the flood fill algorithm to find large Region of Interest which could represent a group of gui elements
        within the same context
        :return: list of rectangles
        :rtype: list[tuple]
        """
        img = self._binary_img.copy()
        treshhold = img.size * 0.06
        orig = self._img.copy()
        mask = numpy.zeros((img.shape[0] + 2, img.shape[1] + 2), numpy.uint8)
        floodflags = 8
        floodflags |= cv2.FLOODFILL_MASK_ONLY
        # floodflags |= cv2.FLOODFILL_FIXED_RANGE   better never set
        floodflags |= (255 << 8)
        found_regions = []
        for index in numpy.ndindex(img.shape):
            if type(index) is tuple:
                if mask[index[0]][index[1]] == 0:
                    num, img, mask, rect = cv2.floodFill(img, mask, (index[1], index[0]), 255, 3, 3, floodflags)

                    if num >= treshhold:
                        found_regions.append(rect)
        found_regions = sorted(found_regions, key=Comparator.area_key_of_rectangles_tuple, reverse=True)
        found_regions = self.generate_non_overlapping_regions(found_regions)
        if self.draw_floodfill:
            tosave = orig.copy()
            for region in found_regions:
                tosave = cv2.rectangle(tosave, (region[0], region[1]), (region[0] + region[2], region[1] + region[3]),
                                       (0, 255, 255), 3)
            hash_t = ImageHashing.marr_hildreth_hashing(tosave)
            hash_t += ".png"
            CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
            dir = CURRENT_PATH.parent.parent.parent.parent.joinpath("Daten", "roi", ).resolve().joinpath(hash_t)
            cv2.imwrite(str(dir), tosave)
        found_regions = sorted(found_regions, key=Comparator.position_key_of_rectangles_tuple, reverse=False)
        return found_regions

    def block_analysis(self, regions: list[tuple]) -> dict[str, list[tuple[int, int, int, int]]]:
        """
        Analyses given blocks/groups of the image. Those groups should have been extracted through flood filling.
        It's extracts from each group, regions of interest (ROIs) where Symbols and Text could be located.
        :param regions:
        :type regions:
        :return: dictionary the Rois per found region(keys).
        :rtype: dict
        """
        ans = {}
        counter = 1
        if self.draw_rois:
            new_dil = self._dilated_img.copy()
        else:
            new_dil = None
        for region in regions:
            # found_text = self.ocr_image(region)
            found_rois = []
            dil = self._dilated_img.copy()
            dil = dil[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
            # Component labeling algorithm using bin trees, to detect the Region of Interest (ROI)
            (numlabels, labels, stats, centroid) = cv2.connectedComponentsWithStats(dil, 8, cv2.CV_32S)
            for j in range(1, numlabels):
                x = max(stats[j, cv2.CC_STAT_LEFT] - 2, 0)
                y = max(stats[j, cv2.CC_STAT_TOP] - 2, 0)
                w = min(stats[j, cv2.CC_STAT_WIDTH] + 4, region[2] - x)
                h = min(stats[j, cv2.CC_STAT_HEIGHT] + 4, region[3] - y)
                # Discards to Small ROIs
                if w <= 15 or h <= 15:
                    continue
                area = stats[j, cv2.CC_STAT_AREA]
                (cx, cy) = centroid[j]
                (mx, my) = (int(x + w / 2), int(y + h / 2))
                d = math.sqrt(((mx - int(cx)) / w) ** 2 + ((my - int(cy)) / h) ** 2)
                # Discards ROIs which Center of Mass is too far off from the middle point
                if d >= 0.2:
                    continue
                found_rois.append((x, y, w, h))

                if self.draw_rois:
                    dil = cv2.rectangle(dil, (x, y), (x + w, y + h), 125, 5)
            # groups overlapping rectangles
            found_rois.extend(found_rois)
            rectangles, weights = cv2.groupRectangles(found_rois, groupThreshold=1, eps=0.2)
            # merge overlapping rectangles
            # TODO rewrite the following method
            rectangles = self.merge_overlapping_rectangles(rectangles)
            # rectangles = found_rois
            # TODO end
            if rectangles is not None:
                if self.draw_rois:
                    for rec in rectangles:
                        new_dil = cv2.rectangle(new_dil, (rec[0] + region[0], rec[1] + region[1]),
                                                (rec[0] + region[0] + rec[2], rec[1] + region[1] + rec[3]), 192, 2)
                rectangles.insert(0, region)
                s = f'R{counter}'
                ans[s] = rectangles
                counter += 1
        if self.draw_rois:
            hash_t = cv2.img_hash.marrHildrethHash(new_dil)
            hash_t = hex(int.from_bytes(hash_t.tobytes(), byteorder='big', signed=False))
            hash_t += ".png"
            CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
            dir = CURRENT_PATH.parent.parent.parent.parent.joinpath("Daten", "roi", ).resolve().joinpath(hash_t)
            cv2.imwrite(str(dir), new_dil)
        return ans

    def generate_non_overlapping_regions(self, regions: list[tuple]):
        if len(regions) <= 1:
            return regions
        i = 0
        j = 1
        while i < len(regions) - 1:
            a = regions[i]
            b = regions[j]
            if Comparator.intersect_rectangles(a, b):
                c = self.cut_rectangle(a, b)
                c = self.discard_small_rects(c, 8, 4, 64)
                del regions[i]
                regions.extend(c)
                regions = sorted(regions, key=Comparator.area_key_of_rectangles_tuple, reverse=True)
                i = 0
                j = 1
            else:
                j += 1
                if j == len(regions):
                    i += 1
                    j = i + 1
        return regions

    def test_rectangle(self, t1, t2):
        img = numpy.zeros((1000, 1000, 3), dtype="uint8")
        cv2.rectangle(img, (t1[0], t1[1]), (t1[0] + t1[2], t1[1] + t1[3]), (0, 255, 0), -1)
        cv2.rectangle(img, (t2[0], t2[1]), (t2[0] + t2[2], t2[1] + t2[3]), (0, 0, 255), -1)
        l2 = self.cut_rectangle(t1, t2)
        for i in l2:
            img = cv2.rectangle(img, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (255, 0, 0), 5)
        cv2.imshow("new", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def increment_colour(self, index):
        return self.colours[index % 64]

    @staticmethod
    def discard_small_rects(regions: list[tuple], w_thresh: int, h_thresh: int, a_thresh: int):
        ans = []
        for i in regions:
            if i[2] < w_thresh:
                continue
            if i[3] < h_thresh:
                continue
            if i[2] * i[3] < a_thresh:
                continue
            ans.append(i)
        return ans

    def print_rectangles(self, regions: list[tuple]):
        img = numpy.zeros((1000, 1000, 3), dtype="uint8")
        colour = (0, 0, 0)
        c = 0
        for i in regions:
            colour = self.increment_colour(c)
            img = cv2.rectangle(img, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), colour, -1)
            c += 1
        cv2.imshow("founds", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    pass
