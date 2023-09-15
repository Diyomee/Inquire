import numpy
import cv2

try:
    from robot.api.logger import console, info

    robot_logger = True
except ImportError:
    import logging

    robot_logger = False
    logging.basicConfig(level=logging.INFO)


class TemplateMatcher:
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def match_many_scaled(self, image: numpy.ndarray, template: numpy.ndarray, confidence: float, threshold: float):
        """
        TODO
        :param image:
        :type image:
        :param template:
        :type template:
        :param confidence:
        :type confidence:
        :param threshold:
        :type threshold:
        :return:
        :rtype:
        """
        raise NotImplementedError
        template_width, template_height = template.shape[::-1]
        found = None
        if image.size != 0:
            for scaled in numpy.linspace(1.0, 0.4, 20):
                img = cv2.resize(image, dsize=None, fx=scaled, fy=scaled, interpolation=cv2.INTER_AREA)
                resized_ratio = image.shape[1] / img.shape[1]
                if img.shape[0] < template_height or img.shape[1] < template_height:
                    break
                arr = self.match_many(image, template, confidence, threshold)
                if found is not None:
                    numpy.concatenate((found, arr))
                else:
                    found = arr
            found = self.non_max_suppression(found, threshold)
        else:
            if robot_logger:
                info(f'match_many_scaled empty image', html=True)
            else:
                logging.info(f'match_many_scaled empty image')
            found = None
        return found

    def match_many(self, image: numpy.ndarray, template: numpy.ndarray, confidence: float,
                   threshold: float) -> numpy.ndarray:
        """
        Finds all occurrences of the given template on the image, which have a confidence better than the given threshold
        :param confidence:
        :type confidence:
        :param image:
        :type image:
        :param template:
        :type template:
        :param threshold:
        :type threshold:
        :return:
        :rtype:
        """
        img = image.copy()
        template_width, template_height = template.shape[::-1]
        result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        (y_coordinates, x_coordinates) = numpy.where(result > confidence)
        rectangles = []
        for (x, y) in zip(x_coordinates, y_coordinates):
            rectangles.append((x, y, x + template_width, y + template_height))
        rectangles = self.non_max_suppression(numpy.array(rectangles), threshold)

        if rectangles is not None:
            rectangles[:, 2] = rectangles[:, 2] - rectangles[:, 0]
            rectangles[:, 3] = rectangles[:, 3] - rectangles[:, 1]
        return rectangles

    def match_one_scaled(self, image: numpy.ndarray, template: numpy.ndarray):
        """
        Utilises template matching to find the best occurrence of the template in any scale in the image
        :param image: Binary image that should be searched for the template
        :type image: numpy.ndarray
        :param template: Binary image of the template that should be found, needs to be smaller!
        :type template: numpy.ndarray
        :return: Bounding rectangle and confidence of the found occurrence
        :rtype: tuple
        """
        template_width, template_height = template.shape[::-1]
        found = None
        if image.size != 0:
            for scaled in numpy.linspace(1.0, 0.4, 20):
                img = cv2.resize(image, dsize=None, fx=scaled, fy=scaled, interpolation=cv2.INTER_AREA)
                resized_ratio = image.shape[1] / img.shape[1]
                # template needs to be smaller than the image
                if img.shape[0] < template_height or img.shape[1] < template_height:
                    break
                (x, y, w, h), c = self.match_one(img, template)
                if found is None or found[1] < c:
                    cv2.imshow("im", img)
                    found = (
                        (
                        int(x * resized_ratio), int(y * resized_ratio), int(w * resized_ratio), int(h * resized_ratio)),
                        c)
        else:
            if robot_logger:
                info(f'match_one_scaled image of size zero', html=True)
            else:
                logging.info(f'match_one_scaled image of size zero')
        return found

    @staticmethod
    def match_one(image: numpy.ndarray, template: numpy.ndarray, mask: numpy.ndarray | None = None) -> tuple[
                                                                                                           int, int, int, int, float] | None:
        """
        Uses template Matching to find exactly one occurrence of the template
        :param mask:
        :type mask:
        :param image: Gray/Binary Image
        :type image: numpy.ndarray
        :param template: Gray/Binary Image
        :type template: numpy.ndarray
        :return: rectangle x,y,w,h ond confidence
        :rtype: tuple[int,int,int,int,float] | None
        """
        img = image.copy()
        template_width, template_height = template.shape[::-1]
        img_width, img_height = img.shape[::-1]
        if template_width > img_width or template_height > img_height:
            return None
        if mask is not None:
            result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED, mask=mask.copy())
        else:
            result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        result[~numpy.isfinite(result)] = 0
        result[result > 1] = 0
        asdf = numpy.argmax(result)
        yyy = asdf // (result.shape[1])
        xxx = asdf % (result.shape[1])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # If method == TM_SQDIFF or TM_SQDIFF_NORMED -> top_left = min_loc
        top_left = max_loc
        # bottom_right = (top_left[0]+template_width, top_left[1] + template_height)
        # If method == TM_SQDIFF or TM_SQDIFF_NORMED -> confidence = 1-minVal
        confidence = max_val
        return top_left[0], top_left[1], template_width, template_height, confidence

    @staticmethod
    def non_max_suppression(boxes: numpy.ndarray, threshold: float) -> numpy.ndarray | None:
        """
        Finds the rectangle with the largest overlapping area with hin all other overlapping rectangles
        https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
        :param boxes: list of rectangles with their top left and bottom right corner. NOT width or height!
        :type boxes: numpy.ndarray
        :param threshold: Overlapping area in %
        :type threshold: float
        :return: list of rectangles
        :rtype: numpy.ndarray
        """
        # No Rectangles to Merge
        if len(boxes) == 0:
            return None
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        picked = []
        # x
        left = boxes[:, 0]
        # y
        top = boxes[:, 1]
        # x2
        right = boxes[:, 2]
        # y2
        bottom = boxes[:, 3]
        area = (right - left + 1) * (bottom - top + 1)
        indexes = numpy.argsort(bottom)
        while len(indexes) > 0:
            current_index = indexes[len(indexes) - 1]
            picked.append(current_index)
            suppressed = [len(indexes) - 1]
            for position in range(len(indexes) - 1, 0, -1):
                other_index = indexes[position]
                new_left = max(left[current_index], left[other_index])
                new_top = max(top[current_index], top[other_index])
                new_right = min(right[current_index], right[other_index])
                new_bottom = min(bottom[current_index], bottom[other_index])
                width = max(0, new_right - new_left + 1)
                height = max(0, new_bottom - new_top + 1)
                overlap_area = float(width * height) / area[other_index]
                if overlap_area > threshold:
                    suppressed.append(position)

            indexes = numpy.delete(indexes, suppressed)
        return boxes[picked].astype("uint")

    @staticmethod
    def __replicate_boundary(img: numpy.ndarray):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        return (hist[0] + hist[255]) / sum(hist) >= 0.5

    @staticmethod
    def generate_binary_image_4bit(image: numpy.ndarray, replicate=False) -> numpy.ndarray:
        """
        Generates a binary image of the given colour image using Laplacian
        :param replicate:
        :type replicate:
        :param image:
        :type image:
        :return:
        :rtype:
        """
        bin_img = image.copy()
        bin_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
        replicate = TemplateMatcher.__replicate_boundary(bin_img)
        bin_img[bin_img < 32] = 0
        for index in range(1, 7):
            bin_img[(bin_img >= 32 * index) & (bin_img < 32 * (index + 1))] = 32 * (index + 1) - 1
        bin_img[(bin_img >= 223)] = 255
        if replicate:
            bin_img = cv2.copyMakeBorder(bin_img, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT,
                                         value=0)
        else:
            bin_img = cv2.copyMakeBorder(bin_img, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_REPLICATE)
        bin_img = cv2.Laplacian(bin_img, cv2.CV_64F)
        bin_img[abs(bin_img) > 16.0] = 255.0
        bin_img[abs(bin_img) <= 16.0] = 0.0
        bin_img = numpy.uint8(bin_img)
        return bin_img

    @staticmethod
    def generate_binary_image(image: numpy.ndarray, replicate=False) -> numpy.ndarray:
        """
        Generates a binary image of the given colour image using Laplacian
        :param replicate:
        :type replicate:
        :param image:
        :type image:
        :return:
        :rtype:
        """
        bin_img = image.copy()
        bin_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
        if not TemplateMatcher.__replicate_boundary(bin_img):
            bin_img = cv2.copyMakeBorder(bin_img, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT,
                                         value=0)
        else:
            bin_img = cv2.copyMakeBorder(bin_img, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_REPLICATE)
        bin_img = cv2.Laplacian(bin_img, cv2.CV_64F)
        bin_img[abs(bin_img) > 0.0] = 255.0
        bin_img = numpy.uint8(bin_img)
        return bin_img


if __name__ == '__main__':
    pass
