import numpy
import cv2
from inquire.detection.RecognizedElement import RecognizedElement

from inquire.detection.ButtonState import ButtonState


class Symbol(RecognizedElement):
    """
    Symbol Class. It's used to store the information about the found symbol, like it's state,...
    """

    def __init__(self, symbol: str,
                 symbol_path: str,
                 image: numpy.ndarray,
                 mask: numpy.ndarray | None = None,
                 binary_image: numpy.ndarray | None = None,
                 rectangle: tuple = (0, 0, 0, 0),
                 confidence: float | None = None,
                 colours: list | None = None,
                 state: ButtonState | None = None):
        """
        Constructor of the symbol class
        :param symbol: Name of the Symbol
        :type symbol: str | None
        :param symbol_path: Path to the file which represents the symbol
        :type symbol_path: str | None
        :param image: Colour Image of the found symbol
        :type image: numpy.ndarray
        :param binary_image: Optional binary image of the symbol
        :type binary_image: numpy.ndarray | None
        :param rectangle: Bounding rectangle position of the image in the application screen.
        :type rectangle: tuple
        :param confidence: Confidence of the detected symbol. Should go from 0 to 1
        :type confidence: float | None
        :param colours: List of colours of the found symbol. In decreasing order of occurred amount
        :type colours: list | None
        :param state: State of the Symbol. Like ACTIVATED, DEACTIVATED, PRESSED, FOCUS, TOGGLE_ON, TOGGLE_OFF
        :type state: ButtonState | None
        """
        super().__init__(image=image,
                         rectangle=rectangle,
                         confidence=confidence,
                         colours=colours,
                         state=state,
                         binary_image=binary_image)
        self.symbol = symbol
        self.mask = mask
        self.symbol_path = symbol_path

    def __str__(self):
        if self.rectangle != (0, 0, 0, 0):
            return f'({self.symbol}, {self.symbol_path}, rec:{self.rectangle})'
        else:
            return f'({self.symbol}, {self.symbol_path})'

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    pass
