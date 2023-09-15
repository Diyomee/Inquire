import xml.etree.ElementTree as ET
import pathlib
from inquire.detection.ButtonState import ButtonState
from inquire.helper.Utilities import Utilities
from typing import Tuple, Set

TRUE_STRINGS = {"true", "1", "y", "yes", "ja", "yeah", "yup", "certainly", "+", "ok", "qui", "c'est bon"}


class XMLParser:
    """
    Parser to load and save xml config files
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self):
        pass

    @staticmethod
    def parse_symbol_selection_xml(xml_path: pathlib.Path) -> tuple[set[str | None], set[str | None]]:
        """
        Parses the given xml pathlib object.
        the xml document has to have the following structure
        <symbols>
            <symbol>
                <path>path to png
                <ignore> whether this symbol should be ignored or not
        :param xml_path: path to the xml file
        :type xml_path: pathlib.Path
        :return: two sets of relative paths, the first specifies the allowed symbols, the second specifies the ignored symbols
        :rtype: dict
        :raises (TypeError, FileExistsError)
        """
        set_positive = set()
        set_negative = set()
        if xml_path.is_file():
            if xml_path.suffix == ".xml":
                tree = ET.parse(str(xml_path))
                it = tree.iter("symbols")
                for symbols in it:
                    it2 = symbols.iterfind("symbol")
                    for symbol in it2:
                        path = symbol.find("path").text
                        ignore = symbol.find("ignore").text.lower() in TRUE_STRINGS
                        if ignore:
                            set_negative.add(path)
                        else:
                            set_positive.add(path)
                return set_positive, set_negative
            else:
                raise TypeError(f'File {xml_path} is not of type .xml')
        else:
            raise FileExistsError(f'File {xml_path} does not exist')

    @staticmethod
    def parse_svg_xml(xml_path: pathlib.Path) -> dict:
        """
        Parses the given xml pathlib object.
        the xml document has to have the following structure
        <images>
            <image id="name">
                <path>path to svg
                <width> render width
                <height> render height
        :param xml_path: path to the xml file
        :type xml_path: pathlib.Path
        :return: Dictionary of svg to png render settings key:path value:[(name,width,height),...]
        :rtype: dict
        :raises (TypeError, FileExistsError)
        """
        dic = {}
        if xml_path.is_file():
            if xml_path.suffix == ".xml":
                tree = ET.parse(str(xml_path))
                it = tree.iter("images")
                for images in it:
                    it2 = images.iterfind("image")
                    for image in it2:
                        name = image.get("id")
                        path = image.find("path").text
                        width = int(image.find("width").text)
                        height = int(image.find("height").text)
                        if path not in dic:
                            dic[path] = []
                        dic[path].append((name, width, height))
                return dic
            else:
                raise TypeError(f'File {xml_path} is not of type .xml')
        else:
            raise FileExistsError(f'File {xml_path} does not exist')

    @staticmethod
    def parse_state_information(xml_path: pathlib.Path) -> dict[
        str, dict[ButtonState, list[list[tuple[int, int, int]]]]]:
        dic: dict[str, dict[ButtonState, list[list[tuple[int, int, int]]]]] = {}
        if xml_path.is_file():
            if xml_path.suffix == ".xml":
                tree = ET.parse(str(xml_path))
                it = tree.iter("states")
                for states in it:
                    text = states.get("text").lower() in TRUE_STRINGS
                    it2 = states.iterfind("state")
                    it4 = states.iterfind("symbol")
                    symbols = []
                    for symbol in it4:
                        symbols.append(symbol.text)
                    for state in it2:
                        it3 = state.iterfind("colour")
                        name = ButtonState[state.get("name")]
                        colours = []
                        for hex_colour in it3:
                            colours.append(Utilities.hex_rgb_to_bgr(hex_colour.text))
                        if text:
                            if "Text" in dic:
                                if name in dic["Text"]:
                                    dic["Text"][name].append(colours)
                                else:
                                    dic["Text"][name] = [colours]
                            else:
                                dic["Text"] = {name: [colours]}
                        for symbol in symbols:
                            if symbol in dic:
                                if name in dic[symbol]:
                                    dic[symbol][name].append(colours)
                                else:
                                    dic[symbol][name] = [colours]
                            else:
                                dic[symbol] = {name: [colours]}
                return dic
            else:
                raise TypeError(f'File {xml_path} is not of type .xml')
        else:
            raise FileExistsError(f'File {xml_path} does not exist')


if __name__ == '__main__':
    pass
