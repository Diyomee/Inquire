import cairosvg
import pathlib
from robot.api.deco import keyword
from inquire.helper.XMLParser import XMLParser


class SvgToPngGenerator:
    """
    Class to handle the generation of pngs from svg files.

    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self, dir_t: str | None = None):
        """
        Constructor of the svg to png generator.
        :param dir_t: abs path to the current working directory
        :type dir_t: str | None
        """
        self.working_dir = pathlib.Path(dir_t)

    def find_all_pngs_in_working_dir(self, sub_dirs: bool = False) -> list[str]:
        """
        Searches the current working directory for png files and returns the paths of all found .png files.
        When sub_dirs is flagged as True, it will also search all subdirectories.
        :param sub_dirs: Flag whether to include subdirectories for the .png search
        :type sub_dirs: bool
        :return: paths to pngs files
        :rtype: list[str]
        """
        ans = []
        if sub_dirs:
            paths = self.working_dir.glob('**/*.png')
            for i in paths:
                ans.append(str(i.resolve()))
            return ans
        else:
            paths = self.working_dir.glob('*.png')
            ans = []
            for i in paths:
                ans.append(str(i.resolve()))
            return ans

    def find_all_svgs_in_working_dir(self, sub_dirs: bool = False) -> list[str]:
        """
        Searches the current working directory for svg files and returns the paths of all found .svg files.
        When sub_dirs is flagged as True, it will also search all subdirectories.
        :param sub_dirs: Flag whether to include subdirectories for the .svg search
        :type sub_dirs: bool
        :return: paths to the found .svg files
        :rtype: str
        """
        if sub_dirs:
            paths = self.working_dir.glob('**/*.svg')
        else:
            paths = self.working_dir.glob('*.svg')
        self.working_dir.resolve()
        ans = []
        for i in paths:
            ans.append(str(i.resolve()))
        return ans

    @staticmethod
    def generate_png(path_to_svg: str, filename: str | None = None, width: int | None = None,
                     height: int | None = None) -> str:
        """
        Renders a png file with the same name as the given svg file in the same directory and returns its path.
        :param filename: name of the png file
        :type filename: str
        :param height: render height of the png
        :type height: int | None
        :param width: render width of the png
        :type width: int | None
        :param path_to_svg: absolute path to the .svg file
        :type path_to_svg: str
        :return: path to the generated .png file
        :rtype: str
        :raises ValueError
        """
        path = pathlib.Path(path_to_svg)
        if path.exists() and path.is_absolute():
            if path_to_svg[-4:] == ".svg":
                if filename is not None:
                    filename += ".png"
                    path_to_png = str(path.parent.joinpath(filename).resolve())
                else:
                    path_to_png = path_to_svg[:-4] + ".png"
                cairosvg.svg2png(url=path_to_svg, write_to=path_to_png, background_color="000000", negate_colors=True,
                                 output_width=width, output_height=height)
                return path_to_png
            else:
                raise ValueError(f'Filetype must be .svg not {path_to_svg}')
        else:
            raise FileNotFoundError(f'File {path_to_svg} does not exists')

    @keyword
    def generate_pngs(self, sub_dirs: bool = True) -> list[str]:
        """
        For each .svg file that can be found in the current saved working directory (working_dir) ,and it's subdirectory
        , it generates/renders a .png with the same file name. The paths to all generated files is then returned
        for further usage.
        :param sub_dirs: Flag whether to include subdirectories for the .svg search
        :type sub_dirs: bool
        :return: paths of the generated .png files
        :rtype: list[str]
        """
        svg_files = self.find_all_svgs_in_working_dir(sub_dirs=sub_dirs)
        png_files = []
        for path in svg_files:
            try:
                p = self.generate_png(path)
            except ValueError:
                continue
            else:
                png_files.append(p)
                self.png_list.add(p)
        return png_files

    @keyword
    def generate_pngs_from_dir(self, dir_t: str, sub_dirs: bool = True) -> list[str]:
        """
        For each .svg file that can be found in the given directory ,and it's subdirectory, it generates/renders
         a .png with the same file name. The paths to all generated files is then returned for further usage.
        :param sub_dirs: Flag whether to include subdirectories for the .svg search
        :type sub_dirs: bool
        :param dir_t: Absolute Path to the directory of .svg files
        :type dir_t: str
        :return: paths of the generated .png files
        :rtype: list[str]
        """
        temp = self.working_dir
        self.working_dir = pathlib.Path(dir_t)
        if self.working_dir.exists():
            png_files = self.generate_pngs(sub_dirs=sub_dirs)
        else:
            self.working_dir = temp
            raise FileExistsError(f'given Path {temp} does not exist')
        self.working_dir = temp
        return png_files

    @keyword
    def set_working_directory_to(self, abs_path: str):
        temp = pathlib.Path(abs_path)
        temp.resolve()
        if temp.exists():
            self.working_dir = temp
        else:
            raise FileExistsError(f'given Path {temp} does not exist')

    @keyword
    def generate_pngs_from_xml(self):
        """
        Generates pngs from svgs using the information from the xml document.
        The xml document has to be in the working directory ,or it's parent
        the xml document has to have the following structure
        <images>
            <image id="name">
                <path>path to svg
                <width> render width
                <height> render height
        The path has to be a relative path from the position of the xml file to the svg file
        :return:
        :rtype:
        """
        xml_paths = self.working_dir.parent.glob('**/*.xml')
        png_paths = []
        for xml_file in xml_paths:
            svg_render_information = XMLParser.parse_svg_xml(xml_file)
            for key, values in svg_render_information.items():
                for value in values:
                    file_path = xml_file.parent
                    file_path = file_path.joinpath(key).resolve()
                    if file_path.is_file():
                        png_path = self.generate_png(str(file_path), value[0], value[1], value[2])
                        png_paths.append(png_path)
        return png_paths

    @keyword
    def generate_pngs_from_xml_in_dir(self, dir_t: str) -> list[str]:
        """
        Generates pngs from svgs using the information from the xml document.
        the xml document has to have the following structure
        <images>
            <image id="name">
                <path>path to svg
                <width> render width
                <height> render height
        :param dir_t:
        :type dir_t:
        :param sub_dirs:
        :type sub_dirs:
        :return:
        :rtype:
        """
        temp = self.working_dir
        self.working_dir = pathlib.Path(dir_t)
        if self.working_dir.exists():
            png_files = self.generate_pngs_from_xml()
        else:
            self.working_dir = temp
            raise FileExistsError(f'given Path {dir_t} does not exist')
        self.working_dir = temp
        return png_files


if __name__ == '__main__':
    pass
