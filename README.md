# Python Modul
## Setup
Install [PaddlePaddle according to their guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/windows-pip_en.html).
remove the following installed packages
```
python -m pip install paddlepaddle==2.5.0 -i https://mirror.baidu.com/pypi/simple
pip uninstall opencv-python -y
pip uninstall opencv-contrib-python -y
```
And reinstall the newest version
```
pip install opencv-contrib-python==4.7.0.72
pip install mss
pip install robotframework
pip install robotframework-dependencylibrary
pip install cairosvg
pip install Levenshtein
pip install mouse
```
Add the directory cv2 of opencv to PYTHONPATH
## How To Use
To use the Package, you need to import the GUITools.robot into your robot file for the tests.
```
Resource          src/GUITools.robot
```

## Robot Frame Work
TODO
### Library Scope
You are able to define the scope in which the library is loaded.
This can be for each test case, the whole test suite,
or globally for all test suites.
You have to insert one of the following line after the class definition.
```
ROBOT_LIBRARY_SCOPE = 'TEST'
ROBOT_LIBRARY_SCOPE = 'SUITE'
ROBOT_LIBRARY_SCOPE = 'GLOBAL'
```
for more Information see [the user guide](https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#creating-test-library-class-or-module "Robots Library Scope")


