from enum import Enum


class ButtonState(Enum):
    """
    States for Buttons and Text fields of the GUI
    """
    INVISIBLE = 0
    DEACTIVATED = 1
    ACTIVATED_INVISIBLE = 2
    ACTIVATED = 3
    PRESSED_INVISIBLE = 4
    PRESSED = 5
    TOGGLED_ON_INVISIBLE = 6
    TOGGLED_ON = 7
    TOGGLED_OFF_INVISIBLE = 8
    TOGGLED_OFF = 9
    HOVER_INVISIBLE = 10
    HOVER = 11
    FOCUS_INVISIBLE = 12
    FOCUS = 13
    SELECTED = 14
