class OSException(Exception):
    """
    Operating System Exception
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = '0.1'
    ROBOT_AUTO_KEYWORDS = False
    __version__ = '0.1'

    def __init__(self, message: str) -> None:
        """
        Constructor
        :param message: Exception Message
        :type message: str
        """
        super().__init__(f'Wrong Operating System: {message}')
