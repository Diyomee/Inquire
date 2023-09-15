*** Settings ***
Library  loggers//VideoLogger.py       ${VIDEO DIR}
Library  loggers//ImageLogger.py       ${SCREENSHOT DIR}
Library  helper//Comparator.py


*** Variables ***
${CAPTURE REGION}
*** Keywords ***
Saves The Found Capture Region As A Global Variable
    ${region}=     VideoLogger.get_capture_region
    Set Global Variable                         ${CAPTURE REGION}  ${region}



Log Video
    VideoLogger.Set Capture Region    ${CAPTURE REGION}
    VideoLogger.Set Filename To    ${SUITE_NAME}  ${TEST_NAME}
    Start Process For Video Recoding

Log Screenshot
    ImageLogger.Set Capture Region    ${CAPTURE REGION}
    ImageLogger.Set Filename To    ${SUITE_NAME}  ${TEST_NAME}

Start Logging
    Saves The Found Capture Region As A Global Variable
    Log Video
    Log Screenshot

Close Log Ans App
    Stop Recording
    Sleep    1s


