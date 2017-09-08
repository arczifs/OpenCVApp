TEMPLATE = app

QT += qml quick
CONFIG += c++11

HEADERS += \
    QmlComponents/CameraItem.h \
    Utils/QPropertyWrapper.h

SOURCES += main.cpp \
    QmlComponents/CameraItem.cpp

RESOURCES += qml.qrc \
    assets.qrc \
    cascadeclassifiers.qrc

INCLUDEPATH += C:/OpenCV3.1/builds/install/include
INCLUDEPATH += C:/opencv_3.3/opencv/dep/tbb2017_20170604oss/include

LIBS += -LC:/opencv_3.3/opencv/dep/tbb2017_20170604oss/lib/intel64/vc14
LIBS += -LC:/OpenCV3.1/builds/install/x64/vc14/lib
LIBS += -LC:/OpenCV3.1/builds/install/x64/vc14/bin

CONFIG(debug, debug|release)
{
    LIBS += -lopencv_calib3d310 \
            -lopencv_core310 \
            -lopencv_highgui310 \
            -lopencv_imgproc310 \
            -lopencv_features2d310 \
            -lopencv_flann310 \
            -lopencv_ml310 \
            -lopencv_objdetect310 \
            -lopencv_photo310 \
            -lopencv_stitching310 \
            -lopencv_superres310 \
            -lopencv_ts310 \
            -lopencv_video310 \
            -lopencv_videostab310 \
            -lopencv_videoio310 \
            -lopencv_imgcodecs310 \
            -ltbb \
            -lopengl32
}

CONFIG(release, debug|release)
{
    LIBS += -lopencv_calib3d310d \
            -lopencv_core310d \
            -lopencv_highgui310d \
            -lopencv_imgproc310d \
            -lopencv_features2d310d \
            -lopencv_flann310d \
            -lopencv_ml310d \
            -lopencv_objdetect310d \
            -lopencv_photo310d \
            -lopencv_stitching310d \
            -lopencv_superres310d \
            -lopencv_ts310d \
            -lopencv_video310d \
            -lopencv_videostab310d \
            -lopencv_videoio310d \
            -lopencv_imgcodecs310d \
            -ltbb_debug \
            -lopengl32
}

DEFINES += QT_DEPRECATED_WARNINGS

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

