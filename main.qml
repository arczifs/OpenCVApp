import QtQuick 2.6
import QtQuick.Window 2.2

import com.qmlVideoComponents 1.0

Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    Rectangle {
        id: root
        color: "#1e1e1e"
        anchors.fill: parent

        CameraItem {
            anchors.fill: parent
        }
    }
}
