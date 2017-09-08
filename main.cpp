#include <QGuiApplication>
#include <QQmlApplicationEngine>

#include "QmlComponents/CameraItem.h"

int main(int argc, char *argv[])
{
    qmlRegisterType<CameraItem>("com.qmlVideoComponents", 1, 0, "CameraItem");

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
