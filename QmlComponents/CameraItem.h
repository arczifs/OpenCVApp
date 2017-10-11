#ifndef CAMERAITEM_H
#define CAMERAITEM_H

#include <QQuickItem>

#include <vector>
#include <atomic>
#include <thread>

#include "Utils/QPropertyWrapper.h"

#include "opencv2/opencv.hpp"

#include "tbb/concurrent_queue.h"
#include "opencv2/core/ocl.hpp"

class CameraItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(int frameRate READ frameRate WRITE frameRate NOTIFY frameRateChanged)
    Q_PROPERTY(int videoWidth READ videoWidth WRITE videoWidth NOTIFY videoWidthChanged)
    Q_PROPERTY(int videoHeight READ videoHeight WRITE videoHeight NOTIFY videoHeightChanged)
    Q_PROPERTY(int cameraInterface READ cameraInterface WRITE cameraInterface NOTIFY cameraInterfaceChanged)
    Q_PROPERTY(QString firstCascadeSource READ firstCascadeSource WRITE firstCascadeSource NOTIFY firstCascadeSourceChanged)
    Q_PROPERTY(QString secondCascadeSource READ secondCascadeSource WRITE secondCascadeSource NOTIFY secondCascadeSourceChanged)

    struct ProcessingChainData {
        cv::Mat image;
        std::vector<cv::Rect> firstCascadeObjects, secondCascadeObjects;
        cv::Mat gray, smallImg;
    };

    using Concurent_queue = tbb::concurrent_bounded_queue<ProcessingChainData* >;
    using Cascade = cv::CascadeClassifier;
public:
    explicit CameraItem();
    ~CameraItem();

    QPropertyWrapper<int> frameRate;
    QPropertyWrapper<int> videoWidth;
    QPropertyWrapper<int> videoHeight;
    QPropertyWrapper<int> cameraInterface;
    QPropertyWrapper<QString> firstCascadeSource;
    QPropertyWrapper<QString> secondCascadeSource;

signals:
    void frameRateChanged();
    void videoWidthChanged();
    void videoHeightChanged();
    void cameraInterfaceChanged();
    void firstCascadeSourceChanged();
    void secondCascadeSourceChanged();
    void capturedImage();

    // QQuickItem interface
protected:
    virtual QSGNode *updatePaintNode(QSGNode *, UpdatePaintNodeData *) override;
    
private:
    void _init();
    bool _loadCascade(Cascade& cascade, QString url);
    void _detectAndDrawTBB(cv::VideoCapture& m_Capture,
                          Concurent_queue& m_GuiQueue,
                          Cascade& cascade,
                          Cascade& nestedCascade,
                          double scale, bool tryFlip);
    void setImage();
    cv::Rect _getSmoothed(const cv::Rect &point);

    // gradient algorithms
    cv::Point _findEyeCenter(cv::Mat face, cv::Rect eye);
    cv::Mat _floodKillEdges(cv::Mat &mat);
    cv::Point _unscalePoint(cv::Point p, cv::Rect origSize);
    bool _floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat);
    bool _inMat(cv::Point p, int rows, int cols);
    void _testPossibleCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out);
    cv::Mat _computeMatXGradient(const cv::Mat &mat);
    cv::Mat _getMatrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
    void _scaleToFastSize(const cv::Mat &src, cv::Mat &dst);
    cv::Point _getSmoothed(const cv::Point &point);

    cv::VideoCapture m_Capture;
    std::atomic<bool> m_Done;
    Cascade m_FirstCascade;
    Cascade m_SecondCascade;
    QString m_FirstCascadeSource;
    QString m_SecondCascadeSource;
    cv::Mat m_Image;
    std::thread m_PipelineRunner;
    Concurent_queue m_GuiQueue;
    cv::Rect m_FacePos;
};

#endif // CAMERAITEM_H
