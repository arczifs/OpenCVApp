#include "CameraItem.h"

#include <assert.h>
#include <thread>
#include <queue>

#include <QSGGeometryNode>
#include <QSGGeometry>
#include <QSGSimpleTextureNode>
#include <QQuickWindow>
#include <QFileInfo>
#include <QDir>
#include <QTemporaryFile>

#include "tbb/pipeline.h"

const double kGradientThreshold = 50.0;
const int kWeightBlurSize = 5;
const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.97;
const bool kEnableWeight = true;
const bool kPlotVectorField = false;
const float kWeightDivisor = 1.0;
const int kFastEyeWidth = 50;
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 25;
const int kEyePercentWidth = 35;
const bool kSmoothFaceImage = false;
const float kSmoothFaceFactor = 0.005;

using namespace cv;

CameraItem::CameraItem()
    : frameRate(this, &CameraItem::frameRateChanged, 15)
    , videoWidth(this, &CameraItem::videoWidthChanged, 640)
    , videoHeight(this, &CameraItem::videoHeightChanged, 480)
    , cameraInterface(this, &CameraItem::cameraInterfaceChanged, 0)
    , firstCascadeSource(this, &CameraItem::firstCascadeSourceChanged, ":/cascades/haarcascade_frontalface_alt.xml")
    , secondCascadeSource(this, &CameraItem::secondCascadeSourceChanged, ":/cascades/haarcascade_eye.xml")
    , m_Done(false)
    , m_FacePos()
{
    ocl::setUseOpenCL(true);

    m_Capture.set(CV_CAP_PROP_FPS, frameRate);
    m_Capture.set(CV_CAP_PROP_FRAME_WIDTH, videoWidth);
    m_Capture.set(CV_CAP_PROP_FRAME_HEIGHT, videoHeight);

    bool connected = connect(this, &CameraItem::capturedImage, this, &CameraItem::setImage, Qt::QueuedConnection);
    assert(connected);

    Q_UNUSED(connected);

    _init();

    setFlag(ItemHasContents, true);
}

CameraItem::~CameraItem()
{
    m_Done = true;
    if (m_PipelineRunner.joinable())
    {
        m_PipelineRunner.join();
    }
    m_Capture.release();

    ProcessingChainData* pData = nullptr;
    while(m_GuiQueue.try_pop(pData))
    {
        delete pData;
        pData = nullptr;
    }
}

QSGNode* CameraItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *)
{
    if (m_Image.empty())
    {
        qDebug()<<"Can't read image";
        return nullptr;
    }

    QSGSimpleTextureNode *resultTexture = static_cast<QSGSimpleTextureNode *>(oldNode);
    if (!resultTexture)
    {
        resultTexture = new QSGSimpleTextureNode();
    }

    GLuint* texture = new GLuint[1];
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, texture[0]);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_Image.cols, m_Image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, m_Image.ptr());

    QSGTexture *t = window()->createTextureFromId(static_cast<unsigned int>(*texture), QSize(videoWidth, videoHeight));

    delete[] texture;

    if (!t)
    {
        qDebug()<<"Can't create texture";
    }
    else
    {
        resultTexture->setRect(boundingRect());
        resultTexture->setTexture(t);
    }

    return resultTexture;
}

void CameraItem::_init()
{
    if(!m_Capture.open(cameraInterface))
    {
        qDebug()<<"Can't open camera interface: "<<cameraInterface;
        assert(true);
    }

    if (!_loadCascade(m_FirstCascade, firstCascadeSource) || !_loadCascade(m_SecondCascade, secondCascadeSource))
    {
        qDebug()<<"Failed to load cascades";
        return;
    }


    m_GuiQueue.set_capacity(2);
    m_PipelineRunner = std::thread(&CameraItem::_detectAndDrawTBB,
                                             this,
                                             std::ref(m_Capture),
                                             std::ref(m_GuiQueue),
                                             std::ref(m_FirstCascade),
                                             std::ref(m_SecondCascade),
                                             1,
                                             true);
}

bool CameraItem::_loadCascade(CameraItem::Cascade &cascade, QString url)
{
    QFile file(url);
    if (!file.open(QIODevice::ReadOnly))
    {
        qDebug()<<"Error openning file "<<url;
        return false;
    }
    QTemporaryFile output;
    if (output.open())
    {
        output.write(file.readAll());
        output.close();
    }
    file.close();

    if (!cascade.load(output.fileName().toStdString()))
    {
        qDebug()<<"Can't load cascade: "<<url;
        return false;
    }

    return true;
}

void CameraItem::_detectAndDrawTBB(VideoCapture &capture,
                                  CameraItem::Concurent_queue &guiQueue,
                                  CameraItem::Cascade &cascade,
                                  CameraItem::Cascade &nestedCascade,
                                  double scale,
                                  bool tryFlip)
{
    const static Scalar colors[] =
        {
            Scalar(255,0,0),
            Scalar(255,128,0),
            Scalar(255,255,0),
            Scalar(0,255,0),
            Scalar(0,128,255),
            Scalar(0,255,255),
            Scalar(0,0,255),
            Scalar(255,0,255)
        };

    tbb::parallel_pipeline(7,
                           tbb::make_filter<void, ProcessingChainData *>(tbb::filter::serial_in_order,
                                                                         [&](tbb::flow_control& fc)->ProcessingChainData*
    {

        auto pData = new ProcessingChainData();
        capture >> pData->image;
        if (m_Done || pData->image.empty())
        {
            delete pData;
            m_Done = true;
            fc.stop();
            return 0;
        }

        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, ProcessingChainData *>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData *pData)->ProcessingChainData*
    {
        cvtColor(pData->image, pData->gray, COLOR_BGR2GRAY);
        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, ProcessingChainData *>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData* pData)->ProcessingChainData*
    {
        double fx = 1 / scale;
        resize(pData->gray, pData->smallImg, Size(), fx, fx, INTER_LINEAR);
        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData* pData)->ProcessingChainData*
    {
        equalizeHist(pData->smallImg, pData->smallImg);
        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData* pData)->ProcessingChainData*
    {
        cascade.detectMultiScale(pData->smallImg, pData->firstCascadeObjects,
                                 1.05, 3, 0 | CASCADE_SCALE_IMAGE,
                                 Size(150, 150));

//        if (tryFlip)
//        {
//            flip(pData->image, pData->image, 1);
//        }

        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData* pData)->ProcessingChainData*
    {
        if (!pData->firstCascadeObjects.empty())
        {
            for (size_t i = 0; i < pData->firstCascadeObjects.size(); ++i)
            {
                m_FacePos = pData->firstCascadeObjects[i];

                Rect smoothedRect = _getSmoothed(m_FacePos);
                Scalar color = colors[i%8];
                Rect faceRect = {cvPoint(cvRound(smoothedRect.x * scale), cvRound(smoothedRect.y * scale)),
                                     cvPoint(cvRound((smoothedRect.x + smoothedRect.width) * scale),
                                             cvRound((smoothedRect.y + smoothedRect.height) * scale))};

                rectangle(pData->image, faceRect, color, 3, 8, 0);

                // draw eyes
                int eye_region_width = faceRect.width * (kEyePercentWidth/100.0);
                int eye_region_height = faceRect.width * (kEyePercentHeight/100.0);
                int eye_region_top = faceRect.height * (kEyePercentTop/100.0);
                Rect leftEyeRegion(smoothedRect.x + faceRect.width * (kEyePercentSide/100.0),
                                       smoothedRect.y + eye_region_top, eye_region_width, eye_region_height);
                Rect rightEyeRegion(smoothedRect.x + faceRect.width - eye_region_width - faceRect.width * (kEyePercentSide/100.0),
                                        smoothedRect.y + eye_region_top, eye_region_width, eye_region_height);

                rectangle(pData->image, leftEyeRegion, color, 3, 8, 0);
                rectangle(pData->image, rightEyeRegion, color, 3, 8, 0);

                Mat faceROI = pData->image(faceRect);
                Point leftPupil = _getSmoothed(_findEyeCenter(pData->image, leftEyeRegion));
//                qDebug()<<"leftPupil: "<<leftPupil.x<<", "<<leftPupil.y;
                circle(pData->image, cvPoint(leftPupil.x + faceRect.x, leftPupil.y + faceRect.y + leftEyeRegion.height / 2), 3, 1234);
            }
        }
        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, void>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData *pData)
    {
        if (!m_Done)
        {
            try
            {
                emit capturedImage();
                guiQueue.emplace(pData);
            }
            catch(...)
            {
                qDebug()<<"Pipeline caught an exception on the queue";
                m_Done = true;
            }
        }
    }
    )
    );
}

Rect CameraItem::_getSmoothed(const Rect& point)
{
    static constexpr int listSize = 5;
    static Rect list[listSize] = {Rect()};
    static short pos = 0;

    if (pos == listSize) pos = 0;
    list[pos] = point;
    ++pos;

    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    for (int i = 0; i < listSize; ++i)
    {
        x += list[i].x;
        y += list[i].y;
        w += list[i].width;
        h += list[i].height;
    }

    return Rect(x / listSize, y / listSize, w / listSize, h / listSize);
}

Point CameraItem::_getSmoothed(const Point& point)
{
    static constexpr int listSize = 10;
    static Point list[listSize] = {Point()};
    static short pos = 0;

    if (pos == listSize) pos = 0;
    list[pos] = point;
    ++pos;

    int x = 0;
    int y = 0;
    for (int i = 0; i < listSize; ++i)
    {
        x += list[i].x;
        y += list[i].y;
    }

    return Point(x / listSize, y / listSize);
}

Point CameraItem::_findEyeCenter(Mat face, Rect eye)
{
    Mat eyeROIUnscaled = face(eye);
    Mat eyeROI;

    _scaleToFastSize(eyeROIUnscaled, eyeROI);

    Mat gradientX = _computeMatXGradient(eyeROI);
    Mat gradientY = _computeMatXGradient(eyeROI.t()).t();

    Mat mags = _getMatrixMagnitude(gradientX, gradientY);

    // compute dynamic threshold
    Scalar stdMagnGrad, meanMagnGrad;
    meanStdDev(mags, meanMagnGrad, stdMagnGrad);
    // ?? square root?
    double stdDev = stdMagnGrad[0] / std::sqrt(mags.rows*mags.cols);
    double dynamicThreshold = kGradientThreshold * stdDev + meanMagnGrad[0];

    // normalize
    for (int y = 0; y <eyeROI.rows; ++y) {
        double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        const double *Mr = mags.ptr<double>(y);
        for (int x = 0; x <eyeROI.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = Mr[x];
            if (magnitude > dynamicThreshold) {
                Xr[x] = gX/magnitude;
                Yr[x] = gY/magnitude;
            } else {
                Xr[x] = 0.0;
                Yr[x] = 0.0;
            }
        }
    }

    //-- Create a blurred and inverted image for weighting
    Mat weight;
    GaussianBlur(eyeROI, weight, Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
    for (int y = 0; y < weight.rows; ++y) {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for (int x = 0; x < weight.cols; ++x) {
            row[x] = (255 - row[x]);
        }
    }

    Mat outSum = Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);

    for (int y = 0; y < weight.rows; ++y) {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        for (int x = 0; x < weight.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            if (gX == 0.0 && gY == 0.0) {
                continue;
            }
            _testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
        }
    }

    // scale all the values down, basically averaging them
    double numGradients = (weight.rows*weight.cols);
    Mat out;
    outSum.convertTo(out, CV_32F,1.0/numGradients);
    //imshow(debugWindow,out);
    //-- Find the maximum point
    Point maxP;
    double maxVal;
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
    //-- Flood fill the edges
    if(kEnablePostProcess) {
        Mat floodClone;
        //double floodThresh = computeDynamicThreshold(out, 1.5);
        double floodThresh = maxVal * kPostProcessThreshold;
        threshold(out, floodClone, floodThresh, 0.0f, THRESH_TOZERO);
        if(kPlotVectorField) {
            //plotVecField(gradientX, gradientY, floodClone);
            //                   imwrite("eyeFrame.png",eyeROIUnscaled);
        }
        Mat mask = _floodKillEdges(floodClone);
        // redo max
        minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
    }
    return _unscalePoint(maxP, eye);
}

Mat CameraItem::_floodKillEdges(Mat &mat) {
    rectangle(mat,Rect(0,0,mat.cols,mat.rows),255);

    Mat mask(mat.rows, mat.cols, CV_8U, 255);
    std::queue<Point> toDo;
    toDo.push(Point(0,0));
    while (!toDo.empty()) {
        Point p = toDo.front();
        toDo.pop();
        if (mat.at<float>(p) == 0.0f) {
            continue;
        }
        // add in every direction
        Point np(p.x + 1, p.y); // right
        if (_floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x - 1; np.y = p.y; // left
        if (_floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x; np.y = p.y + 1; // down
        if (_floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x; np.y = p.y - 1; // up
        if (_floodShouldPushPoint(np, mat)) toDo.push(np);
        // kill it
        mat.at<float>(p) = 0.0f;
        mask.at<uchar>(p) = 0;
    }
    return mask;
}

Point CameraItem::_unscalePoint(Point p, Rect origSize)
{
    float ratio = (((float)kFastEyeWidth)/origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return Point(x,y);
}

bool CameraItem::_floodShouldPushPoint(const Point &np, const Mat &mat)
{
    return _inMat(np, mat.rows, mat.cols);
}

bool CameraItem::_inMat(Point p,int rows,int cols)
{
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

void CameraItem::_testPossibleCentersFormula(int x, int y, const Mat &weight,double gx, double gy, Mat &out) {
    // for all possible centers
    for (int cy = 0; cy < out.rows; ++cy) {
        double *Or = out.ptr<double>(cy);
        const unsigned char *Wr = weight.ptr<unsigned char>(cy);
        for (int cx = 0; cx < out.cols; ++cx) {
            if (x == cx && y == cy) {
                continue;
            }
            // create a vector from the possible center to the gradient origin
            double dx = x - cx;
            double dy = y - cy;
            // normalize d
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;
            double dotProduct = dx*gx + dy*gy;
            dotProduct = std::max(0.0,dotProduct);
            // square and multiply by the weight
            if (kEnableWeight) {
                Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
            } else {
                Or[cx] += dotProduct * dotProduct;
            }
        }
    }
}

Mat CameraItem::_computeMatXGradient(const Mat& mat)
{
    Mat out(mat.rows, mat.cols, CV_64F);

    for (int y = 0; y < mat.rows; ++y)
    {
        const uchar *Mr = mat.ptr<uchar>(y);
        double *Or = out.ptr<double>(y);

        Or[0] = Mr[1] - Mr[0];
        for (int x = 1; x < mat.cols - 1; ++x)
        {
            Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
        }

        Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
    }

    return out;
}

Mat CameraItem::_getMatrixMagnitude(const Mat &matX, const Mat &matY)
{
    Mat mags(matX.rows, matX.cols, CV_64F);

    for (int y = 0; y < matX.rows; ++y)
    {
        const double *Xr = matX.ptr<double>(y);
        const double *Yr = matY.ptr<double>(y);
        double *Mr = mags.ptr<double>(y);

        for (int x = 0; x < matX.cols; ++x)
        {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = sqrt((gX * gX) + (gY * gY));
            Mr[x] = magnitude;
        }
    }
    return mags;
}

void CameraItem::_scaleToFastSize(const Mat &src,Mat &dst)
{
resize(src, dst, Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

void CameraItem::setImage()
{
    ProcessingChainData* pData = nullptr;
    if (m_GuiQueue.try_pop(pData))
    {
        m_Image = pData->image;
        delete pData;
    }

    update();
}

