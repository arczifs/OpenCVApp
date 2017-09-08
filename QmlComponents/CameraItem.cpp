#include "CameraItem.h"

#include <assert.h>
#include <thread>

#include <QSGGeometryNode>
#include <QSGGeometry>
#include <QSGSimpleTextureNode>
#include <QQuickWindow>
#include <QFileInfo>
#include <QDir>
#include <QTemporaryFile>

#include "tbb/pipeline.h"

CameraItem::CameraItem()
    : frameRate(this, &CameraItem::frameRateChanged, 15)
    , width(this, &CameraItem::widthChanged, 640)
    , height(this, &CameraItem::heightChanged, 480)
    , cameraInterface(this, &CameraItem::cameraInterfaceChanged, 0)
    , firstCascadeSource(this, &CameraItem::firstCascadeSourceChanged, ":/cascades/haarcascade_frontalface_alt.xml")
    , secondCascadeSource(this, &CameraItem::secondCascadeSourceChanged, ":/cascades/haarcascade_eye.xml")
    , m_Done(false)
{
    cv::ocl::setUseOpenCL(true);

    m_Capture.set(CV_CAP_PROP_FPS, frameRate);
    m_Capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
    m_Capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);

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

    QSGTexture *t = window()->createTextureFromId(static_cast<unsigned int>(*texture), QSize(width, height));

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

void CameraItem::_detectAndDrawTBB(cv::VideoCapture &capture,
                                  CameraItem::Concurent_queue &guiQueue,
                                  CameraItem::Cascade &cascade,
                                  CameraItem::Cascade &nestedCascade,
                                  double scale,
                                  bool tryFlip)
{
    const static cv::Scalar colors[] =
        {
            cv::Scalar(255,0,0),
            cv::Scalar(255,128,0),
            cv::Scalar(255,255,0),
            cv::Scalar(0,255,0),
            cv::Scalar(0,128,255),
            cv::Scalar(0,255,255),
            cv::Scalar(0,0,255),
            cv::Scalar(255,0,255)
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
        cv::cvtColor(pData->image, pData->gray, cv::COLOR_BGR2GRAY);
        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, ProcessingChainData *>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData* pData)->ProcessingChainData*
    {
        double fx = 1 / scale;
        cv::resize(pData->gray, pData->smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR);
        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData* pData)->ProcessingChainData*
    {
        cv::equalizeHist(pData->smallImg, pData->smallImg);
        return pData;
    }
    )&
    tbb::make_filter<ProcessingChainData*, ProcessingChainData*>(tbb::filter::serial_in_order,
                                           [&](ProcessingChainData* pData)->ProcessingChainData*
    {
        cascade.detectMultiScale(pData->smallImg, pData->firstCascadeObjects,
                                 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE,
                                 cv::Size(30, 30));

//        if (tryFlip)
//        {
//            cv::flip(pData->image, pData->image, 1);
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
                cv::Rect& item = pData->firstCascadeObjects[i];

                cv::Point center;
                cv::Scalar color = colors[i%8];

                cv::rectangle( pData->image, cvPoint(cvRound(item.x*scale), cvRound(item.y*scale)),
                               cvPoint(cvRound((item.x + item.width-1)*scale), cvRound((item.y + item.height-1)*scale)),
                               color, 3, 8, 0);

                const int half_height=cvRound(static_cast<float>(item.height/2));
                const int half_width = cvRound(static_cast<float>(item.width / 2));
                item.y=item.y + half_height / 2;
                item.height = half_height-1;
                item.x= item.x + item.width / 2;
                cv::Mat smallImgROI = pData->smallImg(item);

                // detect eyes
                nestedCascade.detectMultiScale( smallImgROI, pData->secondCascadeObjects,
                                                1.1, 3, 0|cv::CASCADE_SCALE_IMAGE,
                                                cv::Size(30, 30) );

                if (!pData->secondCascadeObjects.empty())
                {
                    for (size_t j = 0; j < pData->secondCascadeObjects.size(); ++j)
                    {
                        cv::Rect& eye = pData->secondCascadeObjects.at(j);
                        center.x = cvRound((eye.x + eye.width / 2 + item.x) * scale);
                        center.y = cvRound((eye.y + eye.height / 2 + item.y) * scale);
                        int radius = cvRound((eye.width + eye.height) * 0.25 * scale);

                        cv::Rect eye_rect = cv::Rect(item.x + eye.x, eye.y + item.y, eye.width, eye.height);
                        cv::Mat eye_img = pData->smallImg(eye_rect);

                        // todo: hough circles

                        cv::rectangle(pData->image, eye_rect, color);
                    }
                }
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

