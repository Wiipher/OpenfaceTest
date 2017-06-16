#include "CameraApi.h" //���SDK��APIͷ�ļ�

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

struct cameraArgs
{
    int                     iCameraCounts;
    int                     iStatus;
    tSdkCameraDevInfo       tCameraEnumList;
    int                     hCamera;
    tSdkCameraCapbility     tCapability;      //�豸������Ϣ
    tSdkFrameHead           sFrameInfo;
    BYTE					* pbyBuffer;
    IplImage 				* iplImage;
    int                 	channel;
};

class Camera
{
private:
	unsigned char * g_pRgbBuffer;	
	cameraArgs args;
	
public:		
	bool		cameraInit();
	void		cameraFree();
	cv::Mat 	cameraGetImg();
};
