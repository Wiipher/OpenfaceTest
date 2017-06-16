#include "camera.h"

bool Camera::cameraInit()
{
    args.iCameraCounts = 0;
    args.iStatus = -1;
    args.iplImage = NULL;
    args.channel = 3;    

    CameraSdkInit(1);

    //枚举设备，并建立设备列表
    CameraEnumerateDevice(&args.tCameraEnumList, &args.iCameraCounts);
	
    //没有连接设备
    if(args.iCameraCounts==0)
	{
		return false;
    }

    //相机初始化。初始化成功后，才能调用任何其他相机相关的操作接口
    args.iStatus = CameraInit(&args.tCameraEnumList, -1, -1, &args.hCamera);

    //初始化失败
    if(args.iStatus != CAMERA_STATUS_SUCCESS)
	{
    	cout << args.iStatus << endl;
		return false;
    }

    //获得相机的特性描述结构体。该结构体中包含了相机可设置的各种参数的范围信息。决定了相关函数的参数
    CameraGetCapability(args.hCamera, &args.tCapability);

    g_pRgbBuffer = (unsigned char*)malloc(args.tCapability.sResolutionRange.iHeightMax * args.tCapability.sResolutionRange.iWidthMax * 3);
    //g_readBuf = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

    /*让SDK进入工作模式，开始接收来自相机发送的图像
    数据。如果当前相机是触发模式，则需要接收到
    触发帧以后才会更新图像。    */
    CameraPlay(args.hCamera);

    /*其他的相机参数设置
    例如 CameraSetExposureTime   CameraGetExposureTime  设置/读取曝光时间
    CameraSetImageResolution  CameraGetImageResolution 设置/读取分辨率
    CameraSetGamma、CameraSetConrast、CameraSetGain等设置图像伽马、对比度、RGB数字增益等等。
    更多的参数的设置方法，，清参考MindVision_Demo。
	本例程只是为了演示如何将SDK中获取的图像，转成OpenCV的图像格式,以便调用OpenCV的图像处理函数进行后续开发*/

    if(args.tCapability.sIspCapacity.bMonoSensor)
	{
        args.channel=1;
        CameraSetIspOutFormat(args.hCamera, CAMERA_MEDIA_TYPE_MONO8);
    }
	else
	{
        args.channel=3;
        CameraSetIspOutFormat(args.hCamera, CAMERA_MEDIA_TYPE_BGR8);
    }
    	
    return true;		
}

void Camera::cameraFree()
{
    CameraUnInit(args.hCamera);
    //注意，现反初始化后再free
    free(g_pRgbBuffer);			
}

cv::Mat Camera::cameraGetImg()
{
    if(CameraGetImageBuffer(args.hCamera, &args.sFrameInfo, &args.pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
	{
	    CameraImageProcess(args.hCamera, args.pbyBuffer, g_pRgbBuffer, &args.sFrameInfo);
	    if (args.iplImage)
    	{
        	cvReleaseImageHeader(&args.iplImage);
        }
        args.iplImage = cvCreateImageHeader(cvSize(args.sFrameInfo.iWidth, args.sFrameInfo.iHeight), IPL_DEPTH_8U, args.channel);
        cvSetData(args.iplImage, g_pRgbBuffer, args.sFrameInfo.iWidth * args.channel);//此处只是设置指针，无图像块数据拷贝，不需担心转换效率
        //以下两种方式都可以显示图像或者处理图像
        #if 0
        cvShowImage("OpenCV Demo", args.iplImage);
        #else
        //Mat Iimag(iplImage);//这里只是进行指针转换，将IplImage转换成Mat类型
        cv::Mat Iimag = cvarrToMat(args.iplImage);
		//imshow("OpenCV Demo", Iimag);
        #endif

        waitKey(5);

	    //在成功调用CameraGetImageBuffer后，必须调用CameraReleaseImageBuffer来释放获得的buffer。
		//否则再次调用CameraGetImageBuffer时，程序将被挂起一直阻塞，直到其他线程中调用CameraReleaseImageBuffer来释放了buffer
		CameraReleaseImageBuffer(args.hCamera, args.pbyBuffer);	
		return Iimag;	
	}	
	return cv::Mat::zeros(1, 1, CV_8UC3);
}