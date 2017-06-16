#include "camera.h"

bool Camera::cameraInit()
{
    args.iCameraCounts = 0;
    args.iStatus = -1;
    args.iplImage = NULL;
    args.channel = 3;    

    CameraSdkInit(1);

    //ö���豸���������豸�б�
    CameraEnumerateDevice(&args.tCameraEnumList, &args.iCameraCounts);
	
    //û�������豸
    if(args.iCameraCounts==0)
	{
		return false;
    }

    //�����ʼ������ʼ���ɹ��󣬲��ܵ����κ����������صĲ����ӿ�
    args.iStatus = CameraInit(&args.tCameraEnumList, -1, -1, &args.hCamera);

    //��ʼ��ʧ��
    if(args.iStatus != CAMERA_STATUS_SUCCESS)
	{
    	cout << args.iStatus << endl;
		return false;
    }

    //�����������������ṹ�塣�ýṹ���а�������������õĸ��ֲ����ķ�Χ��Ϣ����������غ����Ĳ���
    CameraGetCapability(args.hCamera, &args.tCapability);

    g_pRgbBuffer = (unsigned char*)malloc(args.tCapability.sResolutionRange.iHeightMax * args.tCapability.sResolutionRange.iWidthMax * 3);
    //g_readBuf = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

    /*��SDK���빤��ģʽ����ʼ��������������͵�ͼ��
    ���ݡ������ǰ����Ǵ���ģʽ������Ҫ���յ�
    ����֡�Ժ�Ż����ͼ��    */
    CameraPlay(args.hCamera);

    /*�����������������
    ���� CameraSetExposureTime   CameraGetExposureTime  ����/��ȡ�ع�ʱ��
    CameraSetImageResolution  CameraGetImageResolution ����/��ȡ�ֱ���
    CameraSetGamma��CameraSetConrast��CameraSetGain������ͼ��٤���Աȶȡ�RGB��������ȵȡ�
    ����Ĳ��������÷���������ο�MindVision_Demo��
	������ֻ��Ϊ����ʾ��ν�SDK�л�ȡ��ͼ��ת��OpenCV��ͼ���ʽ,�Ա����OpenCV��ͼ���������к�������*/

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
    //ע�⣬�ַ���ʼ������free
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
        cvSetData(args.iplImage, g_pRgbBuffer, args.sFrameInfo.iWidth * args.channel);//�˴�ֻ������ָ�룬��ͼ������ݿ��������赣��ת��Ч��
        //�������ַ�ʽ��������ʾͼ����ߴ���ͼ��
        #if 0
        cvShowImage("OpenCV Demo", args.iplImage);
        #else
        //Mat Iimag(iplImage);//����ֻ�ǽ���ָ��ת������IplImageת����Mat����
        cv::Mat Iimag = cvarrToMat(args.iplImage);
		//imshow("OpenCV Demo", Iimag);
        #endif

        waitKey(5);

	    //�ڳɹ�����CameraGetImageBuffer�󣬱������CameraReleaseImageBuffer���ͷŻ�õ�buffer��
		//�����ٴε���CameraGetImageBufferʱ�����򽫱�����һֱ������ֱ�������߳��е���CameraReleaseImageBuffer���ͷ���buffer
		CameraReleaseImageBuffer(args.hCamera, args.pbyBuffer);	
		return Iimag;	
	}	
	return cv::Mat::zeros(1, 1, CV_8UC3);
}