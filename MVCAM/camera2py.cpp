#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "camera.h"

using namespace boost::python;
using namespace pbcvt;
using namespace cv;

#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
    static void init_ar() {
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }
    
	BOOST_PYTHON_MODULE(pyCamera)
	{
		init_ar();
		
		to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
		pbcvt::matFromNDArrayBoostConverter();
		
		class_<Camera>("Camera")
		.def("cameraInit", &Camera::cameraInit)
		.def("cameraFree", &Camera::cameraFree)
		.def("cameraGetImg", &Camera::cameraGetImg)
		;
	}
