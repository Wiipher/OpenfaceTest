CXX ?=  g++

PYTHON_VERSION = 3.4
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

BOOST_INC = /usr/include/boost
BOOST_LIB = /usr/lib/x86_64-linux-gnu

OPENCV_LIB = `pkg-config --libs opencv`
OPENCV_CFLAGS = `pkg-config --cflags opencv`

MVCAM_INC = /home/lei/MindVision/include
MVCAM_LIB = /home/lei/MindVision/lib/gcc-4.6.3

PBCVT_INC = /home/lei/pyboostcvconverter/include
PBCVT_LIB = /home/lei/pyboostcvconverter/pbcvt.cpython-34m.so

TARGET = pyCamera
SRC = camera2py.cpp camera.cpp
OBJ = camera2py.o camera.o

$(TARGET).so: $(OBJ)
	$(CXX) -shared -Wl,--export-dynamic $(OBJ) -L$(MVCAM_LIB) -lMVSDK -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config-3.4m-x86_64-linux-gnu -lpython$(PYTHON_VERSION) -o $(TARGET).so $(OPENCV_LIB) $(PBCVT_LIB)

$(OBJ): $(SRC)
	$(CXX) -I$(MVCAM_INC) -I$(PBCVT_INC) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) $(OPENCV_CFLAGS) -fPIC -c $(SRC)

clean:
	rm -f $(OBJ)
	rm -f $(TARGET).so