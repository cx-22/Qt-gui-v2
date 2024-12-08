#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include <cmath>

namespace py = pybind11;
using namespace std;
using namespace cv;

py::array_t<uint8_t> resizeLarger(py::array_t<uint8_t> in, int widthFactor, int heightFactor){
	py::buffer_info buffer = in.request();
	int height = buffer.shape[0];
	int width = buffer.shape[1];
	int channels = buffer.shape[2];

	int rows = height * heightFactor;
	int cols = width * widthFactor;

	Mat input(height, width, CV_8UC3, (uint8_t*) buffer.ptr);
	Mat output = Mat::zeros(Size(cols, rows), input.type());

	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			output.at<Vec3b>(i, j) = input.at<Vec3b>(i/heightFactor, j/widthFactor);
		}
	}
	
	py::array_t<uint8_t> out = py::array_t<uint8_t>({rows, cols, channels});
	uint8_t* out_ptr = (uint8_t*) out.request().ptr;
	memcpy(out_ptr, output.data, rows*cols*channels*sizeof(uint8_t));
	return out;
}

py::array_t<uint8_t> quantize(py::array_t<uint8_t> in, int divisor){
	// Makes a table used to quantize color intensity by the given divisor
	Mat table(1, 256, CV_8U);
	for (int i = 0; i < 256; i++){
		table.at<uchar>(0, i) = (uchar)(divisor * (i / divisor));
	}

	py::buffer_info buffer = in.request();
	int rows = buffer.shape[0];
	int cols = buffer.shape[1];
	int channels = buffer.shape[2];

	Mat input(rows, cols, CV_8UC3, (uint8_t*) buffer.ptr);
	Mat output = Mat::zeros(input.size(), input.type());

	// Go through each row in the image
	for (int x = 0; x < rows; x++){
		uchar *ptrIn = input.ptr<uchar>(x);
		uchar *ptrOut = output.ptr<uchar>(x);

		// For every channel in each pixel in the row, take its intensity and
		// quantize using the lookup table, take the result and place it in the
		// appropriate spot in the output image

		for (int y = 0; y < cols; y++){
			for (int c = 0; c < channels; c++){
				ptrOut[y * channels + c] = table.at<uchar>(0, ptrIn[y * channels + c]);
			}
		}
	}

	py::array_t<uint8_t> out = py::array_t<uint8_t>({rows, cols, channels});
	uint8_t* out_ptr = (uint8_t*) out.request().ptr;
	memcpy(out_ptr, output.data, rows*cols*channels*sizeof(uint8_t));
	return out;
}


cv::Mat numpy_to_mat(py::array_t<uint8_t>& input) {
    py::buffer_info info = input.request();
    int rows = info.shape[0];
    int cols = info.shape[1];
    return cv::Mat(rows, cols, CV_8UC3, (unsigned char*)info.ptr);
}

// Helper function to convert cv::Mat to NumPy array
py::array_t<uint8_t> mat_to_numpy(cv::Mat& mat) {
    return py::array_t<uint8_t>(
        { mat.rows, mat.cols, mat.channels() },
        { mat.step[0], mat.step[1], sizeof(uint8_t) },
        mat.data
    );
}




// 0 = all edges, 1 anglemap, / by 2 show diagonal, /3 vertical, /5 horizontal
py::array_t<uint8_t> sobel_filter(py::array_t<uint8_t> in, int blur = 3, int contrast_threshold = 80, bool on_black = false, uchar dvh = 0, float y_percentage = 0){
	Mat input = numpy_to_mat(in);
    Mat copy = input;
	int old_rows = input.rows;
    int old_cols = input.cols;
    Rect roi(0, (old_rows * y_percentage), old_cols, old_rows * (1 - y_percentage));
    Mat cropped = input(roi);
    int rows = cropped.rows;
    int cols = cropped.cols;
    Mat black = Mat::zeros(Size(old_cols, old_rows), input.type());
    cvtColor(cropped, cropped, COLOR_BGR2GRAY);
    Mat sobelX = Mat::zeros(Size(cols, rows), CV_32F);
    Mat sobelY = Mat::zeros(Size(cols, rows), CV_32F);
    Mat* output = (on_black) ? &black : &copy;

    GaussianBlur(cropped, cropped, Size(blur, blur), 0);
    int temp = 0;

	// X
    bool notRight = false;
    bool notLeft = false;
    for (int i = 1; i < rows - 1; i++){
        for (int j = 1; j < cols -1; j++){
            temp = 0;
            notRight = false;
            notLeft = false;
        
            // Not on left edge
            if (j != 0){
                notLeft = true;
                temp += -2 * cropped.at<uchar>(i, j - 1);
            }

            // Not on right edge
            if (j != (cols - 1)){
                notRight = true;
                temp += 2 * cropped.at<uchar>(i, j + 1);
            }

            // Not on top edge
            if (i != 0){
                 if (notLeft){
                    temp += -1 * cropped.at<uchar>(i - 1, j - 1);
                 }
                 
                 if (notRight){
                    temp += 1 * cropped.at<uchar>(i - 1, j + 1);
                 }
            }

            // Not on bottom edge
            if (i != (rows - 1)){
                 if (notLeft){
                    temp += -1 * cropped.at<uchar>(i + 1, j - 1);
                 }
                 
                 if (notRight){
                    temp += 1 * cropped.at<uchar>(i + 1, j + 1);
                 }
            }
            sobelX.at<float>(i, j) = temp;
        }
    }

    // Y
    bool notTop = false;
    bool notBottom = false;
    for (int i = 1; i < rows - 1; i++){
        for (int j = 1; j < cols -1; j++){
            temp = 0;
            notTop = false;
            notBottom = false;
        
            // Not on top edge
            if (i != 0){
                notTop = true;
                temp += -2 * cropped.at<uchar>(i - 1, j);
            }

            // Not on bottom edge
            if (i != (rows - 1)){
                notBottom = true;
                temp += 2 * cropped.at<uchar>(i + 1, j);
            }

            // Not on left edge
            if (j != 0){
                 if (notTop){
                    temp += -1 * cropped.at<uchar>(i - 1, j - 1);
                 }
                 
                 if (notBottom){
                    temp += 1 * cropped.at<uchar>(i + 1, j - 1);
                 }
            }

            // Not on right edge
            if (j != (cols - 1)){
                 if (notTop){
                    temp += -1 * cropped.at<uchar>(i - 1, j + 1);
                 }
                 
                 if (notBottom){
                    temp += 1 * cropped.at<uchar>(i + 1, j + 1);
                 }
            }

            sobelY.at<float>(i, j) = temp;
        }
    }

    float x = 0, y = 0, result = 0, ratio = 0, split = 0.5;
    for (int i = 1; i < rows - 1; i++){
        for (int j = 1; j < cols -1; j++){
            x = sobelX.at<float>(i, j);
            y = sobelY.at<float>(i, j);
            result = sqrt(x * x + y * y);

            x = fabs(x);
            y = fabs(y);

            if (result < contrast_threshold){
                continue;
            }

			if (dvh == 1){
                uchar redI = static_cast<uchar>(min(max(x, 0.0f), 255.0f));
                uchar greenI = static_cast<uchar>(min(max(y, 0.0f), 255.0f));
                output->at<Vec3b>(i + (old_rows * y_percentage), j) = Vec3b(0, greenI, redI);
            } else if (dvh == 0){
                output->at<Vec3b>(i + (old_rows * y_percentage), j) = Vec3b(255, 255, 50);
            } else {
                x = fabs(x);
                y = fabs(y);

                if (x == 0 || y == 0) {
                    ratio = 0; // Horizontal edge
                } else {
                    if (x > y){
                        ratio = y / x;
                    } else if (y > x){
                        ratio = x / y;
                    }
                }
                
                if (ratio > split){
                    if (dvh % 2 == 0){
                        output->at<Vec3b>(i + (old_rows * y_percentage), j) = Vec3b(255, 0, 0);
                    }
                }  else if (ratio < split){
                    if (x > y){
                        if (dvh % 3 == 0){
                            output->at<Vec3b>(i + (old_rows * y_percentage), j) = Vec3b(0, 0, 255);
                        }
                    } else{
                        if (dvh % 5 == 0){
                            output->at<Vec3b>(i + (old_rows * y_percentage), j) = Vec3b(0, 255, 0);
                        }
                    }
                }
        	}
        }
    }
	
	py::array_t<uint8_t> uhh = mat_to_numpy(*output);
    return uhh;
}

PYBIND11_MODULE(cx22, c){
	c.def("resizeLarger", &resizeLarger, "Scales image up");
	c.def("quantize", &quantize, "Quantizes color");
	c.def("sobel_filter", &sobel_filter, "Sobel filter with customization");
}
