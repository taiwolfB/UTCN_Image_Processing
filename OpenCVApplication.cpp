// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>

typedef struct mat_colorT
{
	Mat mat;
	Vec3b color;
}mat_color;

int n8DX[] = { -1, -1, 0, 1, 1, 1, 0, -1 };
int n8DY[] = { 0, 1, 1, 1, 0, -1, -1, -1 };



void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

// lab 1
void testChangeGrayLevel()
{
	char filename[MAX_PATH];

	while (openFileDlg(filename))
	{
		Mat src = imread(filename, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dst.at<uchar>(i,j) = min(255, src.at<uchar>(i, j) + 50);
		
		imshow("sourceImage", src);
		imshow("imageGrayerScale", dst);

		waitKey();
	}

}

void testFourColoredQuarters()
{
	int height = 200;
	int width = 200;

	Mat image = Mat(height, width, CV_8UC3);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (i < height / 2 && j < width / 2)
				image.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
			if (i >= height / 2 && j < width / 2)
				image.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			if (i < height / 2 && j >= width / 2)
				image.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			if (i >= height / 2 && j >= width / 2)
				image.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
		}
	}
	imshow("quarters", image);
	waitKey();
}

void testInverseMatrix()
{
	float values[9] = { 1,2,3,4,51,6,7,8,11 };
	Mat M(3, 3, CV_32FC1, values);
	
	std::cout << M.inv() << "\n";

	getchar();
	getchar();
}

///lab 2

void testRGB24Split()
{
	char filename[MAX_PATH];

	while (openFileDlg(filename))
	{
		Mat src = imread(filename, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat redPixels(height, width, CV_8UC3);
		Mat greenPixels(height, width, CV_8UC3);
		Mat bluePixels(height, width, CV_8UC3);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				redPixels.at<Vec3b>(i, j) = Vec3b(0, 0, src.at<Vec3b>(i, j)[2]);
				greenPixels.at<Vec3b>(i, j) = Vec3b(0, src.at<Vec3b>(i, j)[1], 0);
				bluePixels.at<Vec3b>(i, j) = Vec3b(src.at<Vec3b>(i, j)[0], 0, 0);
			}
		}

		imshow("red", redPixels);
		imshow("green", greenPixels);
		imshow("blue", bluePixels);
	}

	getchar();
	getchar();

}

void testRgb24ToGrayScale()
{
	char filename[MAX_PATH];

	while (openFileDlg(filename))
	{
		Mat src = imread(filename, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat grayScaleFromRgb24(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				grayScaleFromRgb24.at<uchar>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;

		imshow("GrayScale from RGB24", grayScaleFromRgb24);
	}

	getchar();
	getchar();

}

void testConvertingGrayScaleToBlackOrWhite(int thresholdValue)
{
	char filename[MAX_PATH];

	while (openFileDlg(filename))
	{
		Mat src = imread(filename, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat blackOrWhite(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				blackOrWhite.at<uchar>(i, j) = src.at<uchar>(i, j) < thresholdValue ? 0 : 255;

		imshow("GrayScale from RGB24", blackOrWhite);
	}

	getchar();
	getchar();
}

void testRGBtoHSV()
{
	char filename[MAX_PATH];

	while (openFileDlg(filename))
	{
		Mat src = imread(filename, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;

		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{

				float currentNormalizedR = (float)src.at<Vec3b>(i, j)[2] / 255;
				float currentNormalizedG = (float)src.at<Vec3b>(i, j)[1] / 255;
				float currentNormalizedB = (float)src.at<Vec3b>(i, j)[0] / 255;

				float currentMax = max(currentNormalizedR, max(currentNormalizedG, currentNormalizedB));
				float currentMin = min(currentNormalizedR, min(currentNormalizedG, currentNormalizedB));
				float currentC = currentMax - currentMin;


				V.at<uchar>(i, j) = currentMax * 255;

				S.at<uchar>(i, j) = currentMax != 0 ? (currentC / currentMax) * 255 : 0;
				float tmpH;
				if (currentC != 0)
				{
					if (currentMax == currentNormalizedR)
						tmpH = (60 * (currentNormalizedG - currentNormalizedB) / currentC);
					if (currentMax == currentNormalizedG)
						tmpH = (120 + 60 * (currentNormalizedB - currentNormalizedR) / currentC);
					if (currentMax == currentNormalizedB)
						tmpH = (240 + 60 * (currentNormalizedR - currentNormalizedG) / currentC);
				}
				else
					tmpH = 0;

				if (tmpH < 0)
					tmpH += 360;

				H.at<uchar>(i, j) = tmpH * 255 / 360;



			}

		imshow("H", H);
		imshow("S", S);
		imshow("V", V);
		waitKey();
	}

}


// lab 3
int* calculateH(Mat src)
{
	int* h = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			h[src.at<uchar>(i, j)]++;

	return h;
}


float* calculatePDF(Mat src)
{
	float* p = (float*)calloc(256, sizeof(float));
	int* h = calculateH(src);
	int M = src.rows * src.cols;
	for (int i = 0; i < 256; i++)
		*(p + i) = (float)*(h + i) / (float) M;

	return p;
}
float averageInRange(float* p, int start, int end, int windowSize)
{
	float average = 0;

	for (int i = start; i <= end; i++)
		average += *(p + i);

	average /= windowSize * 2 + 1;

	return average;
}

float* calculateAverages(float* p, int windowSize)
{
	float* v = (float*)calloc(256, sizeof(float));

	for (int k = windowSize; k < 256 - windowSize; k++)
		*(v + k) = averageInRange(p, k - windowSize, k + windowSize, windowSize);

	return v;
}



void testShowHistogram()
{
	int* h = (int*)malloc(256 * sizeof(int));
	float* p;
	float max = 0;

	char filename[MAX_PATH];

	while (openFileDlg(filename))
	{
		Mat src = imread(filename, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int M = height * width;
	
		h = calculateH(src);
		p = calculatePDF(src);
					

		showHistogram("GrayScaleHistogram", h, 256, 500);

		waitKey();
	}

	getchar();
	getchar();

}

float maxInWindow(float* p, int start, int end)
{
	float max = -1;

	for (int i = start; i <= end; i++)
		if (max < *(p + i))
			max = *(p + i);

	return max;
}

int findNearestMaximum(int* localMaxima, int localMaximaSize,int target)
{
	int max = *(localMaxima);
	int minDistance = 256;
	int pixel = 0;
	for (int i = 0 ; i < localMaximaSize; i++)
	{	
		int d = abs(*(localMaxima + i) - target); 
		
		if (d < minDistance)
		{
			minDistance = d;
			pixel = *(localMaxima + i);
		}
	}

	return pixel;
}


 void testMultilevelThresholding()
{
	float th = 0.0003;
	float wh = 5;
	float* averages = (float*)calloc(256, sizeof(float));
	int* localMaxima = (int*)calloc(256, sizeof(int));
	float* p;
	int* h;
	float max = 0;
	int localMaximaSize = 1;


	char filename[MAX_PATH];

	while (openFileDlg(filename))
	{
		Mat src = imread(filename, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		h = calculateH(src);
		p = calculatePDF(src);
		//averages = calculateAverages(p,wh);

		
		for (int i = wh; i < 256 - wh; i++)
		{
			int avg = averageInRange(p, i - wh, i + wh,wh);

			if (*(p + i) >avg + th && ( *(p + i) == maxInWindow(p, i - wh, i + wh)))
			{
				
				*(localMaxima + localMaximaSize) = i;
				 localMaximaSize++;
				 
			}
		}

		

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				src.at<uchar>(i, j) = findNearestMaximum(localMaxima, localMaximaSize, src.at<uchar>(i, j));





		imshow("Multilevel Thresholding", src);
		

		waitKey();
	}

	getchar();
	getchar();

}

 // lab 4

 void testHorizontalProjection(Mat src, Vec3b color)
 {
	 int cnt;
	 Mat horizontalProjection = Mat(src.rows, src.cols, CV_8UC1);

	 for (int i = 0; i < src.rows; i++)
	 {
		 cnt = 0;
		 for (int j = 0; j < src.cols; j++)
		 {
			 horizontalProjection.at<uchar>(i, j) = 255;
			 if (src.at<Vec3b>(i, j) == color)
			 {

				 horizontalProjection.at<uchar>(i, cnt) = 0;
				 cnt++;
			 }
		 }
	 }

	 imshow("horizontalProjection", horizontalProjection);


 }

 void testVerticalProjection(Mat src, Vec3b color)
 {
	 int cnt;
	 Mat verticalProjection = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));

	 for (int j = 0; j < src.cols; j++)
	 {
		 cnt = 0;

		 for (int i = 0; i < src.rows; i++)
			if (src.at<Vec3b>(i, j) == color)
			 {
				 verticalProjection.at<uchar>(cnt, j) = 0;
				 cnt++;
			 }
	 }

	 imshow("verticalProjection", verticalProjection);

 }


 void geometricalFeaturesCallBackFunction(int event, int x, int y, int flags, void* param)
 {
	 //More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	 Mat* src = (Mat*)param;
	 
	 if (event == EVENT_LBUTTONDBLCLK)
	 {
		 int area = 0;
		 Vec3b color = src->at<Vec3b>(y, x);

		 for (int i = 0; i < src->rows; i++)
			 for (int j = 0; j < src->cols; j++)
				 if (src->at<Vec3b>(i, j) == color)
					 area++;

		 printf("Area = %d \n", area);

		 int rCenterOfMass = 0;
		 int cCenterOfMass = 0;

		 for (int i = 0; i < src->rows; i++)
			 for (int j = 0; j < src->cols; j++)
			 {
				 if (src->at<Vec3b>(i, j) == color)
				 {
					 rCenterOfMass += i;
					 cCenterOfMass += j;
				 }
			 }

		 rCenterOfMass /= area;
		 cCenterOfMass /= area;

		 printf("Center of mass\n c = %d \n r = %d \n", cCenterOfMass, rCenterOfMass);


		 int regularRowsCenterSquared = 0;
		 int regularColsCenterSquared = 0;
		 int rowsAndColsSum = 0;

		 for (int i = 0; i < src->rows; i++)
			 for (int j = 0; j < src->cols; j++)
			 {
				 if (src->at<Vec3b>(i, j) == color)
				 {
					 rowsAndColsSum += (i - rCenterOfMass) * (j - cCenterOfMass);
					 regularRowsCenterSquared += (i - rCenterOfMass) * (i - rCenterOfMass);
					 regularColsCenterSquared += (j - cCenterOfMass) * (j - cCenterOfMass);
				 }
			 }


		 int Y = 2 * rowsAndColsSum;
		 int X = regularColsCenterSquared - regularRowsCenterSquared;
		 float phi = atan2(Y, X) / 2;

		 if (phi < 0)
			 phi += CV_PI;
		 phi = phi * 180 / CV_PI;

		 float perimeter = 0;


		 for (int i = 0; i < src->rows; i++)
			 for (int j = 0; j < src->cols; j++)
			 {
				 if (src->at<Vec3b>(i, j) == color)
				 {
					 bool neighbor = false;
					 for (int k = 0; k < 8; k++)
						 if (i + n8DX[k] >= 0 && i + n8DX[k] <= src->rows && j + n8DY[k] >= 0 && j + n8DY[k] <= src->cols)
							 if (src->at<Vec3b>(i + n8DX[k], j + n8DY[k]) != color)
							 {
								 neighbor = true;
								 break;
							 }
					 if (neighbor)
						 perimeter++;
				 }
			 }

		 perimeter = perimeter * CV_PI / 4;
		 printf("Perimeter = %f \n", perimeter);

		 float thinessRatio = 4 * CV_PI * area / (perimeter * perimeter);

		 printf("Thiness ratio = %f \n", thinessRatio);

		 int cMax = -1;
		 int cMin = INT_MAX;
		 int rMin = INT_MAX;
		 int rMax = -1;

		 for (int i = 0; i < src->rows; i++)
			 for (int j = 0; j < src->cols; j++)
			 {
				 if (src->at<Vec3b>(i, j) == color)
				 {
					 if (i >= rMax)
						 rMax = i;
					 if (i <= rMin)
						 rMin = i;
					 if (j >= cMax)
						 cMax = j;
					 if (j <= cMin)
						 cMin = j;

				 }
			 }

		 float aspectRatio = (float)(cMax - cMin + 1) / (rMax - rMin + 1);
		 printf("Aspect ratio = %f \n", aspectRatio);	
		  
		 int rA = rCenterOfMass + tan(phi * CV_PI / 180) * (cMin - cCenterOfMass);
		 int rB = rCenterOfMass + tan(phi * CV_PI / 180) * (cMax - cCenterOfMass);
		 Point A(cMin, rA);
		 Point B(cMax, rB);
		 line(*src, A, B, Scalar(0, 0, 0), 2);


		 imshow("elongationAxis", *src);
		 testHorizontalProjection(*src, color);
		 testVerticalProjection(*src, color);
		
	 }
	
 }
 

 void testGeometricalFeatures()
 {
	 Mat src;
	 Mat copySrc;
	 // Read image from file 
	 char fname[MAX_PATH];
	 if (openFileDlg(fname))
	 {
		 src = imread(fname, IMREAD_COLOR);
		 int height = src.rows;
		 int width = src.cols;
		 //Create a window
		 namedWindow("My Window", 1);

		 //set the callback function for any mouse event
		 //mat_color mat_and_color;
		 //mat_and_color.color = Vec3b(0, 0, 0);
		 //mat_and_color.mat = Mat(height, width, CV_8UC1);

		 setMouseCallback("My Window", geometricalFeaturesCallBackFunction, &src);

		 //show the image
		 imshow("My Window", src);

		 // Wait until user press some key
		 waitKey(0);
	 }
 }

 //lab 5

 Mat_<Vec3b> generateColors(int height, int width, Mat_<int> labels) {
	 Mat_<Vec3b> dst(height, width);

	 int maxLabel = 0;
	 for (int i = 0; i < height; i++)
		 for (int j = 0; j < width; j++)
			 if (labels(i, j) > maxLabel)
				 maxLabel = labels(i, j);

	 std::default_random_engine engine;
	 std::uniform_int_distribution<int> distribution(0, 255);

	 std::vector<Vec3b> colors(maxLabel + 1);

	 for (int i = 0; i <= maxLabel; i++) {
		 uchar r = distribution(engine);
		 uchar g = distribution(engine);
		 uchar b = distribution(engine);

		 colors.at(i) = Vec3b(r, g, b);
	 }

	 for (int i = 0; i < height; i++) {
		 for (int j = 0; j < width; j++) {
			 int label = labels(i, j);

			 dst(i, j) = (label > 0) ? colors.at(labels(i, j)) : dst(i, j) = Vec3b(255, 255, 255);
		 }
	 }

	 return dst;
 }


 void testBFSLabelingAlgorithm()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 int height = src.rows;
		 int width = src.cols;
		 int label = 0;
		 Mat_<int> labels = Mat_<int>(height, width,0);

		 for (int i = 0; i < height; i++)
		 {
			 for (int j = 0; j < width; j++)
			 {
					if(src.at<uchar>(i,j) == 0 && labels.at<int>(i,j) == 0 )
					{
						label++;
						std::queue<Point> Q;
						labels.at<int>(i, j) = label;
						Point point = Point(j, i);
						Q.push(point);
						while (!Q.empty())
						{
							Point q = Q.front();
							Q.pop();
							for (int k = 0; k < 8; k++)
							{	
							
								int nextX = q.x + n8DX[k];
								int nextY = q.y + n8DY[k];
								if (nextY >= 0 && nextY < src.rows && nextX >= 0 && nextX < src.cols)
								{
									if (src.at<uchar>(nextY, nextX) == 0 && labels.at<int>(nextY,nextX) == 0)
									{
										labels.at<int>(nextY, nextX) = label;
										Q.push(Point(nextX,nextY));
									}
								}
							}
						}

					}
			 }	
			 
		 }

		 Mat coloredImage = generateColors(height, width, labels);

		 imshow("Multilevel Thresholding", src);
		 imshow("Colored Image", coloredImage);
		 waitKey(0);

		
	 }

 }



 void testTwoPassAlgorithm() {
	 char fileName[MAX_PATH];
	 while (openFileDlg(fileName)) {
		 Mat src = imread(fileName, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;

		 Mat_<int> labels = Mat_<int>(height, width, 0);

		

		 int dx[] = { 0, -1, -1, -1 };
		 int dy[] = { -1, -1, 0, 1 };

		 int label = 0;
		 std::vector<std::vector<int>> edges;
		 edges.resize(width * height + 1);

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 if (src.at<uchar>(i, j) == 0 && labels(i, j) == 0) {
					 std::vector<int> L;

					 for (int k = 0; k < 4; k++)
					 {	
						 int nextX = i + dx[k];
						 int nextY = j + dy[k];
						 if (nextY >= 0 && nextY < src.cols && nextX >= 0 && nextX < src.rows)
							 if (labels(nextX, nextY) > 0)
								 L.push_back(labels(nextX, nextY));
					 }

					 if (L.size() == 0) {
						 label++;
						 labels(i, j) = label;
					 }
					 else {
						 int minElement = *min_element(L.begin(), L.end());

						 labels(i, j) = minElement;
						 for (int elem : L)
							 if (elem != minElement) {
								 edges[minElement].push_back(elem);
								 edges[elem].push_back(minElement);
							 }
					 }
				 }

		 int newLabel = 0;
		 int* newLabels = (int*)calloc(width * height + 1, sizeof(int));

		 for (int j = 1; j <= label; j++)
			 if (newLabels[j] == 0) {
				 newLabel++;
				 newLabels[j] = newLabel;

				 std::queue<int> Q;
				 Q.push(j);

				 while (!Q.empty()) {
					 int poppedElem = Q.front();
					 Q.pop();

					 for (int elem : edges[poppedElem])
						 if (newLabels[elem] == 0) {
							 newLabels[elem] = newLabel;
							 Q.push(elem);
						 }
				 }
			 }

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 labels(i, j) = newLabels[labels(i, j)];


		 Mat_<Vec3b> twoPassImg = generateColors(height, width, labels);

		 imshow("Initial Image", src);
		 imshow("Two-Pass Image", twoPassImg);
		 waitKey(0);
	 }
 }

 //lab 6
 void testBorderTracingAlgorithm()
 {
	 Mat src;
	 Mat copySrc;
	 Mat contour;
	 // Read image from file 
	 char fname[MAX_PATH];

	 int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	 int dx[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	 int AC[10000];
	 int DC[10000];

	 if (openFileDlg(fname))
	 {
		 src = imread(fname, IMREAD_GRAYSCALE);
		 int height = src.rows;
		 int width = src.cols;
		 Point P0 = Point(0, 0);
		 for (int i = 0; i < height; i++)
		 {
			 for (int j = 0; j < width; j++)
				 if (src.at<uchar>(i, j) == 0)
				 {
					 P0.x = j;
					 P0.y = i;
					 j = width + 1;
					 i = height + 1;
				 }
			 
		 }

		 int dir = 7;
		 int n = 0;
		 int k = 0;
		 Point P1 = Point(P0.x, P0.y);
		 Point Pn1 = Point(P0.x, P0.y);
		 Point Pn = Point(P0.x, P0.y);
		 bool foundP1 = false;
		 contour = Mat(height, width, CV_8UC1, Scalar(255));
		// for (int i = 0; i < height; i++)
		//	 for (int j = 0; j < width; j++)
				// contour.at<uchar>(i, j) = 255;
		 contour.at<uchar>(P0.y, P0.x) = 0;

		 do
		 {
			 Pn1 = Point(Pn.x, Pn.y);
			 n++;

			 if (dir % 2 == 0)
				 dir = (dir + 7) % 8;
			 else
				 dir = (dir + 6) % 8;

			 while (src.at<uchar>(Pn.y + dy[dir], Pn.x + dx[dir]) > 0)
				 dir = (dir + 1) % 8;	

			 if (!foundP1)
			 {
				 P1.x = Pn.x + dx[dir];
				 P1.y = Pn.y + dy[dir];
				 foundP1 = true;
			 }

			 Pn.x = Pn.x + dx[dir];
			 Pn.y = Pn.y + dy[dir];

			 
			 contour.at<uchar>(Pn.y, Pn.x) = 0;
			 uchar var = contour.at<uchar>(Pn.y, Pn.x);
			 AC[k] = dir;

			 if (k >= 1)
			 {
				 DC[k - 1] = (AC[k] - AC[k - 1] + 8) % 8;
			 }
			 k++;

		 }while(!((Pn.x == P1.x && Pn.y == P1.y) && (Pn1.x == P0.x && Pn1.y == P0.y) && (n >= 2)));

		 printf("AC :");
		 for (int i = 0; i < k; i++)
			 printf("%d ", AC[i]);
		 printf("\nDC :");
		 for (int i = 0; i < k - 1; i++)
			 printf("%d ", DC[i]);

		 imshow("contour", contour);
		 waitKey();

		 
	 }
	 
 }

 void testContourReconstruction() {

	 Mat src = imread("./files_border_tracing/gray_background.bmp", IMREAD_GRAYSCALE);

	 FILE* fp;
	 fp = fopen("./files_border_tracing/reconstruct.txt", "r");
		
	 int x, y;
	 fscanf(fp, "%d %d", &y, &x);

	 Point P0 = Point(x, y);
	 src.at<uchar>(y, x) = 0;

	 int n;
	 fscanf(fp, "%d", &n);

	 int dx[] = { 1,  1,  0, -1, -1, -1, 0, 1 };
	 int dy[] = { 0, -1, -1, -1,  0,  1, 1, 1 };

	 int dir;
	 for (int i = 0; i < n; i++) {
		 fscanf(fp, "%d", &dir);

		 x = P0.x + dx[dir];
		 y = P0.y + dy[dir];

		 P0 = Point(x, y);

		 src.at<uchar>(y, x) = 0;
	 }

	 imshow("reconstruct", src);

	 waitKey(0);



 }

 //Lab 7 

 bool isInImage(int height, int width, int i, int j)
 {
	 return (i < height && i >= 0 && j >= 0 && j < width);
 }
 Mat createStructuringElement()
 {
	 Mat struct_elem(3, 3, CV_8UC1, Scalar(255));
	 struct_elem.at<uchar>(1, 1) = 0;
	 struct_elem.at<uchar>(1, 0) = 0;
	 struct_elem.at<uchar>(2, 1) = 0;
	 struct_elem.at<uchar>(1, 2) = 0;
	 struct_elem.at<uchar>(0, 1) = 0;

	 return struct_elem;

 }
 void testDilation(int n)
 {

	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = src.clone();
		 Mat struct_elem = createStructuringElement();

		 imshow("initialImage", src);
		 
		 for (int nr = 0; nr < n; nr++)
		 {
		
			 for (int i = 0; i < height; i++)
			 {
				 for (int j = 0; j < width; j++)
				 {
					 if (src.at<uchar>(i, j) == 0)
					 {
						// dst.at<uchar>(i, j) = 0;
						 for(int si = 0 ; si < struct_elem.rows; si++)
							 for(int sj = 0 ; sj < struct_elem.cols; sj++)
								 if (struct_elem.at<uchar>(si, sj) == 0)
								 {
										 int nextI = i + si - 1;
										 int nextJ = j + sj - 1;
										 if (isInImage(height, width, nextI, nextJ))
										 {
											 dst.at<uchar>(nextI, nextJ) = 0;
										 }
								}

					 }
				 }
			 }
			 src = dst.clone();
			 
		 }

		
		 imshow("dilation", dst);

		
		 waitKey();
	 }

 }

 void testErosion(int n)
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = src.clone();
		 Mat struct_elem = createStructuringElement();

		 imshow("initialImage", src);

		 for (int nr = 0; nr < n; nr++)
		 {

			 for (int i = 0; i < height; i++)
			 {
				 for (int j = 0; j < width; j++)
				 {
					 if (src.at<uchar>(i, j) == 0)
					 {
						 dst.at<uchar>(i, j) = 0;
						 int cnt = 0;
						 bool flag = true;
						 for (int si = 0; si < struct_elem.rows && flag; si++)
							 for (int sj = 0; sj < struct_elem.cols && flag; sj++)
								 if (struct_elem.at<uchar>(si, sj) == 0)
								 {
									 int nextI = i + si - 1;
									 int nextJ = j + sj - 1;
									 if (isInImage(height, width, nextI, nextJ) && src.at<uchar>(nextI, nextJ) == 255)
									 {
										 dst.at<uchar>(i, j) = 255;
										 flag = false;
									 }
								 }
						
					 }
				 }
			 }
			 src = dst.clone();

		 }


		 imshow("erosion", dst);


		 waitKey();
	 }
 }

 void testOpening()
 {

	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = src.clone();
		 Mat struct_elem = createStructuringElement();

		 imshow("initialImage", src);

		 //Erosion
		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 if (src.at<uchar>(i, j) == 0)
				 {
					 dst.at<uchar>(i, j) = 0;
					 int cnt = 0;
					 bool flag = true;

					 for (int si = 0; si < struct_elem.rows && flag; si++)
						 for (int sj = 0; sj < struct_elem.cols && flag; sj++)
							 if (struct_elem.at<uchar>(si, sj) == 0)
							 {
								 int nextI = i + si - 1;
								 int nextJ = j + sj - 1;
								 if (isInImage(height, width, nextI, nextJ) && src.at<uchar>(nextI, nextJ) == 255)
								 {
									 dst.at<uchar>(i, j) = 255;
									 flag = false;
								 }
							 }

				 }

			 src = dst.clone();

			 //Dilate
			 for (int i = 0; i < height; i++)
				 for (int j = 0; j < width; j++)
					 if (src.at<uchar>(i, j) == 0)
					 {
						 dst.at<uchar>(i, j) = 0;
						 for (int si = 0; si < struct_elem.rows; si++)
							 for (int sj = 0; sj < struct_elem.cols; sj++)
								 if (struct_elem.at<uchar>(si, sj) == 0)
								 {
									 int nextI = i + si - 1;
									 int nextJ = j + sj - 1;
									 if (isInImage(height, width, nextI, nextJ))
									 {
										 dst.at<uchar>(nextI, nextJ) = 0;
									 }
								 }

					 }

			 imshow("opening", dst);

			 waitKey();
	 }
}


 void testClosing()
 {

	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = src.clone();
		 Mat struct_elem = createStructuringElement();

		 imshow("initialImage", src);


		 //Dilate
		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 if (src.at<uchar>(i, j) == 0)
				 {
					 dst.at<uchar>(i, j) = 0;
					 for (int si = 0; si < struct_elem.rows; si++)
						 for (int sj = 0; sj < struct_elem.cols; sj++)
							 if (struct_elem.at<uchar>(si, sj) == 0)
							 {
								 int nextI = i + si - 1;
								 int nextJ = j + sj - 1;
								 if (isInImage(height, width, nextI, nextJ))
								 {
									 dst.at<uchar>(nextI, nextJ) = 0;
								 }
							 }

				 }


		 src = dst.clone();

		 //Erosion
		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 if (src.at<uchar>(i, j) == 0)
				 {
					 dst.at<uchar>(i, j) = 0;
					 int cnt = 0;
					 bool flag = true;

					 for (int si = 0; si < struct_elem.rows && flag; si++)
						 for (int sj = 0; sj < struct_elem.cols && flag; sj++)
							 if (struct_elem.at<uchar>(si, sj) == 0)
							 {
								 int nextI = i + si - 1;
								 int nextJ = j + sj - 1;
								 if (isInImage(height, width, nextI, nextJ) && src.at<uchar>(nextI, nextJ) == 255)
								 {
									 dst.at<uchar>(i, j) = 255;
									 flag = false;
								 }
							 }

				 }


	

		 imshow("closing", dst);

		 waitKey();
	 }
 }

 void testBoundaryExtraction()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = src.clone();
		 Mat struct_elem = createStructuringElement();

		 imshow("initialImage", src);

		 //Erosion
		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 if (src.at<uchar>(i, j) == 0)
				 {
					 dst.at<uchar>(i, j) = 0;
					 int cnt = 0;
					 bool flag = true;

					 for (int si = 0; si < struct_elem.rows && flag; si++)
						 for (int sj = 0; sj < struct_elem.cols && flag; sj++)
							 if (struct_elem.at<uchar>(si, sj) == 0)
							 {
								 int nextI = i + si - 1;
								 int nextJ = j + sj - 1;
								 if (isInImage(height, width, nextI, nextJ) && src.at<uchar>(nextI, nextJ) == 255)
								 {
									 dst.at<uchar>(i, j) = 255;
									 flag = false;
								 }
							 }

				 }
		 Mat temp = dst.clone();
		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 if (temp.at<uchar>(i, j) != src.at<uchar>(i, j))
					 dst.at<uchar>(i, j) = 0;
				 else
					 dst.at<uchar>(i, j) = 255;


		 imshow("boundary_extraction", dst);

		 waitKey();
	 }
 }

 bool equalMats(Mat mat1, Mat mat2)
 {
	 
	 for (int i = 0; i < mat1.rows; i++)
		 for (int j = 0; j < mat1.cols; j++)
			 if (mat1.at<uchar>(i, j) != mat2.at<uchar>(i, j))
				 return false;

	 return true;
 }
 
 Mat intersectMat(Mat mat1, Mat mat2)
 {
	 Mat newMat = Mat(mat1.rows, mat1.cols, CV_8UC1, Scalar(255));
	 for (int i = 0; i < mat1.rows; i++)
		 for (int j = 0; j < mat1.cols; j++)
			 if (mat1.at<uchar>(i, j) == mat2.at<uchar>(i, j))
				 newMat.at<uchar>(i, j) = mat1.at<uchar>(i,j);
	 return newMat;
 }

 Mat dilate(Mat src)
 {
	 
	 int height = src.rows;
	 int width = src.cols;
	 Mat dst = src.clone();
	 Mat struct_elem = createStructuringElement();
	 for (int i = 0; i < height; i++)
		 for (int j = 0; j < width; j++)
			 if (src.at<uchar>(i, j) == 0)
			 {
				 dst.at<uchar>(i, j) = 0;
				 for (int si = 0; si < struct_elem.rows; si++)
					 for (int sj = 0; sj < struct_elem.cols; sj++)
						 if (struct_elem.at<uchar>(si, sj) == 0)
						 {
							 int nextI = i + si - 1;
							 int nextJ = j + sj - 1;
							 if (isInImage(height, width, nextI, nextJ))
							 {
								 dst.at<uchar>(nextI, nextJ) = 0;
							 }
						 }

			 }

	 return dst;
 }
 void testRegionFilling()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;
		
		 Mat struct_elem = createStructuringElement();
		 Mat prevX = Mat(height, width, CV_8UC1, Scalar(255));
		 
		 Mat Ac = Mat(height, width, CV_8UC1, Scalar(255));

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 Ac.at<uchar>(i, j) = src.at<uchar>(i, j) == 0 ? 255 : 0;


		 imshow("initialImage", src);

		 prevX.at<uchar>(height / 2, width / 2) = 0;


		 while(1)
		 {	
			 Mat currX = dilate(prevX);
			 Mat temp = intersectMat(currX, Ac);
			 if (equalMats(prevX, temp))
				 break;
			 prevX = temp.clone();
			 
		 } 
		

		 imshow("regionFilling", prevX);

		 waitKey();
	 }
 }

 //lab 8
 float getMeanValue(int height, int width, int h[256])
 {
	 float mean = 0;
	 float M = height * width;
	 for (int g = 0; g < 256; g++)
		 mean += g * h[g];
	 return mean / M;
 }

 float getStandardDeviation(int h[256], float mean, int M)
 {
	 float stdDeviation = 0;
	 for (int g = 0; g < 256; g++)
		 stdDeviation += (g - mean) * (g - mean) * h[g];
	 stdDeviation /= M;
	 stdDeviation = sqrt(stdDeviation);
	 return stdDeviation;
 }
 void  testMeanStdDeviation()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;

		 int* h = calculateH(src);
		 float mean = getMeanValue(height, width, h);
		 float stdDeviation = getStandardDeviation(h, mean, height * width);

		 printf("%f %f", mean, stdDeviation);

		

		 waitKey();
	 }
 }
 float calculateMeanInRange(int start, int end, int h[256])
 {
	 float mean = 0;
	 int N = 0;
	 for (int g = start; g <= end; g++)
	 {
		 mean += g * h[g];
		 N += h[g];
	 }
	 mean /= N;
	 return mean;
	 
 }
 void testBasicGlobalThresholdingAlgorithm()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;

		 int* h = calculateH(src);
		 float mean = getMeanValue(height, width, h);
		 float stdDeviation = getStandardDeviation(h, mean, height * width);

		 int iMax = -1, iMin = 256;
		 for (int i = 0; i < 256; i++)
		 {
			 if (h[i] != 0 && i > iMax)
				 iMax = i;
			 if (h[i] != 0 && i < iMin)
				 iMin = i;
		 }
		 float currT = (float)(iMax + iMin) / (float)2;
		 float prevT = currT;

		 do
		 {
			 float mean1 = calculateMeanInRange(iMin, (int)currT, h);
			 float mean2 = calculateMeanInRange((int)currT + 1, iMax, h);
			 prevT = currT;
			 currT = (mean1 + mean2) / 2;
	
		 } while (abs(currT - prevT) >  0.1);

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 src.at<uchar>(i, j) = src.at<uchar>(i, j) < (int)currT ? 0 : 255;

		 imshow("basic thresholding", src);


		 waitKey();
	 }
 }	

 void testBrightnessChange(int offset)
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;

		 int* h = calculateH(src);
		 showHistogram("srcHistogram", h, 256, height);

		 float mean = getMeanValue(height, width, h);
		 float stdDeviation = getStandardDeviation(h, mean, height * width);

		 for(int i = 0 ; i < height ; i++)
			 for (int j = 0; j < width; j++)
			 {
				 if (src.at<uchar>(i, j) + offset < 0)
					 src.at<uchar>(i, j) = 0;
				 if (src.at<uchar>(i, j) + offset > 255)
					 src.at<uchar>(i, j) = 255;
				 else
					 src.at<uchar>(i, j) += offset;
			 }

		 h = calculateH(src);
		 showHistogram("slideHistogram", h, 256, height);
		 imshow("brightness change", src);


		 waitKey();
	 }
 }

 void testContrastChange(int gOutMax, int gOutMin)
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 int gInMin = 256;
		 int gInMax = -1;
		 int* h = calculateH(src);
		 showHistogram("srcHistogram", h, 256, height);

		 for (int i = 0; i < 256; i++)
		 {
			 if (h[i] != 0 && i > gInMax)
				 gInMax = i;
			 if (h[i] != 0 && i < gInMin)
				 gInMin = i;
		 }

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
			 {
				 src.at<uchar>(i, j) =  gOutMin + (src.at<uchar>(i, j) - gInMin) * ((float)(gOutMax - gOutMin) / (float)(gInMax - gInMin));
		
			 }

				
		 h = calculateH(src);
		 showHistogram("contrastChangeHistogram", h, 256, height);
		 imshow("contrast change", src);


		 waitKey();
	 }
 }

 void testGammaCorrection(float gamma)
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;

		 int* h = calculateH(src);
		 showHistogram("srcHistogram", h, 256, height);

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
			 {
				 uchar val = 255 * pow((float)src.at<uchar>(i, j) / (float)255, gamma);
				 if (val < 0)
					 val = 0;
				 if (val > 255)
					 val = 255;
				 src.at<uchar>(i, j) = val;
			 }

		 h = calculateH(src);
		 showHistogram("gammaCorrectionHistogram", h, 256, height);
		 imshow("gammaCorrectionImage ", src);


		 waitKey();
	 }
 }

 float* calculateCPDF(Mat src)
 {
	 float* PDF = calculatePDF(src);
	 float* CPDF = (float*)calloc(256, sizeof(float));
	 CPDF[0] = PDF[0];
	 for (int g = 1; g < 256; g++)
		 CPDF[g] = CPDF[g - 1] + PDF[g];
	 return CPDF;
	 
 }

 void testHistogramEqualization()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 int gInMin = 256;
		 int gInMax = -1;
		 
		 int* h = calculateH(src);
		 showHistogram("srcHistogram", h, 256, height);

		 float* CPDF = calculateCPDF(src);


		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
			 {
				 uchar val = 255 * CPDF[src.at<uchar>(i, j)];
				 if (val < 0)
					 val = 0;
				 if (val > 255)
					 val = 255;
				 src.at<uchar>(i, j) = val;
			 }


		 h = calculateH(src);
		 showHistogram("equalizedHistogram", h, 256, height);
		 imshow("equalizedHistogramImage", src);


		 waitKey();
	 }
 }

 //Lab 9

 Mat createConvolutionalCore3x3()
 {
	 Mat_<float> struct_elem(3, 3, (float)1);
	 struct_elem(0, 0) = 1;
	 struct_elem(0, 1) = 1;
	 struct_elem(0, 2) = 1;
	 struct_elem(1, 0) = 1;
	 struct_elem(1, 1) = 1;
	 struct_elem(1, 2) = 1;
	 struct_elem(2, 0) = 1;
	 struct_elem(2, 1) = 1;
	 struct_elem(2, 2) = 1;

	 return struct_elem;

 }
 Mat createConvolutionalCore5x5()
 {
	 return Mat_<float>(5,5, (float)1);
 }

 Mat createConvolutionalCoreGaussian()
 {
	 Mat_<float> struct_elem(3, 3, (float)1);
	 struct_elem(0, 0) = 1;
	 struct_elem(0, 1) = 2;
	 struct_elem(0, 2) = 1;
	 struct_elem(1, 0) = 2;
	 struct_elem(1, 1) = 4;
	 struct_elem(1, 2) = 2;
	 struct_elem(2, 0) = 1;
	 struct_elem(2, 1) = 2;
	 struct_elem(2, 2) = 1;

	 return struct_elem;
 }

 Mat createConvolutionalCoreLaplace()
 {
	 Mat_<float> struct_elem(3, 3, (float)1);
	 struct_elem(0, 0) = 0;
	 struct_elem(0, 1) = -1;
	 struct_elem(0, 2) = 0;
	 struct_elem(1, 0) = -1;
	 struct_elem(1, 1) = 4;
	 struct_elem(1, 2) = -1;
	 struct_elem(2, 0) = 0;
	 struct_elem(2, 1) = -1;
	 struct_elem(2, 2) = 0;

	 return struct_elem;
 }

 Mat_<float> createHighPassCore()
 {
	 Mat_<float> struct_elem(3, 3, (float)1);
	 struct_elem(0, 1) = -1;
	 struct_elem(0, 0) = -1;
	 struct_elem(0, 2) = -1;
	 struct_elem(1, 0) = -1;
	 struct_elem(1, 1) = 9;
	 struct_elem(1, 2) = -1;
	 struct_elem(2, 0) = -1;
	 struct_elem(2, 1) = -1;
	 struct_elem(2, 2) = -1;
	 return struct_elem;
 }

 Mat findConvolutionalMat(Mat src, bool is3x3, bool isHighPass, bool isGaussian, bool isLaplace)
 {
	 Mat_<float> core;
	 Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	 if (is3x3)
	 {
		 if (isHighPass)
		 {
			 core = createHighPassCore();
		 }
		 else
			 core = createConvolutionalCore3x3();
	 }
	 else
		 core = createConvolutionalCore5x5();
	 if (isGaussian)
	 {
		 core = createConvolutionalCoreGaussian();
	 }
	 if (isLaplace)
	 {
		 core = createConvolutionalCoreLaplace();
	 }
	
	 for (int i = 0; i < src.rows; i++)
		 for (int j = 0; j < src.cols; j++)
		 {	
			 float sum = 0;
			 for (int ci = 0; ci < core.rows; ci++)
				 for (int cj = 0; cj < core.cols; cj++)
				 {
					 int nextI = i - core.cols / 2 + ci;
					 int nextJ = j - core.cols / 2 + cj;
					 if (isInImage(src.rows, src.cols, nextI, nextJ))
						 sum += (float)src.at<uchar>(nextI, nextJ) * core(ci, cj);
				 }
			 if (is3x3)
			 {
				 
				 if (isHighPass || isLaplace)
				 {		
					 
						 if (sum > 255)
							 dst.at<uchar>(i, j) = 255;
						 else if (sum < 0)
							 dst.at<uchar>(i, j) = 0;
						 else
							 dst.at<uchar>(i, j) = (int)sum;
				 }
				 else 
				 {
					 
					 if (isGaussian)
						 dst.at<uchar>(i, j) = (int)sum / 16;
					 else
						 dst.at<uchar>(i, j) = (int)sum / 9;
				 }
				 
			 }
			 else
			 {
				 dst.at<uchar>(i, j) = (int)sum / 25;
			 }
		 }

	 return dst;
 }
 void testMeanFilter3x3()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 Mat core = createConvolutionalCore3x3();
		 imshow("src", src);
		 int height = src.rows;
		 int width = src.cols;
		 Mat convolutionalMat = findConvolutionalMat(src, true,false,false,false);
		 
		

		 imshow("meanfilter3x3", convolutionalMat);
		 waitKey();
	 }
 }

 void testMeanFilter5x5()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 Mat filter5x5 = findConvolutionalMat(src, false,false,false,false);


		 

		 imshow("meanfilter5x5", filter5x5);
		 waitKey();
	 }
 }

 void testGaussianFilter()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 Mat gaussian = findConvolutionalMat(src, true,false,true,false);


		 imshow("gaussianFilter", gaussian);


		 waitKey();
	 }
 }

 void testLaplaceFilter()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;	
		 Mat laplace = findConvolutionalMat(src, true, true, false, true);


		 imshow("LaplaceFilter", laplace);


		 waitKey();
	 }
 }

 void testHighPassFilter()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 Mat highpass = findConvolutionalMat(src, true,true,false,false);


		 imshow("highPassFilter", highpass);


		 waitKey();
	 }
 }

void centering_transform(Mat_<float> img)
 {
	 for (int i = 0; i < img.rows; i++)
		 for (int j = 0; j < img.cols; j++)
			 img(i, j) = ((i + j) & 1) ? -img(i, j) : img(i, j);

 }

 Mat generic_frequency_domain_filter(Mat src, bool isFourierSpectrum, bool isIdeal, bool isLowPass, bool isCutLow)
 {
	 Mat srcf;
	 src.convertTo(srcf, CV_32FC1);
	  centering_transform(srcf);
	 
	 Mat fourier;
	 dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	 Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	 split(fourier, channels);

	 Mat mag, phi;
	 magnitude(channels[0], channels[1], mag);
	 phase(channels[0], channels[1], phi);

	 float maxLog = 0;
	 for (int i = 0; i < mag.rows; i++)
		 for (int j = 0; j < mag.cols; j++)
			 if (log(mag.at<float>(i, j) + 1) > maxLog)
				 maxLog = log(mag.at<float>(i, j) + 1);

	 Mat normalizedMagnitude = Mat(mag.rows, mag.cols, CV_8UC1, Scalar(255));

	for (int i = 0; i < mag.rows; i++)
		for (int j = 0; j < mag.cols; j++)
			normalizedMagnitude.at<uchar>(i, j) = log(mag.at<float>(i, j) + 1) * 255 / maxLog;
	
	if (isFourierSpectrum)
		return normalizedMagnitude;

	 if (isIdeal)
	 {
		 int H = fourier.rows;
		 int W = fourier.cols;
		 int R = 10;

		 Mat idealLowChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		 Mat idealHighChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		 for (int i = 0; i < H; i++)
			 for (int j = 0; j < W; j++) {
				 float check = (i - H / 2) * (i - H / 2) + (j - W / 2) * (j - W / 2);
				 if (check > R * R) { 
					 idealLowChannels[0].at<float>(i, j) = 0;
					 idealLowChannels[1].at<float>(i, j) = 0;

					 idealHighChannels[0].at<float>(i, j) = channels[0].at<float>(i, j);
					 idealHighChannels[1].at<float>(i, j) = channels[1].at<float>(i, j);
				 }
				 else {
					 idealLowChannels[0].at<float>(i, j) = channels[0].at<float>(i, j);
					 idealLowChannels[1].at<float>(i, j) = channels[1].at<float>(i, j);

					 idealHighChannels[0].at<float>(i, j) = 0;
					 idealHighChannels[1].at<float>(i, j) = 0;
				 }
			 }

		 if (isLowPass)
		 {
			 Mat idealLowDst, idealLowDst_f;
			 merge(idealLowChannels, 2, fourier);
			 dft(fourier, idealLowDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
			 centering_transform(idealLowDst_f);
			 normalize(idealLowDst_f, idealLowDst, 0, 255, NORM_MINMAX, CV_8UC1);
			 return idealLowDst;
		 }
		 else
		 {
			 Mat idealHighDst, idealHighDst_f;
			 merge(idealHighChannels, 2, fourier);
			 dft(fourier, idealHighDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
			 centering_transform(idealHighDst_f);
			 normalize(idealHighDst_f, idealHighDst, 0, 255, NORM_MINMAX, CV_8UC1);
			 return idealHighDst;
		 }

		
	 }
	 else
	 {
		 std::cout << "here" << std::endl;
		 int H = fourier.rows;
		 int W = fourier.cols;
		 int height = src.rows;
		 int width = src.cols;

		 Mat gaussLowChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		 Mat gaussHighChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		 for (int i = 0; i < H; i++)
			 for (int j = 0; j < W; j++) {
				 gaussLowChannels[0].at<float>(i, j) = channels[0].at<float>(i, j) * exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100);
				 gaussLowChannels[1].at<float>(i, j) = channels[1].at<float>(i, j) * exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100);

				 gaussHighChannels[0].at<float>(i, j) = channels[0].at<float>(i, j) * (1 - exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100));
				 gaussHighChannels[1].at<float>(i, j) = channels[1].at<float>(i, j) * (1 - exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100));
			 }

		 if (isCutLow)
		 {
			 std::cout << "here1" << std::endl;
			 Mat gaussLowDst, gaussLowDst_f;
			 merge(gaussLowChannels, 2, fourier);
			 dft(fourier, gaussLowDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
			 centering_transform(gaussLowDst_f);
			 normalize(gaussLowDst_f, gaussLowDst, 0, 255, NORM_MINMAX, CV_8UC1);
			 return gaussLowDst;
		}
		 else
		 {
			 std::cout << "here2" << std::endl;
			 Mat gaussHighDst, gaussHighDst_f;
			 merge(gaussHighChannels, 2, fourier);
			 dft(fourier, gaussHighDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
			 centering_transform(gaussHighDst_f);
			 normalize(gaussHighDst_f, gaussHighDst, 0, 255, NORM_MINMAX, CV_8UC1);
			 return gaussHighDst;
		 }

		
	 }

 }

 void frequencyDomainFilter() {
	 char fileName[MAX_PATH];

	 while (openFileDlg(fileName)) {
		 Mat src = imread(fileName, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;

		 Mat src_f;
		 src.convertTo(src_f, CV_32FC1);

		 centering_transform(src_f);

		 Mat fourier;
		 dft(src_f, fourier, DFT_COMPLEX_OUTPUT);

		 Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
		 split(fourier, channels);

		 Mat mag, phi;
		 magnitude(channels[0], channels[1], mag);
		 phase(channels[0], channels[1], phi);

		 float maxLog = 0.0;
		 for (int i = 0; i < mag.rows; i++)
			 for (int j = 0; j < mag.cols; j++) {
				 int v = log(mag.at<float>(i, j) + 1);
				 maxLog = (v > maxLog) ? v : maxLog;
			 }

		 Mat normalizedMagnitude(mag.rows, mag.cols, CV_8UC1);
		 for (int i = 0; i < mag.rows; i++)
			 for (int j = 0; j < mag.cols; j++) {
				 float v = log(mag.at<float>(i, j) + 1);
				 normalizedMagnitude.at<uchar>(i, j) = v * 255 / maxLog;
			 }

		 imshow("Initial Image", src);
		 imshow("Magnitude Image", normalizedMagnitude);

		 int H = fourier.rows;
		 int W = fourier.cols;
		 Mat idealLowChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		 Mat idealHighChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		 for (int i = 0; i < H; i++)
			 for (int j = 0; j < W; j++) {
				 float check = (i - H / 2) * (i - H / 2) + (j - W / 2) * (j - W / 2);
				 if (check > 100) { // R = 10 
					 idealLowChannels[0].at<float>(i, j) = 0;
					 idealLowChannels[1].at<float>(i, j) = 0;

					 idealHighChannels[0].at<float>(i, j) = channels[0].at<float>(i, j);
					 idealHighChannels[1].at<float>(i, j) = channels[1].at<float>(i, j);
				 }
				 else {
					 idealLowChannels[0].at<float>(i, j) = channels[0].at<float>(i, j);
					 idealLowChannels[1].at<float>(i, j) = channels[1].at<float>(i, j);

					 idealHighChannels[0].at<float>(i, j) = 0;
					 idealHighChannels[1].at<float>(i, j) = 0;
				 }
			 }

		 Mat idealLowDst, idealLowDst_f;
		 merge(idealLowChannels, 2, fourier);
		 dft(fourier, idealLowDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		 centering_transform(idealLowDst_f);
		 normalize(idealLowDst_f, idealLowDst, 0, 255, NORM_MINMAX, CV_8UC1);
		 imshow("Ideal Low Filter Image", idealLowDst);

		 Mat idealHighDst, idealHighDst_f;
		 merge(idealHighChannels, 2, fourier);
		 dft(fourier, idealHighDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		 centering_transform(idealHighDst_f);
		 normalize(idealHighDst_f, idealHighDst, 0, 255, NORM_MINMAX, CV_8UC1);
		 imshow("Ideal High Filter Image", idealHighDst);

		 Mat gaussLowChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		 Mat gaussHighChannels[] = { Mat::zeros(fourier.size(), CV_32F), Mat::zeros(fourier.size(), CV_32F) };
		 for (int i = 0; i < H; i++)
			 for (int j = 0; j < W; j++) {
				 gaussLowChannels[0].at<float>(i, j) = channels[0].at<float>(i, j) * exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100);
				 gaussLowChannels[1].at<float>(i, j) = channels[1].at<float>(i, j) * exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100);

				 gaussHighChannels[0].at<float>(i, j) = channels[0].at<float>(i, j) * (1 - exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100));
				 gaussHighChannels[1].at<float>(i, j) = channels[1].at<float>(i, j) * (1 - exp((-(pow(height / 2 - i, 2) + pow(width / 2 - j, 2))) / 100));
			 }

		 Mat gaussLowDst, gaussLowDst_f;
		 merge(gaussLowChannels, 2, fourier);
		 dft(fourier, gaussLowDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		 centering_transform(gaussLowDst_f);
		 normalize(gaussLowDst_f, gaussLowDst, 0, 255, NORM_MINMAX, CV_8UC1);
		 imshow("Gaussian-Cut Low Filter Image", gaussLowDst);

		 Mat gaussHighDst, gaussHighDst_f;
		 merge(gaussHighChannels, 2, fourier);
		 dft(fourier, gaussHighDst_f, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
		 centering_transform(gaussHighDst_f);
		 normalize(gaussHighDst_f, gaussHighDst, 0, 255, NORM_MINMAX, CV_8UC1);
		 imshow("Gaussian-Cut High Filter Image", gaussHighDst);

		 waitKey(0);
	 }
 }

 void testLogMagnitudeFourierSpectrum()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 Mat logMagnitude = generic_frequency_domain_filter(src, true, false, false,false);


		 imshow("logMagnitudeFourierSpectrum", logMagnitude);


		 waitKey();
	 }
 }

 void testIdealLowPassFilter()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = generic_frequency_domain_filter(src, false, true, true,false);
		 imshow("IdealLowPassFilter", dst);

		 waitKey();
	 }
 }

 void testIdealHighPassFilter()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = generic_frequency_domain_filter(src, false, true, false,false);
		 imshow("idealHighPassFilter", dst);
		 waitKey();
	 }
 }

 void testGaussianCutLPF()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = generic_frequency_domain_filter(src, false, false, false, true);


		 imshow("gaussianCutLPF", dst);


		 waitKey();
	 }

 }

 void testGaussianCutHPF()
 {
	 char filename[MAX_PATH];

	 while (openFileDlg(filename))
	 {
		 Mat src = imread(filename, IMREAD_GRAYSCALE);
		 imshow("src", src);

		 int height = src.rows;
		 int width = src.cols;
		 Mat dst = generic_frequency_domain_filter(src, false, false, false, false);

		 imshow("gaussianCutHPF", dst);


		 waitKey();
	 }
 }

 
 //lab 10
 Mat_<float> getConvolution(Mat_<uchar> img, Mat_<float> H) {
	 int height = img.rows;
	 int width = img.cols;

	 int filterSize_rows = H.rows;
	 int filterSize_cols = H.cols;

	 Mat_<float> convolution(height, width);

	 for (int i = 0; i < height; i++)
		 for (int j = 0; j < width; j++) {
			 float sum = 0;

			 for (int ii = 0; ii < filterSize_rows; ii++)
				 for (int jj = 0; jj < filterSize_cols; jj++) {
					 int neighbour_i = i + ii - filterSize_rows / 2;
					 int neighbour_j = j + jj - filterSize_cols / 2;
					 if (isInImage(height, width, neighbour_i, neighbour_j))
						 sum += H(ii, jj) * img(neighbour_i, neighbour_j);
				 }

			 convolution(i, j) = sum;
		 }

	 return convolution;
 }


 void test_salt_peper(int w) {
	 char fileName[MAX_PATH];

	 while (openFileDlg(fileName)) {
		 Mat src = imread(fileName, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;

		 Mat dst(height, width, CV_8UC1);
		 std::vector<uchar> v;

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++) {
				 v.clear();

				 for (int k = 0; k < w; k++)
					 for (int l = 0; l < w; l++) {
						 int nextI = i - (w / 2) + k;
						 int nextJ = j - (w / 2) + l;

						 if (isInImage(height, width, nextI, nextJ))
							 v.push_back(src.at<uchar>(nextI, nextJ));
					 }

				 std::sort(v.begin(), v.end());
				 dst.at<uchar>(i, j) = v.at(v.size() / 2);
			 }

		 imshow("Initial Image", src);
		 imshow("SaltPepper", dst);

		 waitKey(0);
	 }
 }

 void test_gaussian_1(float sigma) {
	 char fileName[MAX_PATH];

	 while (openFileDlg(fileName)) {
		 Mat src = imread(fileName, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;

		 Mat dst(height, width, CV_8UC1);

		 float aux = 6 * sigma;
		 int w = aux + 0.5;
		 if (w % 2 == 0)
			 w++;

		 Mat gauss(w, w, CV_32F);
		 float sum = 0;
		 for (int i = 0; i < w; i++)
			 for (int j = 0; j < w; j++) {
				 gauss.at<float>(i, j) = ((1 / (2 * CV_PI * sigma * sigma)) * exp(-(pow((i - (w / 2)), 2) + pow((j - (w / 2)), 2)) / (2 * sigma * sigma)));
				 sum += gauss.at<float>(i, j);
			 }

		 printf("Gauss 1 Sum: %f\n", sum);

		 double t = (double)getTickCount();

		 Mat convolution = getConvolution(src, gauss);

		 t = ((double)getTickCount() - t) / getTickFrequency();

		 printf("Time: %.3f [ms]\n", t * 1000);
		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 dst.at<uchar>(i, j) = convolution.at<float>(i, j) / sum;

		 imshow("Initial Image", src);
		 imshow("Gauss 1", dst);

		 waitKey(0);
	 }
 }

 Mat test_gaussian_2(float sigma) {
	 char fileName[MAX_PATH];

	 while (openFileDlg(fileName)) {
		 Mat src = imread(fileName, IMREAD_GRAYSCALE);

		 int height = src.rows;
		 int width = src.cols;

		 Mat dst(height, width, CV_8UC1);

		 float aux = 6 * sigma;
		 int w = aux + 0.5;
		 if (w % 2 == 0)
			 w++;

		 Mat gauss(w, w, CV_32F);
		 float sum = 0;
		 for (int i = 0; i < w; i++)
			 for (int j = 0; j < w; j++) {
				 gauss.at<float>(i, j) = ((1 / (2 * CV_PI * sigma * sigma)) * exp(-(pow((i - (w / 2)), 2) + pow((j - (w / 2)), 2)) / (2 * sigma * sigma)));
				 sum += gauss.at<float>(i, j);
			 }

		 printf("Gauss 2 Sum: %f\n", sum);

		 Mat gauss_1d = Mat(w, 1, CV_32F);
		 Mat gauss_2d = Mat(1, w, CV_32F);
		 float sum_1d = 0;
		 float sum_2d = 0;
		 for (int i = 0; i < w; i++) {
			 gauss_1d.at<float>(i, 0) = gauss.at<float>(i, (w / 2));
			 sum_1d += gauss_1d.at<float>(i, 0);
		 }

		 for (int j = 0; j < w; j++) {
			 gauss_2d.at<float>(0, j) = gauss.at<float>((w / 2), j);
			 sum_2d += gauss_2d.at<float>(0, j);
		 }

		 double t = (double)getTickCount();
		 Mat conv = getConvolution(src, gauss_1d);
		 t = ((double)getTickCount() - t) / getTickFrequency();

		 Mat tmp = Mat(height, width, CV_32F);
		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 tmp.at<float>(i, j) = conv.at<float>(i, j) / sum_1d;

		 double t2 = (double)getTickCount();
		 Mat conv2 = getConvolution(tmp, gauss_2d);
		 t2 = ((double)getTickCount() - t2) / getTickFrequency();

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 dst.at<uchar>(i, j) = conv2.at<float>(i, j) / sum_2d;


		 printf("Time=%.3f [ms]\n", (t + t2) * 1000);

		 imshow("Initial Image", src);
		 imshow("Gauss 2", dst);
		 return dst;

		 waitKey(0);
	 }
 }


 // lab 11
 Mat_<float> getConvolutionSx()
 {
	 Mat_<float> Sx(3, 3);
	
	 Sx(0, 0) = -1;
	 Sx(0, 1) = 0;
	 Sx(0, 2) = 1;
	 Sx(1, 0) = -2;
	 Sx(1, 1) = 0;
	 Sx(1, 2) = 2;
	 Sx(2, 0) = -1;
	 Sx(2, 1) = 0;
	 Sx(2, 2) = 1;

	 return Sx;

 }

 Mat_<float> getConvolutionSy()
 {
	 Mat_<float> Sy(3, 3);

	 Sy(0, 0) = 1;
	 Sy(0, 1) = 2;
	 Sy(0, 2) = 1;
	 Sy(1, 0) = 0;
	 Sy(1, 1) = 0;
	 Sy(1, 2) = 0;
	 Sy(2, 0) = -1;
	 Sy(2, 1) = -2;
	 Sy(2, 2) = -1;

	 return Sy;

 }

 void testEdgeCannyComplete()
 {
	 //Step 1 - Noise filtering
	 Mat gaussianFilteredImage = test_gaussian_2(0.5);

	 //Step 2 - Gradient magnitude orientation
	 Mat_<float> Sx = getConvolutionSx();
	 Mat_<float> Sy = getConvolutionSy();
	 Mat_<float> G(gaussianFilteredImage.rows, gaussianFilteredImage.cols);
	 Mat_<float> phi(gaussianFilteredImage.rows, gaussianFilteredImage.cols);
	 Mat_<uchar> normalizedG(gaussianFilteredImage.rows, gaussianFilteredImage.cols);

	 for (int i = 0; i < gaussianFilteredImage.rows; i++)
		 for (int j = 0; j < gaussianFilteredImage.cols; j++)
		 {
			 float sumX = 0;
			 float sumY = 0;
			 for (int ci = 0; ci < Sx.rows; ci++)
				 for (int cj = 0; cj < Sx.cols; cj++)
				 {
					 int nextI = i - Sx.cols / 2 + ci;
					 int nextJ = j - Sx.cols / 2 + cj;
					 if (isInImage(gaussianFilteredImage.rows, gaussianFilteredImage.cols, nextI, nextJ))
					 {
						 sumX += (float)gaussianFilteredImage.at<uchar>(nextI, nextJ) * Sx(ci, cj);
						 sumY += (float)gaussianFilteredImage.at<uchar>(nextI, nextJ) * Sy(ci, cj);
					 }

				 }
			 G(i, j) = sqrt(pow(sumX, 2) + pow(sumY, 2)) / ( 4 * sqrt(2));
			 normalizedG(i, j) = G(i, j);
			 phi(i, j) = atan2(sumY, sumX);
		 }

	 imshow("normalizedGradient", normalizedG);

		 int height = G.rows;
		 int width = G.cols;
		 Mat_<uchar> nonMaximaSuppresion(height, width);
		 Mat_<float> angles(height, width);

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
			 {
				 angles(i, j) = phi(i, j) * 180.0 / CV_PI;
				 angles(i, j) = angles(i, j) < 0 ? angles(i, j) + 180.0 : angles(i, j);
				
			 }

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
			 {
				 float q = 0;
				 float r = 0;

				 if ((( (0 <= angles(i, j)) && (angles(i,j) < 22.5)) || ((157.5 <= angles(i, j)) )) && isInImage(height, width, i, j + 1) && isInImage(height, width, i, j - 1))
				 {
					 //zone 0
					 q = G(i, j + 1);
					 r = G(i, j - 1);
				 }
				 else if ( ((22.5 <= angles(i, j)) && (angles(i, j) < 67.5)) && isInImage(height, width, i + 1, j - 1) && isInImage(height, width, i - 1, j + 1))
				 {
					 //zone 1
					 q = G(i + 1, j - 1);
					 r = G(i - 1, j + 1);
				 }

				 else if ( (67.5 <= angles(i, j)) && (angles(i , j) < 112.5) && isInImage(height, width, i + 1, j) && isInImage(height, width, i - 1, j))
				 {
					 //zone 2
					 q = G(i + 1, j);
					 r = G(i - 1, j);
				 }

				 else if ( (112.5 <= angles(i, j)) && (angles(i, j) < 157.5) && isInImage(height, width, i - 1, j - 1) && isInImage(height, width, i + 1, j + 1))
				 {
					 //zone 3
					 q = G(i - 1, j - 1);
					 r = G(i + 1, j + 1);
				 }

				 if ((G(i, j) >= q) && (G(i, j) >= r))
					 nonMaximaSuppresion(i, j) = (uchar)G(i, j);
				 else
					 nonMaximaSuppresion(i, j) = 0;
			 }

		 imshow("nonMaximaSuppresion", nonMaximaSuppresion);


		 //adaptive thresholding
		 float p = 0.1; // parameter -> chosen by us
		 int histogram[256] = { 0 };

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 histogram[nonMaximaSuppresion(i, j)]++;

		 for (int i = 0; i < 256; i++)
			 printf("histogram[%d] = %d\n", i, histogram[i]);

		 int no_edge_pixels = p * (height * width - histogram[0]);

		 float k = 0.4; // parameter -> chosen by us
		 int partial_no_edge_pixels = 0;
		 int t_high = 0;

		 for (int i = 255; i > 0; i--)
			 if (partial_no_edge_pixels < no_edge_pixels)
			 {
				 partial_no_edge_pixels += histogram[i];
				 t_high = i;
			 }

		 int t_low = t_high * k;


		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 if (nonMaximaSuppresion(i, j) < t_low)
					 nonMaximaSuppresion(i, j) = 0;
				 else if (nonMaximaSuppresion(i, j) > t_high)
					 nonMaximaSuppresion(i, j) = 255;
				 else
					 nonMaximaSuppresion(i, j) = 127;

		 imshow("adaptive thresholding", nonMaximaSuppresion);


		 for (int i = 0; i < height; i++)
		 {
			 for (int j = 0; j < width; j++)
			 {
				 if (nonMaximaSuppresion(i, j) == 255 )
				 {
					 std::queue<Point> Q;
					 Point point = Point(j, i);
					 Q.push(point);
					 while (!Q.empty())
					 {
						 Point q = Q.front();
						 Q.pop();
						 for (int k = 0; k < 8; k++)
						 {
							 int nextX = q.x + n8DX[k];
							 int nextY = q.y + n8DY[k];
							 if (nextY >= 0 && nextY < nonMaximaSuppresion.rows && nextX >= 0 && nextX < nonMaximaSuppresion.cols)
							 {
								 if (nonMaximaSuppresion(nextY, nextX) == 127)
								 {	
									 nonMaximaSuppresion(nextY, nextX) = 255;
									 Q.push(Point(nextX, nextY));
								 }
							 }
						 }
					 }

				 }
			 }

		 }

		 for (int i = 0; i < height; i++)
			 for (int j = 0; j < width; j++)
				 if (nonMaximaSuppresion(i, j) == 127)
					 nonMaximaSuppresion(i, j) = 0;


		 imshow("final edge canny", nonMaximaSuppresion);

			






	//Step 3 - non maxima suppresion
	 waitKey(0);
 }


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		//Lab 1
		printf(" 10 - Change gray level -- Lab 1\n");
		printf(" 11 - Four colored quarters -- Lab 1\n");
		printf(" 12 - Inverse Matrix -- Lab 1\n");
		//Lab 2
		printf(" 13 - RGB 24 Split Channels -- Lab 2\n");
		printf(" 14 - RGB 24 to GrayScale -- Lab 2\n");
		printf(" 15 - GrayScale to Black or White -- Lab 2\n");
		printf(" 16 - RGB to HSV -- Lab 2\n");
		//Lab 3
		printf(" 17 - Show Histogram-- Lab 3\n");
		printf(" 18 - Multilevel Thresholding -- Lab 3\n");
		printf(" 19 - Algoritm pe care nu l-am facut -- lab 3\n");
		//Lab 4
		printf(" 20 - GometricalFeatures -- Lab 4\n");
		//Lab 5
		printf(" 21 - BFS Labeling Algorithm - Lab 5\n");
		printf(" 22 - Two pass algorithm - Lab 5\n");
		//Lab 6
		printf(" 23 - Border Tracing Algorithm - Lab 6\n");
		printf(" 24 - Reconstruct - Lab 6 \n");
		//Lab 7
		printf(" 25 - Dilation - Lab 7 \n");
		printf(" 26 - Erosion - Lab 7 \n");
		printf(" 27 - Opening - Lab 7 \n");
		printf(" 28 - Closing - Lab 7 \n");
		printf(" 29 - Boundary Extraction - Lab 7 \n");
		printf(" 30 - Region Filling - Lab 7 \n");
		//lab 8
		printf(" 31 - Mean std deviation - Lab 8 \n");
		printf(" 32 - Basic global thresholding algorithm- Lab 8 \n");
		printf(" 33 - Brightness change - Lab 8 \n");
		printf(" 34 - Contrast change - Lab 8 \n");
		printf(" 35 - Gamma corection - Lab 8 \n");
		printf(" 36 - Histogram equalization - Lab 8 \n");
		//lab 9
		printf(" 37 - Mean filter 3x3 - Lab 9 \n");
		printf(" 38 - Mean filter 5x5 - Lab 9 \n");
		printf(" 39 - Gaussian filter 3x3 - Lab 9 \n");
		printf(" 40 - LaPlace filter 3x3 - Lab 9 \n");
		printf(" 41 - High-pass filter 3x3 - Lab 9 \n");
		printf(" 42 - Log Magnitude Fourier Spectrum - Lab 9 \n");
		printf(" 43 - Ideal low-pass filter - Lab 9 \n");
		printf(" 44 - Ideal high-pass filter - Lab 9 \n");
		printf(" 45 - Gaussian-cut lpf - Lab 9 \n");
		printf(" 46 - Gaussian-cut hpf - Lab 9 \n");

		//lab 10
		printf(" 47 - Salt and Pepper - Lab 10 \n");
		printf(" 48 - Gaussian 1 - Lab 10 \n");
		printf(" 49 - Gaussian 2 - Lab 10 \n");

		//lab 11
		printf(" 50 - Canny Edge Detection  - Complete \n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				testChangeGrayLevel();
				break;
			case 11: 
				testFourColoredQuarters();
				break;
			case 12:
				testInverseMatrix();
				break;
			case 13:
				testRGB24Split();
				break;
			case 14:
				testRgb24ToGrayScale();
				break;
			case 15:
				while (true)
				{
				
					printf("Please enter a threshold value!\n");
					int thresholdValue;
					scanf("%d", &thresholdValue);
					testConvertingGrayScaleToBlackOrWhite(thresholdValue);
					break;
				}
				break;
			case 16:
				testRGBtoHSV();
				break;
			case 17:
				testShowHistogram();
				break;
			case 18:
				testMultilevelThresholding();
				break;
			case 19: 
				break;
			case 20:
				testGeometricalFeatures();
				break;
			case 21:
				testBFSLabelingAlgorithm();
				break;
			case 22:
				testTwoPassAlgorithm();
				break;
			case 23:
				testBorderTracingAlgorithm();
				break;
			case 24:
				testContourReconstruction();
				break;
			case 25:
				printf("Enter number of times to apply Dilation\n");
				int nrDelation;
				scanf("%d", &nrDelation);
				testDilation(nrDelation);
				break;
			case 26:
				printf("Enter number of times to apply Erosion\n");
				int nrErosion;
				scanf("%d", &nrErosion);
				testErosion(nrErosion);
				break;
			case 27:
				testOpening();
				break;
			case 28:
				testClosing();
				break;
			case 29:
				testBoundaryExtraction();
				break;
			case 30:
				testRegionFilling();
				break;
			case 31:
				testMeanStdDeviation();
				break;
			case 32:
				testBasicGlobalThresholdingAlgorithm();
				break;
			case 33:
				printf("Enter brightness offset\n");
				int offset;
				scanf("%d", &offset);
				testBrightnessChange(offset);
				break;
			case 34:
				int gOutMin, gOutMax;
				printf("Enter g out min\n");
				scanf("%d", &gOutMin);
				printf("Enter g out max\n");
				scanf("%d", &gOutMax);
				testContrastChange(gOutMax, gOutMin);
				break;
			case 35:
				float gamma;
				printf("Enter gamma \n");
				scanf("%f", &gamma);
				testGammaCorrection(gamma);
				break;
			case 36:
				testHistogramEqualization();
				break;
			case 37:
				testMeanFilter3x3();
				break;
			case 38:
				testMeanFilter5x5();
				break;
			case 39:
				testGaussianFilter();
				break;
			case 40:
				testLaplaceFilter();
				break;
			case 41:
				testHighPassFilter();
				break;
			case 42:
				testLogMagnitudeFourierSpectrum();
				break;
			case 43:
				testIdealLowPassFilter();
				break;
			case 44:
				testIdealHighPassFilter();
				break;
			case 45:
				testGaussianCutLPF();
				break;
			case 46:
				testGaussianCutHPF();
				break;
			case 47:
				int n;
				printf("Enter n value\n");
				scanf("%d", &n);
				test_salt_peper(n);
				break;
			case 48:
				float g;
				printf("Enter g value\n");
				scanf("%f", &g);
				test_gaussian_1(g);
				waitKey(0);
				break;
			case 49:
				float g1;
				printf("Enter g value\n");
				scanf("%f", &g1);
				test_gaussian_2(g1);
				waitKey(0);
				break;
			case 50:
				testEdgeCannyComplete();
				break;
	}
	}
	while (op!=0);
	return 0;
}