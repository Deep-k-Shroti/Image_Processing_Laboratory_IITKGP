#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <conio.h>
#include <math.h>


#define LEVEL 256
#define INTENSITY_MAX 255
#define INTENSITY_MIN 0

using namespace std;
using namespace cv;


// This function takes image object, histogram and size
// Function reads the image and creates a histogram.
void imageToHistogram(Mat image,float histogram[], int size){

      for(int i = 0; i < LEVEL ; i++){
            histogram[i] = 0;
      }

      for(int y = 0 ; y < image.rows ; y ++){
            for(int x = 0 ; x < image.cols ; x++){
                  histogram[(int)image.at<uchar>(y,x)]++;
            }
      }

      for(int i = 0; i < LEVEL ; i++){
            histogram[i] = histogram[i]/size;
      }

      return;

}

// This fucnction is used to calculate the transfer function of from a given
// histogram. The transfer function created is just the cumulative frequency distribution
void calTranFunc(float histogram[], float tranFunc[]){

      tranFunc[0] = histogram[0];
      for(int i = 1 ; i < LEVEL; i++){
            tranFunc[i] = histogram[i] + tranFunc[i - 1];
            // histogram[i] += histogram[i-1];
      }
      return;
}

//This funtion prints the histogram on the console
void printHistogram(float histogram[]){
      for(int i = 0 ; i < LEVEL ; i++){
            cout << histogram[i] << ",";
      }
      cout << endl;
      return ;
}

// THis function is used to map the histogram to the intensity values that will be displayed on the image
void intensityMapping(float tranFunc[], float histogram[]){
      float tranFuncMin = INTENSITY_MAX+1;
      for(int i = 0; i < LEVEL ; i++){
            if(tranFuncMin > tranFunc[i]){
                  tranFuncMin = tranFunc[i];
            }
      }

      for(int i = 0; i < LEVEL ; i++){
            histogram[i] = (((tranFunc[i] - tranFuncMin)/(1 - tranFuncMin))*(LEVEL - 1) + 0.5);
      }

      return;
}


//Function to convert an Red Grenn Blue(RGB) space to Hue Saturation Intensity(HSI) space
void convert_RGB_To_HSI(Mat inputImage, Mat inputImageHSI, float **H, float **S, float **I){
      double r, g, b, h, s, in;

      for(int i = 0; i < inputImage.rows; i++)
      {
            for(int j = 0; j < inputImage.cols; j++)
            {

                  b = inputImage.at<Vec3b>(i, j)[0];
                  g = inputImage.at<Vec3b>(i, j)[1];
                  r = inputImage.at<Vec3b>(i, j)[2];



                  float min_val = 0.0;
                  min_val = min(r, min(b,g));
                  s = 1 - 3*(min_val/(b + g + r));

                  in = (b + g + r) / 3; // TO SEE

                  if(s < 0.00001)
                  {
                        s = 0.0;
                  }else if(s > 0.99999){
                        s = 1.0;
                  }

                  if(s != 0.0)
                  {
                  h = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g)*(r - g)) + ((r - b)*(g - b)));
                  h = acos(h);

                  if(b <= g)
                  {
                        h = h;
                  } else{
                        h = ((360 * 3.14159265) / 180.0) - h;
                  }
            }else{
                  h = 0.0;
            }

                  inputImageHSI.at<Vec3b>(i, j)[0] = H[i][j] = (h * 180) / 3.14159265;
                  inputImageHSI.at<Vec3b>(i, j)[1] = S[i][j] =  s*100;
                  inputImageHSI.at<Vec3b>(i, j)[2] = I[i][j] = in;

            }
      }
      return ;
}

//Function to convert an  Hue Saturation Intensity(HSI) space to Red Grenn Blue(RGB) space
void convert_HSI_To_RGB(Mat outputImage , Mat inputImageHSI, float **H ,float **S, float **I){
      float r, g, b, h, s, in;

      for(int i = 0; i < inputImageHSI.rows; i++){
            for(int j = 0; j < inputImageHSI.cols; j++){

                  h = H[i][j];
                  s = S[i][j]/100;
                  in = I[i][j];

                  if( h >= 0.0 && h < 120.0){
                        b = in*(1 - s);
                        r = in*(1 + (s*cos(h*3.14159265/180.0)/cos((60 - h)*3.14159265/180.0)));
                        g = 3*in - (r + b);
                  }else if( h >= 120.0 && h < 240.0){
                        h = h - 120;
                        r = in*(1 - s);
                        g = in*(1 + (s*cos(h*3.14159265/180.0)/cos((60 - h)*3.14159265/180.0)));
                        b = 3*in - (r + g);
                  }else{
                        h = h - 240;
                        g = in*(1 - s);
                        b = in*(1 + (s*cos(h*3.14159265/180.0)/cos((60 - h)*3.14159265/180.0)));
                        r = 3*in - (g + b);
                  }

                  if(b < INTENSITY_MIN)
                        b = INTENSITY_MIN;
                  if(b > INTENSITY_MAX)
                        b = INTENSITY_MAX;

                  if(g < INTENSITY_MIN)
                        g = INTENSITY_MIN;
                  if(g > INTENSITY_MAX)
                        g = INTENSITY_MAX;

                  if(r < INTENSITY_MIN)
                        r = INTENSITY_MIN;
                  if(r > INTENSITY_MAX)
                        r = INTENSITY_MAX;

                  outputImage.at<Vec3b>(i, j)[0] = round(b);
                  outputImage.at<Vec3b>(i, j)[1] = round(g);
                  outputImage.at<Vec3b>(i, j)[2] = round(r);

            }
      }
      return ;
}


//Functio to create histogram from intensity value calculated from the RGB to HSI function
void intensityToHistogram( float **I,float histogram[], int rows , int cols){

      for(int i = 0; i < LEVEL ; i++){
            histogram[i] = 0;
      }

      for(int y = 0 ; y < rows ; y ++){
            for(int x = 0 ; x < cols ; x++){
                  histogram[(int)I[y][x]]++;
            }
      }

      for(int i = 0; i < LEVEL ; i++){
            histogram[i] = histogram[i]/(rows*cols);
      }

      return;

}

//Function to match histogram of the input image to the target image
void histogramMatching(float inputTranFunc[], float targetTranFunc[], float histogram[] , float targetHistogram[]){

      for(int i = 0; i < LEVEL ; i++){
            int j = 0;
            do {
                  histogram[i] = j;
                  j++;
            }while(inputTranFunc[i] > targetTranFunc[j]);
      }
      return;
}

//Function to display histogram of an image and to write the historam in the outout file
void showHistogram(Mat& image, string fileName){
    int bins = 256;             // number of bins
    int nc = image.channels();    // number of channels
    vector<Mat> histogram(nc);       // array for storing the histograms
    vector<Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = {0,0,0};      // peak value for each histogram

    // The rest of the code will be placed here
	for (int i = 0; i < histogram.size(); i++)
    histogram[i] = Mat::zeros(1, bins, CV_32SC1);

	for (int i = 0; i < image.rows; i++){
		for (int j = 0; j < image.cols; j++){
			for (int k = 0; k < nc; k++){
				uchar val = nc == 1 ? image.at<uchar>(i,j) : image.at<Vec3b>(i,j)[k];
				histogram[k].at<int>(val) += 1;
			}
		}
	}

	for (int i = 0; i < nc; i++){
		for (int j = 0; j < bins-1; j++)
			hmax[i] = histogram[i].at<int>(j) > hmax[i] ? histogram[i].at<int>(j) : hmax[i];
	}

	const char* wname[3] = { "Blue", "Green", "Red" };
	Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

	for (int i = 0; i < nc; i++){
		canvas[i] = Mat::ones(125, bins, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < bins-1; j++){
			line(
				canvas[i],
				Point(j, rows),
				Point(j, rows - (histogram[i].at<int>(j) * rows/hmax[i])),
				nc == 1 ? Scalar(255, 255, 255) : colors[i],
				1, 8, 0
			);
		}

		imshow(nc == 1 ? fileName : wname[i]+fileName, canvas[i]);
		// string name = string(wname[i])+".jpg";
		// imwrite(nc == 1 ? fileName+".jpg" : name, canvas[i]);
	}
}

Mat readImage(string &fileName, string type){
      cout << endl << "Please Select " <<type<<" Image." << endl;
      cout << "Example <grayscale/lena.bmp> <grayscale/Baboon_dull.jpg>" << endl;
      cout << "Example <lena_color.jpg>  <PeppersRGB.bmp>" << endl << endl;
      cin >> fileName;

    //  fileName = "TestImages/" + fileName;
      cout<< "File Selected: " << fileName << endl;
      Mat inputImage = imread(fileName, CV_LOAD_IMAGE_UNCHANGED);

      return inputImage;
}

int main(){

      int option;
      string name;
      cout << "Assignment 2: Histogram Equalisation and Matching" << endl;
      cout << "Select the following options" << endl;
      cout << "   1. Histogram Equalisation" << endl;
      cout << "   2. Histogram Matching" << endl;
      cin  >> option;
      if(option == 1){
            cout << "               HISTOGRAM EQUALISATION                    "<< endl;
            string fileName;

		Mat inputImage = readImage(fileName, "Input");
		if (inputImage.empty()){
                  cerr << "Error: Loading image" << endl;
                  _getch();
                  return -1;
            }

            if(inputImage.channels() == 1){
                  cout << "Grayscale Image" << endl;
                  int size = inputImage.rows * inputImage.cols;

                  float histogram[LEVEL];
                  cout << "HISTOGRAM" << endl ;
                  imageToHistogram(inputImage,histogram , size);
                  printHistogram(histogram);

                  float tranFunc[LEVEL];
                  cout << "CUMULATIVE HISTOGRAM" << endl;
                  calTranFunc(histogram, tranFunc);
                  printHistogram(tranFunc);

                  float outHistogram[LEVEL];
                  cout << "OUTPUT HISTOGRAM" << endl;
                  intensityMapping(tranFunc, outHistogram);
                  for(int i = 0 ; i < LEVEL ; i++){
                        cout << (int)outHistogram[i] << ",";
                  }

                  Mat outputImage = inputImage.clone();
                  for (int y = 0; y < inputImage.rows; y++){
                        for (int x = 0; x < inputImage.cols; x++){
                              outputImage.at<uchar>(y, x) = saturate_cast<uchar>(saturate_cast<int>(outHistogram[inputImage.at<uchar>(y, x)]));
                        }
                  }

                  namedWindow("Original Image");
                  imshow("Original Image", inputImage);
                  showHistogram(inputImage, " Original Histogram");

                  namedWindow("Histogram Equilized Image");
                  imshow("Histogram Equilized Image", outputImage);
                  showHistogram(outputImage, " Equalized Histogram");

                  waitKey();
            }
            if(inputImage.channels() == 3){
                  cout << "RGB Image" << endl;

                  Mat inputImageHSI(inputImage.rows, inputImage.cols , inputImage.type());

                  float **H =  new float*[inputImage.rows];
                  float **S =  new float*[inputImage.rows];
                  float **I =  new float*[inputImage.rows];
                  for(int i = 0; i < inputImage.rows; i++){
                        H[i] = new float[inputImage.cols];
                        S[i] = new float[inputImage.cols];
                        I[i] = new float[inputImage.cols];
                  }

                  convert_RGB_To_HSI(inputImage, inputImageHSI, H , S , I);

                  float histogram[LEVEL];
                  cout << "HISTOGRAM" << endl ;
                  intensityToHistogram(I , histogram, inputImage.rows, inputImage.cols);
                  printHistogram(histogram);

                  float tranFunc[LEVEL];
                  cout << "CUMULATIVE HISTOGRAM" << endl;
                  calTranFunc(histogram, tranFunc);
                  printHistogram(tranFunc);

                  float outHistogram[LEVEL];
                  cout << "OUTPUT HISTOGRAM" << endl;
                  intensityMapping(tranFunc, outHistogram);
                  for(int i = 0 ; i < LEVEL ; i++){
                        cout << (int)outHistogram[i] << ",";
                  }

                  float **outI =  new float*[inputImage.rows];
                  for(int i = 0; i < inputImage.rows; i++){
                        outI[i] = new float[inputImage.cols];
                  }

                  for (int i = 0; i < inputImage.rows; i++){
                        for (int j = 0; j < inputImage.cols; j++){
                              outI[i][j] = (int)outHistogram[(int)I[i][j]];
                        }
                  }

                  Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type());
                  convert_HSI_To_RGB(outputImage , inputImageHSI , H , S , outI);

                  namedWindow("Original Image",CV_WINDOW_AUTOSIZE);
                  imshow("Original Image", inputImage);
                  showHistogram(inputImage, " Original Histogram");

                  namedWindow("HSI Image", CV_WINDOW_AUTOSIZE);
                  imshow("HSI Image", inputImageHSI);

                  namedWindow("RGB Histogram Equilized Image", CV_WINDOW_AUTOSIZE);
                  imshow("RGB Histogram Equilized Image", outputImage);
                  showHistogram(outputImage, " Equalized Histogram");

                  waitKey();
            }
      }else if(option == 2){
            cout << "               HISTOGRAM MATCHING                    "<< endl;

            string inputFileName, targetFileName;
            // inputFileName = (char*)malloc(50*sizeof(char));
            // targetFileName = (char*)malloc(50*sizeof(char));

            Mat inputImage = readImage(inputFileName, "Input");
		if (inputImage.empty()){
                  cerr << "Error: Loading image" << endl;
                  _getch();
                  return -1;
            }

            Mat targetImage = readImage(targetFileName, "Target");
		if (targetImage.empty()){
                  cerr << "Error: Loading image" << endl;
                  _getch();
                  return -1;
            }


            cout << "inputFileName: " << inputFileName <<  endl;
            cout << "targetFileName: " << targetFileName << endl;

            if(inputImage.channels() == 1 && targetImage.channels() == 1 ){
                  int inputSize = inputImage.rows * inputImage.cols;

                  float inputHistogram[LEVEL];
                  cout << "INPUT HISTOGRAM" << endl ;
                  imageToHistogram(inputImage, inputHistogram , inputSize);
                  printHistogram(inputHistogram);

                  float inputTranFunc[LEVEL];
                  cout << "INPUT CUMULATIVE HISTOGRAM" << endl;
                  calTranFunc(inputHistogram, inputTranFunc);
                  printHistogram(inputTranFunc);

                  int targetSize = targetImage.rows * targetImage.cols;
                  float targetHistogram[LEVEL];
                  cout << "TARGET HISTOGRAM" << endl ;
                  imageToHistogram(targetImage,targetHistogram , targetSize);
                  printHistogram(targetHistogram);

                  float targetTranFunc[LEVEL];
                  cout << "TARGET CUMULATIVE HISTOGRAM" << endl;
                  calTranFunc(targetHistogram, targetTranFunc);
                  printHistogram(targetTranFunc);

                  float outHistogram[LEVEL];
                  cout << "OUTPUT HISTOGRAM" << endl;
                  histogramMatching(inputTranFunc, targetTranFunc, outHistogram, targetHistogram);
                  for(int i = 0 ; i < LEVEL ; i++){
                        cout << outHistogram[i] << ",";
                  }

                  Mat outputImage = inputImage.clone();

                  for (int y = 0; y < inputImage.rows; y++){
                        for (int x = 0; x < inputImage.cols; x++){
                              outputImage.at<uchar>(y, x) = (int)(outHistogram[inputImage.at<uchar>(y, x)]);
                        }
                  }

                  namedWindow("Original Image");
                  imshow("Original Image", inputImage);
                  showHistogram(inputImage, " Original Histogram");

                  namedWindow("Target Image");
                  imshow("Target Image", targetImage);
                  showHistogram(targetImage, " Target Histogram");

                  namedWindow("Histogram Matched Image");
                  imshow("Histogram Matched Image", outputImage);
                  showHistogram(outputImage, " Matched Histogram");

                  waitKey();

            }else{
                  cout << "RGB Histogram Matching" << endl;

                  cout << "inputFileName: " << inputFileName <<  endl;
                  cout << "targetFileName: " << targetFileName << endl;


                  Mat inputImage = imread(inputFileName, CV_LOAD_IMAGE_COLOR);
                  Mat inputImageHSI(inputImage.rows, inputImage.cols , inputImage.type());

                  float **inputImage_H =  new float*[inputImage.rows];
                  float **inputImage_S =  new float*[inputImage.rows];
                  float **inputImage_I =  new float*[inputImage.rows];
                  for(int i = 0; i < inputImage.rows; i++){
                        inputImage_H[i] = new float[inputImage.cols];
                        inputImage_S[i] = new float[inputImage.cols];
                        inputImage_I[i] = new float[inputImage.cols];
                  }

                  // float r,g,b;
                  // b = inputImage.at<Vec3b>(0, 0)[0];
                  // g = inputImage.at<Vec3b>(0, 0)[1];
                  // r = inputImage.at<Vec3b>(0, 0)[2];
                  // cout << "R:" << r <<"; G: "<< g << "; B:" << b << endl;

                  convert_RGB_To_HSI(inputImage, inputImageHSI, inputImage_H , inputImage_S , inputImage_I);

                  // float h,s,in;
                  // h = inputImageHSI.at<Vec3b>(0, 0)[0];
                  // s = inputImageHSI.at<Vec3b>(0, 0)[1]/100;
                  // in = inputImageHSI.at<Vec3b>(0, 0)[2];
                  // cout << "H:" << h <<"; S: "<< s << "; I:" << in << endl;
                  // cout << "H:" << inputImage_H[0][0] <<"; S: "<< inputImage_S[0][0] << "; I:" << inputImage_I[0][0] << endl;

                  // Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type());
                  // convert_HSI_To_RGB(outputImage , inputImageHSI , inputImage_H , inputImage_S , inputImage_I);
                  //
                  // b = outputImage.at<Vec3b>(0, 0)[0];
                  // g = outputImage.at<Vec3b>(0, 0)[1];
                  // r = outputImage.at<Vec3b>(0, 0)[2];
                  // cout << "R:" << r <<"; G: "<< g << "; B:" << b << endl;
                  //
                  // namedWindow("Original Image");
                  // imshow("Original Image", inputImage);
                  // showHistogram(inputImage, " Original Histogram");
                  //
                  // namedWindow("Histogram Matched Image");
                  // imshow("Histogram Matched Image", outputImage);
                  // showHistogram(outputImage, " Matched Histogram");




                  float inputHistogram[LEVEL];
                  cout << "INPUT HISTOGRAM" << endl ;
                  intensityToHistogram(inputImage_I , inputHistogram, inputImage.rows, inputImage.cols);
                  printHistogram(inputHistogram);


                  float inputTranFunc[LEVEL];
                  cout << "INPUT CUMULATIVE HISTOGRAM" << endl;
                  calTranFunc(inputHistogram, inputTranFunc);
                  printHistogram(inputTranFunc);



                  Mat targetImage = imread(targetFileName, CV_LOAD_IMAGE_COLOR);
                  Mat targetImageHSI(targetImage.rows, targetImage.cols , targetImage.type());

                  float **targetImage_H =  new float*[targetImage.rows];
                  float **targetImage_S =  new float*[targetImage.rows];
                  float **targetImage_I =  new float*[targetImage.rows];
                  for(int i = 0; i < targetImage.rows; i++){
                        targetImage_H[i] = new float[targetImage.cols];
                        targetImage_S[i] = new float[targetImage.cols];
                        targetImage_I[i] = new float[targetImage.cols];
                  }

                  convert_RGB_To_HSI(targetImage, targetImageHSI, targetImage_H , targetImage_S , targetImage_I);

                  float targetHistogram[LEVEL];
                  cout << "TARGET HISTOGRAM" << endl ;
                  intensityToHistogram(targetImage_I , targetHistogram, targetImage.rows, targetImage.cols);
                  printHistogram(targetHistogram);


                  float targetTranFunc[LEVEL];
                  cout << "TARGET CUMULATIVE HISTOGRAM" << endl;
                  calTranFunc(targetHistogram, targetTranFunc);
                  printHistogram(targetTranFunc);

                  float outHistogram[LEVEL];
                  cout << "OUTPUT HISTOGRAM" << endl;
                  histogramMatching(inputTranFunc, targetTranFunc, outHistogram, targetHistogram);
                  for(int i = 0 ; i < LEVEL ; i++){
                        cout << outHistogram[i] << ",";
                  }


                  float **outI =  new float*[inputImage.rows];
                  for(int i = 0; i < inputImage.rows; i++){
                        outI[i] = new float[inputImage.cols];
                  }

                  for (int i = 0; i < inputImage.rows; i++){
                        for (int j = 0; j < inputImage.cols; j++){
                              outI[i][j] = (int)outHistogram[(int)inputImage_I[i][j]];
                        }
                  }

                  Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type());
                  convert_HSI_To_RGB(outputImage , inputImageHSI , inputImage_H , inputImage_S , outI);


                  namedWindow("Original Image");
                  imshow("Original Image", inputImage);
                  showHistogram(inputImage, " Original Histogram");

                  namedWindow("Target Image");
                  imshow("Target Image", targetImage);
                  showHistogram(targetImage, " Target Histogram");

                  namedWindow("Histogram Matched Image");
                  imshow("Histogram Matched Image", outputImage);
                  showHistogram(outputImage, " Matched Histogram");

                  waitKey();

            }

      }else{
            cout << "Please choose the correct option" << endl;
            return -1;
      }

      waitKey();
      _getch();
      return 0;
}
