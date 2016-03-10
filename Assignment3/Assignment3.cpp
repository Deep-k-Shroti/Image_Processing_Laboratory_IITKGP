#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <windows.h>
#include <vector>
#include <string>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

int fileName = 0;
int filterType = 0;
int filterSize = 0;

string filenames[7] = {"jetplane.jpg", "lake.jpg", "lena_gray_512.jpg", "livingroom.jpg", "mandril_gray.jpg", "pirate.jpg", "walkbridge.jpg"};
vector<string> fileList;//(filenames, filenames+7);

enum Filter { None, Mean_F, Median_F, Gradient_H, Gradient_V, Laplacian_F, Sobel_H, Sobel_V};
const string filters[9] = {"No", "Mean", "Median", "Gradient_H", "Gradient_V", "Laplacian", "Sobel_H", "Sobel_V", "Sobel_D"};

int gradient_H_3[3][3] = {  {-1, -1, -1 },
				    { 0,  0,  0 },
			   	    { 1,  1,  1 }};

int gradient_H_5[5][5] = { { -1, -1, -1, -1, -1 },
				   { -2, -2, -2, -2, -2 },
				   {  0,  0,  0,  0,  0 },
				   {  2,  2,  2,  2,  2 },
				   {  1,  1,  1,  1,  1 } };

int gradient_H_7[7][7] =  { { -1, -1, -1, -1, -1, -1, -1 },
				    { -2, -2, -2, -2, -2, -2, -2 },
				    { -3, -3, -3, -3, -3, -3, -3 },
				    { 0, 0, 0, 0, 0, 0, 0 },
				    { 3, 3, 3, 3, 3, 3, 3 },
				    { 2, 2, 2, 2, 2, 2, 2 },
				    { 1, 1, 1, 1, 1, 1, 1 } };

int gradient_V_3[3][3] = { { 1, 0, -1 },
				   { 1, 0, -1 },
				   { 1, 0, -1 } };

int gradient_V_5[5][5] ={ {-1,-2,0,2,1},
{-1,-2,0,2,1},
{-1,-2,0,2,1},
{-1,-2,0,2,1},
{-1,-2,0,2,1}
};

int gradient_V_7[7][7] = {{-1,-2,-3,0,3,2,1},
{-1,-2,-3,0,3,2,1},
{-1,-2,-3,0,3,2,1},
{-1,-2,-3,0,3,2,1},
{-1,-2,-3,0,3,2,1},
{-1,-2,-3,0,3,2,1},
{-1,-2,-3,0,3,2,1}};


int laplacian_3[3][3] = {	{-1, -1, -1},
							{-1, 8, -1},
							{-1, -1, -1}};

int laplacian_5[5][5] = {	{-1, 3, -4, -3, -1},
					{-3, 0,  6,  0, -3},
					{-4, 6, 20,  6, -4},
					{-3, 0,  6,  0, -3},
					{-1,-3, -4, -3, -1}};

int laplacian_7[7][7] = {	{-2, -3, -4, -6, -4, -3, -2},
					{-3, -5, -4, -3, -4, -5, -3},
					{-4, -4,  9, 20,  9, -4, -4},
					{-6, -3, 20, 36, 20, -3, -6},
					{-4, -4,  9, 20,  9, -4, -4},
					{-3, -5, -4, -3, -4, -5, -3},
					{-2, -3, -4, -6, -4, -3, -2}};



int sobel_H_3[3][3] = { {  1,  2,  1},
						{  0,  0,  0},
						{ -1, -2, -1} };

int sobel_H_5[5][5] = { { 1,   4,   7,   4,  1},
						{ 2,  10,  17,  10,  2},
						{ 0,   0,   0,   0,  0},
						{-2, -10, -17, -10, -2},
						{ -1, -4, -7, -4, -1 } };

int sobel_H_7[7][7] = { {  1,  4,  9,  13,  9,  4,  1},
						{  3, 11, 26,  34, 26, 11,  3},
						{  3, 13, 30,  40, 30, 13,  3},
						{  0,  0,  0,   0,  0,  0,  0},
						{ -3, -13,-30,-40,-30,-13, -3},
						{ -3, -11,-26, 34,-26,-11, -3},
						{ -1, -4, -9, -13, -9, -4, -1} };

int sobel_V_3[3][3] = { { -1, 0, 1 },
						{ -2, 0, 2 },
						{ -1, 0, 1 } };


int sobel_V_5[5][5] = { { -1,  -2, 0,  2, 1 },
						{ -4, -10, 0, 10, 4 },
						{ -7, -17, 0, 17, 7 },
						{ -4, -10, 0, 10, 4 },
						{ -1,  -2, 0, -2, 1 } };



int sobel_V_7[7][7] = { { -1,  -3,  -3, 0, 3,  3,   1},
				{ -4, -11, -13, 0, 13, 11,  4},
				{ -9, -26, -30, 0, 30, 26,  9},
				{ -13, -34,-40, 0, 40, 34, 13},
				{  -9, -26,-30, 0, 30, 26,  9},
				{  -4, -11,-13, 0, 13, 11,  4},
				{   1,  -3, -3, 0,  3,  3,  1} };

int sobel_D_3[3][3] = { { 0,  1, 2},
						{ -1, 0, 1},
						{ -2,-1, 0} };

typedef struct Frame{
    int side;
    int **data;
}Frame;


Frame allocateWindow (int size , int *temp){
	Frame window;
	window.side = size;
	window.data = new int*[window.side];

	for (int i = 0; i < window.side; i++)
		window.data[i] = new int[window.side];

	for (int i = 0; i < window.side; i++){
		for (int j = 0; j < window.side; j++){
			// cout << *((temp + i*size) + j)  << ",";
			window.data[i][j] = *((temp + i*size) + j);
		}
		cout << endl;
	}
	return (window);
}

void GetFiles(vector<string> &out, const string &directory)
{
	HANDLE dir;
	WIN32_FIND_DATA file_data;
	wchar_t dir_L[256];
	mbstowcs((wchar_t*)dir_L, (directory + "/*.jpg").c_str(), 256);
	if ((dir = FindFirstFile(dir_L, &file_data)) == INVALID_HANDLE_VALUE)
		return; /* No files found */

	do {
		char filename[256];
		wcstombs((char*)filename, file_data.cFileName, 256);
		const string file_name = filename;
		const string full_file_name = directory + "/" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (is_directory)
			continue;

		out.push_back(full_file_name);
	} while (FindNextFile(dir, &file_data));

	FindClose(dir);
}

Mat general(Mat input, Frame *window){
  int i, j, count, sum;
  i = j = count = 0;
  int new_size = (window->side)/2;

  Mat output = input.clone();


  for(i = 0; i < input.cols; i++){
    for(j = 0; j < input.rows; j++){
      sum = 0;
      count = 0;
      for(int i1 = -new_size; i1 <= new_size; i1++){
        if(((i + i1) >= 0 ) && ((i + i1) < input.cols)){
          for(int j1 = -new_size; j1 <= new_size; j1++){

            if(((j + j1) >= 0) && ((j + j1) < input.rows)){
              sum += window->data[i1 + new_size][j1 + new_size] * input.at<uchar>(i + i1, j + j1);
              count += abs(window->data[i1 + new_size][j1 + new_size]);
            }
          }
        }
      }
	if(filterType == 5){
		if(sum >  255){
			sum = 255;
		}
		else if(sum < -255){
			sum = -255;
		}
		if(sum < 0)
			sum = -sum;
	}else{
		if(sum >  255)
			sum = 255;
		else if(sum < 0)
			sum = 0;
	}

      output.at<uchar>(i, j ) = sum;//(sum/count);
	  //cout<< sum <<endl;
    }
  }

  return output;
}

Mat Gradient(Mat input, string type , int size){
	cout << "Gradient Operator ";
	Frame mask;
	if (type == "Horizontal"){
		cout << "Type: " << type << " ";
		if(size == 3){
			cout << "Size: " << size << endl;
			mask = allocateWindow(3, (int *)gradient_H_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = allocateWindow(5, (int *)gradient_H_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = allocateWindow(7, (int *)gradient_H_7);
		}
	}

	if (type == "Vertical"){
		cout << "Type: " << type << " ";
		if(size == 3){
			cout << "Size: " << size << endl;
			mask = allocateWindow(3, (int *)gradient_V_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = allocateWindow(5, (int *)gradient_V_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = allocateWindow(7, (int *)gradient_V_7);
		}
	}

	Mat output = general(input, &mask);

	return output;
}

Mat Laplacian(Mat input, int size){
	Frame mask;
	cout << " Laplacian Operator ";
	if(size == 3){
		cout << "Size: " << size << endl;
		mask = allocateWindow(3, (int *)laplacian_3);
	}

	if(size == 5){
		cout << "Size: " << size << endl;
		mask = allocateWindow(5, (int *)laplacian_5);
	}

	if(size == 7){
		cout << "Size: " << size << endl;
		mask = allocateWindow(7, (int *)laplacian_7);
	}

	Mat output = general(input, &mask);

	return output;
}

Mat Sobel(Mat input, string type,  int size){
	Frame mask;
	cout << "Sobel Operator ";

	if (type == "Horizontal"){
		cout << "Type: " << type << " ";
		if(size == 3){
			cout << "Size: " << size << endl;
			mask = allocateWindow(3, (int *)sobel_H_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = allocateWindow(5, (int *)sobel_H_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = allocateWindow(7, (int *)sobel_H_7);
		}
	}

	if (type == "Vertical"){
		cout << "Type: " << type << " ";
		if(size == 3){
			cout << "Size: " << size << endl;
			mask = allocateWindow(3, (int *)sobel_V_3);
		}
		if(size == 5){
			cout << "Size: " << size << endl;
			mask = allocateWindow(5, (int *)sobel_V_5);
		}
		if(size == 7){
			cout << "Size: " << size << endl;
			mask = allocateWindow(7, (int *)sobel_V_7);
		}
	}

	if (type == "Diagonal"){
		cout << "Type: " << type << " ";
		if (size == 3){
			cout << "Size: " << size << endl;
			mask = allocateWindow(3, (int *)sobel_D_3);
		}
	}

	Mat output = general(input, &mask);

	return output;
}

Mat Mean(Mat input, int size){
  int i, j, count, sum;
  i = j = count = 0;
  int new_size = size/2;

  Mat output = input.clone();


  for(i = 0; i < input.cols; i++){
    for(j = 0; j < input.rows; j++){
      sum = 0;
      count = 0;
      for(int i1 = -new_size; i1 <= new_size; i1++){
        if(((i + i1) >= 0 ) && ((i + i1) < input.cols)){
          for(int j1 = -new_size; j1 <= new_size; j1++){

            if(((j + j1) >= 0) && ((j + j1) < input.rows)){
              sum += input.at<uchar>(i + i1, j + j1);
              count++;
            }
          }
        }
      }

      output.at<uchar>(i, j ) = (sum/count);
    }
  }


  return output;

}

Mat Median(Mat input, int size){
  int i, j, count, median;
  i = j = count = 0;
  int new_size = size/2;



  Mat output = input.clone();

  vector<int> frame ( size*size, 0);
  for(i = 0; i < input.cols; i++){
    for(j = 0; j < input.rows; j++){

      count = 0;
      frame.clear();
      for(int i1 = -new_size; i1 <= new_size; i1++){
        if(((i + i1) >= 0 ) && ((i + i1) < input.cols)){
          for(int j1 = -new_size; j1 <= new_size; j1++){

            if(((j + j1) >= 0) && ((j + j1) < input.rows)){
                frame.push_back(input.at<uchar>(i + i1, j + j1));
            }
          }
        }
      }


      sort(frame.begin(), frame.end());
      size = frame.size();
      if (size%2)
         median = frame[size / 2];
      else
         median = (frame[size / 2 - 1] + frame[size / 2]) / 2;

      output.at<uchar>(i, j ) = median;
    }
  }



  return output;

}

void onFilterChange(int, void*){
	cout<< "Applying "<< filters[ filterType ] <<" Filter: ";

	Mat img2 = imread(fileList[fileName], IMREAD_GRAYSCALE);

	switch( filterType){
	case 0:
		break;

	case 1:
		img2 = Mean( img2, 2*filterSize+3);
		break;

	case 2:
		img2 = Median(img2, 2*filterSize+3);
		break;

	case 3:
		img2 = Gradient(img2, "Horizontal", 2*filterSize+3 );
		break;

	case 4:
		img2 = Gradient(img2, "Vertical", 2*filterSize+3 );
		break;

	case 5:
		img2 = Laplacian(img2, 2*filterSize+3);
		break;

	case 6:
		img2 = Sobel(img2, "Horizontal", 2*filterSize+3);
		break;
	case 7:
		img2 = Sobel(img2, "Vertical", 2*filterSize+3);
		break;
	case 8:
		img2 = Sobel(img2, "Diagonal", 2 * filterSize + 3);
		break;
	}
	imshow("Output Image", img2);
	cout<< "Operation Complete"<<endl;
}

void onSizeChange(int, void*){
	cout<< "Size of "<< filters[ filterType ] <<" Filter changed to "<< 2*filterSize+3 <<endl;
	onFilterChange(filterType, 0);
}

void onInputChange(int, void*){
	cout << "Opening: " << fileList[fileName] << endl;

	Mat img = imread(fileList[fileName], IMREAD_GRAYSCALE);

	if (!img.data){
		cout << "Error:Image not found" <<endl;
		return;
	}
	imshow("Input Image", img);
	onFilterChange(filterType , 0);
}

int main() {

	GetFiles(fileList, "./TestImages");
	GetFiles(fileList, "./TestImages/Noisy Images");

	cvNamedWindow("Input Image");
	cvNamedWindow("Output Image");

	string currentFile = fileList[1];

	createTrackbar("Source Choose", "Input Image", &fileName, fileList.size() - 1, onInputChange);
	createTrackbar("Filter Choose", "Output Image", &filterType, 8, onFilterChange);
	createTrackbar("Size Choose", "Output Image", &filterSize, 2, onSizeChange);

	onInputChange(0 , 0);
	onFilterChange(0 , 0);
	onFilterChange(0 , 0);


	waitKey(0);

}
