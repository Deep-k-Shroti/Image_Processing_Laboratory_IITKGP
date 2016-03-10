#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <cmath>
# define M_PI           3.14159265358979323846  /* pi */

using namespace cv;
using namespace std;

// GLOBAL VARIABLES

// FILTER OPTIONS
const int IDEAL_LPF = 0;
const int IDEAL_HPF = 1;
const int GAUSSIAN_LPF = 2;
const int GAUSSIAN_HPF = 3;
const int BUTTER_LPF = 4;
const int BUTTER_HPF = 5;

// FILTER NAMES
const string filters[6] = { "IDEAL_LPF", "IDEAL_HPF", "GAUSSIAN_LPF", "GAUSSIAN_HPF", "BUTTER_LPF", "BUTTER_HPF" };

// GLOBAL PARAMETERS USED BY THE PARAMETERS
int cutoff_G = 1; // 0.1 to 1 @ inc = 0.1
int gaussianSigma_G = 1; // 1 to 100 @ inc = 10
int butterN_G = 1; // 1 to 10 @ inc = 1
int butterC_G = 1; // 0.1 to 1 @ inc = 0.1
int fileID = 0;
int filterID = 0;
vector<string> files;

// MAT DATA STRUCTURES FOR STORING THE IMAGES
Mat image;
Mat FFTImg_After;
Mat FFTImg_Before;
Mat FilterImg;
Mat IFFTImg;
float inc = 0.05;

void GetFilesInDirectory(std::vector<string> &out, const string &directory)
{
	HANDLE dir;
	WIN32_FIND_DATA file_data;
	wchar_t dir_L[256];
	mbstowcs((wchar_t*)dir_L, (directory + "/*").c_str(), 256);
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

class ComplexFloat {
public:
	double real;
	double img;

public:
	ComplexFloat()
	{
		this->real = 0;
		this->img = 0;
	}
	ComplexFloat(double real, double img)
	{
		this->real = real;
		this->img = img;
	}
	ComplexFloat operator+(const ComplexFloat& b)
	{
		double r = real + b.real;
		double i = img + b.img;
		return ComplexFloat(r, i);
	}
	ComplexFloat operator-(const ComplexFloat& b)
	{
		double r = real - b.real;
		double i = img - b.img;
		return ComplexFloat(r, i);
	}
	ComplexFloat operator*(const ComplexFloat& b)
	{
		double k1 = b.real*(real + img);
		double k2 = real*(b.img - b.real);
		double k3 = img*(b.img + b.real);
		return ComplexFloat(k1 - k3, k1 + k2);
	}

	ComplexFloat operator*(const double& b)
	{
		return ComplexFloat(real*b, img*b);
	}

	void operator*=(const double& b)
	{
		real *= b;
		img *= b;
	}

	ComplexFloat operator/(const double& b)
	{
		return ComplexFloat(real / b, img / b);
	}

	void operator=(const double& b)
	{
		real = b;
		img = 0;
	}

	double magnitude()
	{
		return sqrt(real*real + img*img);
	}
	void print() {
		cout << real << " + " << img << "i";
	}

};

template<typename T>
void Transpose(T** matrix, int N)
{
	T temp;
	for (int i = 0; i < N; i++) {
		T* start = matrix[i] + i;
		for (int j = i + 1; j < N; j++) {
			temp = matrix[i][j];
			matrix[i][j] = matrix[j][i];
			matrix[j][i] = temp;
		}
	}
}

template<typename T>
void FFTShift(T** matrix, int N)
{
	T temp;
	int offset = N / 2;
	for (int i = 0; i < offset; i++) {
		T* start = matrix[i] + i;
		for (int j = 0; j < offset; j++) {
			temp = matrix[i][j];
			matrix[i][j] = matrix[i + offset][j + offset];
			matrix[i + offset][j + offset] = temp;
		}
	}

	for (int i = N / 2; i < N; i++) {
		T* start = matrix[i] + i;
		for (int j = 0; j < offset; j++) {
			temp = matrix[i][j];
			matrix[i][j] = matrix[i - offset][j + offset];
			matrix[i - offset][j + offset] = temp;
		}
	}
}

template<typename T>
void FFTShift(Mat &matrix, int N)
{
	T temp;
	int offset = N / 2;
	for (int i = 0; i < offset; i++) {
		for (int j = 0; j < offset; j++) {
			temp = matrix.at<T>(i, j);
			matrix.at<T>(i, j) = matrix.at<T>(i + offset, j + offset);
			matrix.at<T>(i + offset, j + offset) = temp;
		}
	}

	for (int i = N / 2; i < N; i++) {
		for (int j = 0; j < offset; j++) {
			temp = matrix.at<T>(i, j);
			matrix.at<T>(i, j) = matrix.at<T>(i - offset, j + offset);
			matrix.at<T>(i - offset, j + offset) = temp;
		}
	}
}

//ASSUMPTIONS
//WHEN CALLING THIS FUNCTION
//arrSize = N
//gap = 1
//zeroLoc = 0

ComplexFloat* FFT(uchar* x, int N, int arrSize, int zeroLoc, int gap)
{
	ComplexFloat* fft;
	fft = new ComplexFloat[N];

	int i;
	if (N == 2)
	{
		fft[0] = ComplexFloat(x[zeroLoc] + x[zeroLoc + gap], 0);
		fft[1] = ComplexFloat(x[zeroLoc] - x[zeroLoc + gap], 0);
	}
	else
	{
		ComplexFloat wN = ComplexFloat(cos(2 * M_PI / N), sin(-2 * M_PI / N));//exp(-j2*pi/N)
		ComplexFloat w = ComplexFloat(1, 0);
		gap *= 2;
		ComplexFloat* X_even = FFT(x, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		ComplexFloat* X_odd = FFT(x, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		ComplexFloat todd;
		for (i = 0; i < N / 2; ++i)
		{
			//FFT(0) IS EQUAL TO FFT(N-1) SYMMETRICAL AROUND N/2
			todd = w*X_odd[i];
			fft[i] = X_even[i] + todd;
			fft[i + N / 2] = X_even[i] - todd;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return fft;
}
ComplexFloat* FFT(ComplexFloat* x, int N, int arrSize, int zeroLoc, int gap)
{
	ComplexFloat* fft;
	fft = new ComplexFloat[N];

	int i;
	if (N == 2)
	{
		fft[0] = x[zeroLoc] + x[zeroLoc + gap];
		fft[1] = x[zeroLoc] - x[zeroLoc + gap];
	}
	else
	{
		ComplexFloat wN = ComplexFloat(cos(2 * M_PI / N), sin(-2 * M_PI / N));//exp(-j2*pi/N)
		ComplexFloat w = ComplexFloat(1, 0);
		gap *= 2;
		ComplexFloat* X_even = FFT(x, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		ComplexFloat* X_odd = FFT(x, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		ComplexFloat todd;
		for (i = 0; i < N / 2; ++i)
		{
			//FFT(0) IS EQUAL TO FFT(N-1) SYMMETRICAL AROUND N/2
			todd = w*X_odd[i];
			fft[i] = X_even[i] + todd;
			fft[i + N / 2] = X_even[i] - todd;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return fft;
}
ComplexFloat* IFFT(ComplexFloat* fft, int N, int arrSize, int zeroLoc, int gap)
{
	ComplexFloat* signal;
	signal = new ComplexFloat[N];

	int i;
	if (N == 2)
	{
		signal[0] = fft[zeroLoc] + fft[zeroLoc + gap];
		signal[1] = fft[zeroLoc] - fft[zeroLoc + gap];
	}
	else
	{
		ComplexFloat wN = ComplexFloat(cos(2 * M_PI / N), sin(2 * M_PI / N));//exp(j2*pi/N)
		ComplexFloat w = ComplexFloat(1, 0);
		gap *= 2;
		ComplexFloat* X_even = IFFT(fft, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		ComplexFloat* X_odd = IFFT(fft, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		ComplexFloat todd;
		for (i = 0; i < N / 2; ++i)
		{
			//FFT(0) IS EQUAL TO FFT(N-1) SYMMETRICAL AROUND N/2
			todd = w * X_odd[i];
			signal[i] = (X_even[i] + todd) * 0.5;
			signal[i + N / 2] = (X_even[i] - todd) * 0.5;
			w = w * wN; // Get the next root(conjugate) among Nth roots of unity
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return signal;
}

ComplexFloat** FFT2(Mat& source) {
	cout << "Applying FFT2" << endl;

	if (source.rows != source.cols) {
		cout << "Image is not Valid";
		return nullptr;
	}
	int N = source.rows;
	//cout << "Image size:" << N << endl;
	ComplexFloat** FFT2Result_h;
	FFT2Result_h = new ComplexFloat*[N];

	// ROW WISE FFT
	for (int i = 0; i < N; i++) {
		uchar* row = source.ptr<uchar>(i);
		FFT2Result_h[i] = FFT(row, N, N, 0, 1);
	}

	//cout << "final: " << endl;
	Transpose<ComplexFloat>(FFT2Result_h, N);

	// COLUMN WISE FFT
	for (int i = 0; i < N; i++) {
		FFT2Result_h[i] = FFT(FFT2Result_h[i], N, N, 0, 1);
	}
	Transpose<ComplexFloat>(FFT2Result_h, N);

	return FFT2Result_h;
}

ComplexFloat** IFFT2(ComplexFloat** source, int N) {

	cout << "Applying IFFT2" << endl;

	ComplexFloat** ifftResult;
	ifftResult = new ComplexFloat*[N];
	// ROW WISE FFT
	for (int i = 0; i < N; i++) {
		ifftResult[i] = IFFT(source[i], N, N, 0, 1);
	}

	//cout << "final: " << endl;
	Transpose<ComplexFloat>(ifftResult, N);

	int d = N*N;
	// COLUMN WISE FFT
	for (int i = 0; i < N; i++) {
		ifftResult[i] = IFFT(ifftResult[i], N, N, 0, 1);
		for (int j = 0; j < N; j++) {
			ifftResult[i][j] = ifftResult[i][j] / d;
		}
	}
	Transpose<ComplexFloat>(ifftResult, N);

	cout << endl;

	return ifftResult;
}

void Complex2Mat(ComplexFloat** source, Mat& dest, int N, bool shift = false, float maxF = 1.0) {
	// Convert a complex matrix to a Mat data structure (magnitude of 
	// the complex no. are used) for showing as an image
	if (shift) {
		FFTShift(source, N);
	}
	dest = Mat(N, N, CV_32F, cv::Scalar::all(0));
	float min = 99999;
	float max = 0;

	// Find min and max
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			source[i][j] = source[i][j] / N;
			float m = source[i][j].magnitude();
			if (m < min) {
				min = m;
			}
			if (m > max) {
				max = m;
			}
		}
	}


	// Normalize the image
	float range = (max - min);
	for (int i = 0; i < N; i++) {
		float *p = dest.ptr<float>(i);
		for (int j = 0; j < N; j++) {
			p[j] = (source[i][j].magnitude() - min) * maxF / range;
		}
	}
	//cout << "Min: " << min << " Max:" << max;
}

void ApplyFilter(ComplexFloat** source, Mat& filterImg, int N, int FilterType) {
	float cutoff = cutoff_G*inc; //Compute Ideal filter cutoff
	float sigma_squared = gaussianSigma_G*inc + inc; //Compute Gaussing filter sigma from trackbar input
	int butter_n = butterN_G; // Butterworth parameter n
	// Cutoff lies in [0, 2]
	cutoff *= cutoff; // Square it to avoid further sqrt
	filterImg = Mat(N, N, CV_32F); // Image for showing the frequency spectrum of the filter
	float d = N*N;
	ComplexFloat** filterFFT;
	switch (FilterType) {
	case IDEAL_LPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float f = (i*i / d) + (j*j / d);
				if (i == 5 && j == 5) {
					cout << "Cutoff Frequency is:" << f << endl;
				}
				if (f > cutoff) {
					// Remove the components outside the
					// cutoff frequency
					source[i][j] = 0;
					source[N - 1 - i][N - 1 - j] = 0;
					source[N - 1 - i][j] = 0;
					source[i][N - 1 - j] = 0;
				}
				else {
					// Filter coeff = 1 withing cutoff frequency range
					filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = 1;
				}
			}
		}
		break;
	case IDEAL_HPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float f = (i*i / d) + (j*j / d);
				if (i == 5 && j == 5) {
					cout << "Cutoff Frequency is:" << f << endl;
				}
				if (f <= cutoff) {
					// Remove the components @ less than the
					// cutoff frequency
					source[i][j] = 0;
					source[N - 1 - i][N - 1 - j] = 0;
					source[N - 1 - i][j] = 0;
					source[i][N - 1 - j] = 0;
				}
				else {
					// Filter coeff = 1 withing cutoff frequency range
					filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = 1;
				}
			}
		}
		break;
	case GAUSSIAN_LPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * M_PI*i / N, 2); // omega x squared
				float wy2 = pow(2 * M_PI*j / N, 2); // omega y squared
				float coeff = exp(-(wx2 + wy2) / (2 * sigma_squared)); // Gaussian filter coeff @ (wx, wy)
				source[i][j] *= coeff;
				source[N - 1 - i][N - 1 - j] *= coeff;
				source[N - 1 - i][j] *= coeff;
				source[i][N - 1 - j] *= coeff;

				filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = coeff;
			}
		}
		break;
	case GAUSSIAN_HPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * M_PI*i / N, 2); // omega x squared
				float wy2 = pow(2 * M_PI*j / N, 2); // omega y squared
				float coeff = 1 - exp(-(wx2 + wy2) / (2 * sigma_squared)); // Gaussian filter coeff @ (wx, wy)
				source[i][j] *= coeff;
				source[N - 1 - i][N - 1 - j] *= coeff;
				source[N - 1 - i][j] *= coeff;
				source[i][N - 1 - j] *= coeff;

				filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	case BUTTER_LPF:
		cutoff = pow((butterC_G*inc + inc) * M_PI, 2);
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * M_PI*i / N, 2);
				float wy2 = pow(2 * M_PI*j / N, 2);
				float coeff = 1 / (1 + pow((wx2 + wy2) / cutoff, 2 * butter_n)); // Butterworth filter coeff @ (wx, wy)
				source[i][j] *= coeff;
				source[N - 1 - i][N - 1 - j] *= coeff;
				source[N - 1 - i][j] *= coeff;
				source[i][N - 1 - j] *= coeff;

				filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	case BUTTER_HPF:
		cutoff = pow((butterC_G*inc + inc) * M_PI, 2);
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * M_PI*i / N, 2);
				float wy2 = pow(2 * M_PI*j / N, 2);
				float coeff = 1 / (1 + pow(cutoff / (wx2 + wy2), 2 * butter_n)); // Butterworth filter coeff @ (wx, wy)
				source[i][j] *= coeff;
				source[N - 1 - i][N - 1 - j] *= coeff;
				source[N - 1 - i][j] *= coeff;
				source[i][N - 1 - j] *= coeff;

				filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	}
}

// Event handlers for trackbars
void on_cutoff(int, void*) {
	cout << "Cutoff frequency chosen: " << cutoff_G*inc + inc << endl;
}

void on_gSigma(int, void*) {
	cout << "Sigma Value: " << gaussianSigma_G*inc + inc << endl;
}

void on_butterN(int, void*) {
	cout << "ButterWorth Order: " << butterN_G << endl;
}

void on_butterC(int, void*) {
	cout << "ButterWorth Order: " << (butterC_G*inc + inc) << endl;
}
void on_fileChange(int, void*) {
	cout << "File: " << files[fileID] << endl;

	image = imread(files[fileID], IMREAD_GRAYSCALE); // Read the file

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}
	imshow("Source File", image); // Show our image inside it.
}

void on_filterChange(int, void*) {
	cout << "Filter: " << filters[filterID] << endl;
}

void StartOP(string filename) {

	if (!image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}
	ComplexFloat** fft2result = FFT2(image);
	Complex2Mat(fft2result, FFTImg_Before, image.rows, false, 255);
	// Apply Filter
	ApplyFilter(fft2result, FilterImg, image.rows, filterID);
	// Roate the FFT Matrix to bring freq(0, 0) at the middle of the image
	FFTShift<float>(FilterImg, image.rows);
	//ApplyFilter(fft2result, image.rows, IDEAL_LPF, 0.15);

	ComplexFloat** ifft2result = IFFT2(fft2result, image.rows);
	float maxF = 1;
	switch (filterID) {
	case IDEAL_LPF:
	case GAUSSIAN_LPF:
	case BUTTER_LPF:
		maxF = 255;
	}
	Complex2Mat(fft2result, FFTImg_After, image.rows, true, maxF);
	Complex2Mat(ifft2result, IFFTImg, image.rows);


	FFTShift<float>(FFTImg_Before, image.rows);

	// Show the results
	imshow("FFT window Before", FFTImg_Before);
	imshow("FFT window After", FFTImg_After);
	imshow("Filter Spectrum", FilterImg);
	imshow("Output Image", IFFTImg);
}

const int ENTER_KEY = 13;
const int ESC_KEY = 27;

int main()
{
	int displaySize = 400;
	string folder;
	//cout << "Mention the input folder [must be inside working folder of executable]: " << endl;
	//cin >> folder;
	folder = "TestImages";
	GetFilesInDirectory(files, folder); // Get the filenames in mentioned directory

//	namedWindow("Display window", 0);
	//resizeWindow("Display window", 500, 400);

	namedWindow("Source File", 0);
	resizeWindow("Source File", displaySize, displaySize);
	// Create trackbars
	createTrackbar("Image", "Source File", &fileID, files.size() - 1, on_fileChange);
	createTrackbar("Filter", "Source File", &filterID, 5, on_filterChange);
	createTrackbar("Ideal Filter cutoff ", "Source File", &cutoff_G, 0.5 / inc, on_cutoff); // cutoff
	createTrackbar("Gaussian Filter Sigma ", "Source File", &gaussianSigma_G, 2 / inc, on_gSigma); // gaussian sigma
	createTrackbar("ButterWorth n ", "Source File", &butterN_G, 10, on_butterN); // butterworth n
	createTrackbar("ButterWorth c", "Source File", &butterC_G, 1 / inc, on_butterC); // butterworth c

	// Create the windows for showing results
	

	//namedWindow("FFT window Before", 0);
	//resizeWindow("FFT window Before", displaySize, displaySize);

	//namedWindow("FFT window After", 0);
	//resizeWindow("FFT window After", displaySize, displaySize);

	namedWindow("Filter Spectrum", 0);
	resizeWindow("Filter Spectrum", displaySize, displaySize);

	namedWindow("Output Image", 0);
	resizeWindow("Output Image", displaySize, displaySize);

	int c = 0;
	// Wait for keyboard inputs
	// Enter -> do filter for the current chosen image
	// Escape -> Exit application
	while (c != ESC_KEY) {
		c = waitKey(0);
		if (c == ENTER_KEY) {
			cout << "Working";
			StartOP(files[fileID]);
		}
	}

	return 0;
}