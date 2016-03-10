# Image_Processing_Laboratory_IITKGP
##Image_Processing_Laboratory_IITKGP(DIP) Laboratory Spring 2016.

This repository contains all DIP Lab Assignments. The Experiment are written in C without using OpenCV library except for reading of image.
Code written using Visual Studio 2010.

##Directories included <br />
/Assignment1 <br />
/Assignment2 <br />
/Assignment3 <br />
/Assignment4 <br />
/Assignment5 <br />
Each directory includes the experiment problem statement(.pdf),README(.txt), TestImages, source code(.cpp) and report(.pdf).

##The list of experiments:
1. BMP File Format <br />
	Write C/C++ modular functions to read, diagonally flip, and then write BMP image files. All functions must support at least 24-bit RGB, and 8-bit grayscale image formats. <br />
	1. ReadBMP: <br />
	   	* Input: Filename of input image. <br />
		* Output: BMP header structure, Image pixel array loaded onto memory. <br />
	2. ConvertFlipGrayscale: <br />
		* Input: Image pixel array. <br />
		* Output: Grayscale-converted and diagonally flipped (transposed) pixel array. <br />
		* Input: Filename of output (grayscale) image, BMP header structure, Image pixel array.
		3. WriteBMP: <br />
		* Output: BMP file written on disk. <br />
	Use the above functions to read a 256 × 256 24-bit RGB colored Lena image, and write it as an 8-bit	grayscale onto a different file on the disk. <br />

2. Histogram Equalization and Matching <br />
	Write C++/Image-J modular functions to <br />
	1. Perform histogram equalization on the 512 × 512 grayscale lena_gray_dark.jpg image. Perform the same on other low-contrast, dark, normal (gray/colored) images. <br />
	2. Perform histogram matching of the same images with respect to a standard image (e.g. lena.jpg). <br />
	Display the histograms of the original image and the enhanced images and document the observations. <br />

3. Spatial Filtering <br />
	Write C/C++ modular functions/subroutines to design spatial filters - mean, median, gradient, Laplacian, Sobel kernels (horizontal, vertical, diagonal) on a stack of grayscale images (say, 15 images per stack) <br />
	Use OpenCV (or) ImageJ for image reading, writing and GUI development only. Use the OpenCV tracker (slider) functionality to show outputs for varying sizes of neighborhoods. You may have different sliders to select
	(i) Image (ii) Filter (iii) Neighborhood size <br />
	1. Input: Path to the stack of images. Input stack should contain the (provided) noisy images, and may also contain the normal test images, e.g. jetplane.jpg, lake.jpg. <br />
	2. Output: Filtered stack of images should be shown beside input stack in the same pane of GUI with a slider to vary filter/kernel size/change image. <br />

4. Frequency Filtering <br />
	Write C++/Image-J modular functions to perform the following operations on the 512 × 512 grayscale test images, e.g. jetplane.jpg, lake.jpg. <br />
	1. FFT2 (takes input image filename as the argument; gives 2D FFT coefficients as output) <br />
	2. IFFT2 (takes 2D FFT coefficients as input argument; gives the back-projected/ reconstructed
	    image as output) <br />
	3. Perform Ideal, Gaussian, and Butterworth low-pass and high-pass filtering, taking cut-off
	    frequency, F, and image filename as input arguments) respectively with <br />
		* Ideal_LPF <br />
		* Ideal_HPF <br />
		* Gaussian_LPF <br />
		* Gaussian_HPF <br />
		* Butterworth_LPF <br />
		* Butterworth_HPF <br />
	Display the (shifted) magnitude spectrums of the input, the filter and the filtered output. You may
	make use of the tracker/slider function to choose images, filter types and cut-off frequencies. <br />

5. Morphological Operations <br />
	Write C++/Image-J modular functions to perform the following operations on the given test image, ricegrains_mono.bmp. All functions must support binary images. <br />
	Make separate functions for erosion, dilation, opening, and closing of binary images. <br />
	1. ErodeBinary, DilateBinary <br />
		* Input: Binary image, structuring element <br />
		* Output: Eroded/dilated image <br />
	2. OpenBinary, CloseBinary <br />
		* Input: Binary image, structuring element <br />
		* Output: Opened/closed image <br />

##References
- (BMP file format wikipedia)[http://en.wikipedia.org/wiki/BMP_file_format] <br />
- (Histogram Equalization wikipedia)[http://en.wikipedia.org/wiki/Histogram_equalization] <br />
- (Morphological Operations: Prof. P. K. Biswas, Lecture 33 - 36, Video Lecture Series (NPTEL) on Digital Image Processing)[https://www.youtube.com/watch?v=CVV0TvNK6pk&list=PL1F076D1A98071E24]<br />
- R. C. Gonzalez and R. Woods, Digital Image Processing, Reading, MA: Addison-Wesley, 1992.

##Setup
- Download the directory <br />
- Create a project Win32 Console Application in Visual studio. <br />
- Copy the folder files to the project. <br />
- Include the ".cpp" file as the source file. <br />
- Include necessary OpenCV libraries [https://www.youtube.com/watch?v=LNlnjBLPJd8] <br />
- Run the ".cpp" code <br />
