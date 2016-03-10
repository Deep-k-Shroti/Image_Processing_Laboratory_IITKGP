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
1. BMP File Format 
	Write C/C++ modular functions to read, diagonally flip, and then write BMP image files. All functions
	must support at least 24-bit RGB, and 8-bit grayscale image formats. <br />
	(a) ReadBMP:
		a. Input: Filename of input image
		b. Output: BMP header structure, Image pixel array loaded onto memory
	(b) ConvertFlipGrayscale:
		a. Input: Image pixel array
		b. Output: Grayscale-converted and diagonally flipped (transposed) pixel array
	(c) WriteBMP:
		a. Input: Filename of output (grayscale) image, BMP header structure, Image pixel array
		b. Output: BMP file written on disk
	Use the above functions to read a 256 × 256 24-bit RGB colored Lena image, and write it as an 8-bit
	grayscale onto a different file on the disk. <br />

2. Histogram Equalization and Matching
	Write C++/Image-J modular functions to
	(a) Perform histogram equalization on the 512 × 512 grayscale lena_gray_dark.jpg image. Perform
	    the same on other low-contrast, dark, normal (gray/colored) images.
	(b) Perform histogram matching of the same images with respect to a standard image (e.g.
	    lena.jpg).
	Display the histograms of the original image and the enhanced images and document the
	observations.

3. Spatial Filtering
	Write C/C++ modular functions/subroutines to design spatial filters - mean, median, gradient,
	Laplacian, Sobel kernels (horizontal, vertical, diagonal) on a stack of grayscale images (say, 15
	images per stack)
	Use OpenCV (or) ImageJ for image reading, writing and GUI development only. Use the OpenCV
	tracker (slider) functionality to show outputs for varying sizes of neighborhoods. You may have
	different sliders to select (i) Image (ii) Filter (iii) Neighborhood size
	(a) Input: Path to the stack of images. Input stack should contain the (provided) noisy images, and
	    may also contain the normal test images, e.g. jetplane.jpg, lake.jpg.
	(b) Output: Filtered stack of images should be shown beside input stack in the same pane of GUI 
	    with a slider to vary filter/kernel size/change image.

4. Frequency Filtering
	Write C++/Image-J modular functions to perform the following operations on the 512 × 512
	grayscale test images, e.g. jetplane.jpg, lake.jpg.
	(a) FFT2 (takes input image filename as the argument; gives 2D FFT coefficients as output)
	(b) IFFT2 (takes 2D FFT coefficients as input argument; gives the back-projected/ reconstructed
	    image as output)
	(c) Perform Ideal, Gaussian, and Butterworth low-pass and high-pass filtering, taking cut-off
	    frequency, F, and image filename as input arguments) respectively with <br />
		Ideal_LPF <br />
		Ideal_HPF <br />
		Gaussian_LPF <br /> 
		Gaussian_HPF <br />
		Butterworth_LPF <br />
		Butterworth_HPF <br />

	Display the (shifted) magnitude spectrums of the input, the filter and the filtered output. You may
	make use of the tracker/slider function to choose images, filter types and cut-off frequencies.

5. Morphological Operations
	Write C++/Image-J modular functions to perform the following operations on the given test image,
	ricegrains_mono.bmp. All functions must support binary images.
	Make separate functions for erosion, dilation, opening, and closing of binary images
	(a) ErodeBinary, DilateBinary
		Input: Binary image, structuring element <br />
		Output: Eroded/dilated image <br />
	(b) OpenBinary, CloseBinary
		Input: Binary image, structuring element <br />
		Output: Opened/closed image <br />

##Setup
- Download the directory <br />
- Create a project Win32 Console Application in Visual studio. <br />
- Copy the folder files to the project. <br />
- Include the ".cpp" file as the source file. <br />
- Include necessary OpenCV libraries [https://www.youtube.com/watch?v=LNlnjBLPJd8] <br />
- Run the ".cpp" code <br />

