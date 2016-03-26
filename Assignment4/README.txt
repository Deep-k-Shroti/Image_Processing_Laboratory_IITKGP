Frequency Filtering 
Write C++/Image-J modular functions to perform the following operations on the 512 × 512 grayscale test images, e.g. jetplane.jpg, lake.jpg. 

FFT2 (takes input image filename as the argument; gives 2D FFT coefficients as output) 
IFFT2 (takes 2D FFT coefficients as input argument; gives the back-projected/ reconstructed image as output) 
Perform Ideal, Gaussian, and Butterworth low-pass and high-pass filtering, taking cut-off frequency, F, and image filename as input arguments) respectively with 
Ideal_LPF 
Ideal_HPF 
Gaussian_LPF 
Gaussian_HPF 
Butterworth_LPF 
Butterworth_HPF 
Display the (shifted) magnitude spectrums of the input, the filter and the filtered output. You may make use of the tracker/slider function to choose images, filter types and cut-off frequencies. 

Prerequisite: Keep images in the TestImages folder which is in same folder as the '.cpp' file.
Run: Execute the executable after compiling. Use slider to change input image, filter type or filter parameters.
