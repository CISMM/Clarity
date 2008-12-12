#include <cstdio>

#include <Clarity.h>

#include "../common/Test_Common.h"


int main(int argc, char* argv[]) {
  float *inputImage, *kernelImage;
  Clarity_Dim3 imageDims; // Image dimensions
  Clarity_Dim3 kernelDims; // Kernel dimensions

  // Initialize image data arrays.
  inputImage  = Test_GenerateTrueImage(&imageDims);
  kernelImage = Test_GenerateGaussianKernel(&kernelDims, 3.0f);

  // Write image and PSF to files.
  FILE *fp = fopen("image_f32.raw", "wb");
  fwrite(inputImage, sizeof(float), imageDims.x*imageDims.y*imageDims.z, fp);
  fclose(fp);

  fp = fopen("psf_f32.raw", "wb");
  fwrite(kernelImage, sizeof(float), kernelDims.x*kernelDims.y*kernelDims.z, fp);
  fclose(fp);

  // Initialize Clarity by registering this application as a client.
  Clarity_Register();

  // We'll create test data here by convolving the input image with the PSF.
  float *convolvedImage = 
    (float *) malloc(sizeof(float)*imageDims.x*imageDims.y*imageDims.z);
  Clarity_Convolve(inputImage, imageDims, kernelImage, kernelDims,
		   convolvedImage);

  // Write out convolved image.
  fp = fopen("convolved_f32.raw", "wb");
  fwrite(convolvedImage, sizeof(float), imageDims.x*imageDims.y*imageDims.z, fp);
  fclose(fp);

  // We need to allocate memory for the deconvolution result
  float *deconvolvedImage = 
    (float *) malloc(sizeof(float)*imageDims.x*imageDims.y*imageDims.z);

  // Now we are ready to apply a deconvolution algorithm. We'll try the
  // maximum likelihood algorithm.
  int iterations = 10;
  Clarity_JansenVanCittertDeconvolve(convolvedImage, imageDims,
                                     kernelImage, kernelDims,
                                     deconvolvedImage, iterations);

  // Write out deconvolved image.
  fp = fopen("deconvolved_f32.raw", "wb");
  fwrite(deconvolvedImage, sizeof(float), imageDims.x*imageDims.y*imageDims.z, fp);
  fclose(fp);

  // See how far off the deconvolved image is from the known input image.
  Test_ReportMatch(inputImage, deconvolvedImage, imageDims);

  // Free up the memory used by images
  free(inputImage);
  free(kernelImage);  
  free(convolvedImage);
  free(deconvolvedImage);

  // Unregister this application as a client.
  Clarity_UnRegister();

  return 0;
}
