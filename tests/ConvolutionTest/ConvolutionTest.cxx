#include <cstdio>

#include "Clarity.h"

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

  // Unregister
  Clarity_UnRegister();

  free(inputImage);
  free(kernelImage);
  free(convolvedImage);

  return 0;
}
