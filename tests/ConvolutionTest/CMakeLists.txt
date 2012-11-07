ADD_EXECUTABLE(ConvolutionTest ConvolutionTest.cxx)

TARGET_LINK_LIBRARIES(ConvolutionTest
  Clarity
  ${FOUND_CUFFT}
  ${FFTW_LIBRARY}
  ${FFTW_THREADS_LIBRARY}
)
