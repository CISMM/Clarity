ADD_EXECUTABLE(ConvolutionTest ConvolutionTest.cxx)

TARGET_LINK_LIBRARIES(ConvolutionTest
	Clarity
	fftw3f 
	fftw3f_threads
	${FOUND_CUFFT})
