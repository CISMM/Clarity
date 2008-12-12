ADD_EXECUTABLE(MaximumLikelihoodTest MaximumLikelihoodTest.cxx)

TARGET_LINK_LIBRARIES(MaximumLikelihoodTest
	Clarity
	fftw3f
	fftw3f_threads
	${FOUND_CUFFT})
