ADD_EXECUTABLE(WienerFilterTest WienerFilterTest.cxx)

TARGET_LINK_LIBRARIES(WienerFilterTest
	Clarity
	fftw3f
	fftw3f_threads
	${FOUND_CUFFT})
