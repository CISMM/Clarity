ADD_EXECUTABLE(JansenVanCittertTest JansenVanCittertTest.cxx)

TARGET_LINK_LIBRARIES(JansenVanCittertTest
	Clarity
	fftw3f
	fftw3f_threads
	${FOUND_CUFFT})
