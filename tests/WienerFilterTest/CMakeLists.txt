ADD_EXECUTABLE(WienerFilterTest WienerFilterTest.cxx)

TARGET_LINK_LIBRARIES(WienerFilterTest
  Clarity
  ${FOUND_CUFFT}
  ${FFTW_LIBRARY}
  ${FFTW_THREADS_LIBRARY}
)
