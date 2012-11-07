ADD_EXECUTABLE(MaximumLikelihoodTest MaximumLikelihoodTest.cxx)

TARGET_LINK_LIBRARIES(MaximumLikelihoodTest
  Clarity
  ${FOUND_CUFFT}
  ${FFTW_LIBRARY}
  ${FFTW_THREADS_LIBRARY}
)
