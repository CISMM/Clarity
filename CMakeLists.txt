CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(Clarity)

option(BUILD_SHARED_LIBS "Build Clarity with shared libraries." OFF)

FIND_LIBRARY(FFTW3_LIBRARY Clarity C:/NSRG)
IF(FFTW3_LIBRARY)
  LINK_LIBRARIES(${FFTW3_LIBRARY})
ENDIF(FFTW3_LIBRARY)

FIND_PATH(FFTW3_INCLUDE_DIR fftw3.h C:/NSRG)
IF(FFTW3_INCLUDE_DIR)
  INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIR})
ENDIF(FFTW3_INCLUDE_DIR)

set( Clarity_MAJOR_VERSION "1")
set( Clarity_MINOR_VERSION "0")
set( Clarity_BUILD_VERSION "0")

set( Clarity_VERSION_FLAT "${Clarity_MAJOR_VERSION}${Clarity_MINOR_VERSION}${Clarity_BUILD_VERSION}" )
set( Clarity_VERSION "${Clarity_MAJOR_VERSION}.${Clarity_MINOR_VERSION}.${Clarity_BUILD_VERSION}" )

include_directories(${Clarity_BINARY_DIR})
include_directories(${Clarity_SOURCE_DIR} ${Clarity_SOURCE_DIR}/include)

set( Clarity_SRCS
  src/Clarity.cxx
  src/FFT.cxx
  src/Stopwatch.cxx
  src/WienerDeconvolve.cxx
  src/JansenVanCittertDeconvolve.cxx
  src/SmoothedJansenVanCittertDeconvolve.cxx
  src/IDivergenceDeconvolve.cxx
  src/MaximumLikelihoodDeconvolve.cxx
  )
  
configure_file(
  ${Clarity_SOURCE_DIR}/Clarity.h.in
  ${Clarity_BINARY_DIR}/Clarity.h
)

add_library( Clarity ${Clarity_SRCS} )
set_target_properties(Clarity PROPERTIES 
  VERSION ${Clarity_VERSION}
  SOVERSION ${Clarity_MAJOR_VERSION}.${Clarity_MINOR_VERSION}
)