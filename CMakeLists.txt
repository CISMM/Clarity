CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(Clarity)

option(BUILD_SHARED_LIBS "Build Clarity with shared libraries." OFF)

set( Clarity_MAJOR_VERSION "1")
set( Clarity_MINOR_VERSION "0")
set( Clarity_BUILD_VERSION "0")

set( Clarity_VERSION_FLAT "${Clarity_MAJOR_VERSION}${Clarity_MINOR_VERSION}${Clarity_BUILD_VERSION}" )
set( Clarity_VERSION "${Clarity_MAJOR_VERSION}.${Clarity_MINOR_VERSION}.${Clarity_BUILD_VERSION}" )

include_directories(${Clarity_BINARY_DIR})
include_directories(${Clarity_SOURCE_DIR})

set( Clarity_SRCS
  src/Clarity.cxx
  src/WienerDeconvolve.cxx
  src/JansenVanCittertDeconvolve.cxx
  src/SmoothedJansenVanCittertDeconvolve.cxx
  src/IDivergenceDeconvolve.cxx
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