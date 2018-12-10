cmake_minimum_required(VERSION 3.0)

project(KissFFT VERSION 1.3.0)

add_library(kiss_fft STATIC
  # main FFT implementation
  kiss_fft.c
  # from the tools/ directory
  tools/kfc.c 
  tools/kiss_fftnd.c 
  tools/kiss_fftndr.c 
  tools/kiss_fftr.c
  )

# on Windows, make sure that the symbols get exported
if(WIN32)
  target_sources(kiss_fft PRIVATE kiss_fft.def)
endif()

target_include_directories(kiss_fft
  PRIVATE .
  PRIVATE tools)

install(TARGETS kiss_fft
  RUNTIME DESTINATION lib
  ARCHIVE DESTINATION lib # for .lib files on Windows
  LIBRARY DESTINATION lib)

install(
  FILES kiss_fft.h tools/kfc.h tools/kiss_fftnd.h tools/kiss_fftndr.h tools/kiss_fftr.h
  DESTINATION include)
