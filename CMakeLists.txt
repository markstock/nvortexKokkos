#
# nvortexKokkos
#
PROJECT (nvortexKokkkos)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()
SET (CMAKE_BUILD_TYPE "Release" CACHE STRING "Select which configuration to build")
SET (Kokkos_ROOT "kokkos" CACHE STRING "Root of kokkos installation")
SET (USE_OMP FALSE CACHE BOOL "Use OpenMP multithreading")
SET (USE_VC FALSE CACHE BOOL "Use Vc for vector arithmetic")
SET (CMAKE_VERBOSE_MAKEFILE on)
SET (CMAKE_EXPORT_COMPILE_COMMANDS on)

IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()

IF (CMAKE_COMPILER_IS_GNUCXX)
  SET (CMAKE_CXX_FLAGS "-Wall -Wformat -std=c++14")
  SET (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  SET (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -g -ggdb3")
ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # The Cray compiler reports as Clang to most versions of CMake
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep -c Cray
                  OUTPUT_VARIABLE INTERNAL_HAVE_CRAY_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  IF (INTERNAL_HAVE_CRAY_COMPILER) #not actually Clang
    # cray compiler misidentifies as Clang and needs this instead
    SET (CMAKE_CXX_FLAGS "-Wall -Wformat -std=c++14")
  ELSE()
    # real clang needs this
    SET (CMAKE_CXX_FLAGS "-Wall -Wformat -std=c++14 -stdlib=libc++")
  ENDIF()
  SET (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  SET (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb")
ELSEIF (MSVC)
  SET (CMAKE_CXX_FLAGS "/Ox")
ENDIF ()

#find_package(Kokkos)
ADD_SUBDIRECTORY(kokkos)

#target_link_libraries(myTarget Kokkos::kokkos)

INCLUDE_DIRECTORIES ( "src" )

ADD_EXECUTABLE ( "nvKok01.bin" "src/nvKok01.cpp" )
TARGET_LINK_LIBRARIES( "nvKok01.bin" Kokkos::kokkos )

