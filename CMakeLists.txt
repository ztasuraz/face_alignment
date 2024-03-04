cmake_minimum_required(VERSION 3.10)

set( CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" )


project(HelloWorld)

find_package(OpenCV CONFIG REQUIRED)
find_package(cpptrace CONFIG REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
message("hello world " ${OpenCV_LIB_DIR} ${OpenCV_LIBS})

# Additional Library Directories
link_directories( ${OpenCV_LIB_DIR} )
link_libraries(${OpenCV_LIBS})

add_executable(landmarker landmarker.cpp)
add_library(face_preprocess face_preprocess.hpp)
add_library(data_class data_class.h)


target_link_libraries(face_preprocess PUBLIC ${OpenCV_LIBS})
target_link_libraries(face_preprocess PRIVATE cpptrace::cpptrace)
target_link_libraries(landmarker PRIVATE ${OpenCV_LIBS})
target_link_libraries(landmarker PRIVATE cpptrace::cpptrace)
