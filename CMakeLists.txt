cmake_minimum_required(VERSION 3.5)
project(DataMiningProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

if(${OpenCV_VERSION} VERSION_LESS 2.3.1)
    message (FATAL_ERROR "OpenCV version is not compatible: ${OpenCV_VERSION}")
endif()

set(SOURCE_FILES main.cpp)
add_executable(DataMiningProject ${SOURCE_FILES})
target_link_libraries(DataMiningProject ${OpenCV_LIBS})
set_target_properties(DataMiningProject PROPERTIES OUTPUT_NAME carrecognition)