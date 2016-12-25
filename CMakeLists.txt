
cmake_minimum_required(VERSION 2.8.11)

project(mrf_segment)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=gnu++14 -Wno-deprecated-declarations)
endif()

set(CMAKE_BUILD_TYPE RelWithDebInfo)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(GFlags REQUIRED)

set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.58 COMPONENTS filesystem serialization REQUIRED)

include_directories(third_party)
include_directories(third_party/GridCut-1.3/include)
include_directories(third_party/GridCut-1.3/examples/include)

add_subdirectory(src)
