
cmake_minimum_required(VERSION 2.8.11)

project(mrf_segment)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=gnu++14)
endif()

include_directories(third_party/GridCut-1.3/include)
include_directories(third_party/GridCut-1.3/examples/include)

add_subdirectory(src)
