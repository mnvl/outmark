
cmake_minimum_required(VERSION 2.8.11)

project(Segm)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=gnu++0x)
endif()

add_subdirectory(src)
