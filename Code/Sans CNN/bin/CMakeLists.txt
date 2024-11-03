cmake_minimum_required(VERSION 3.0)

project(main)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Trouver OpenCV
find_package(OpenCV REQUIRED)

include_directories(../include)
link_directories(../lib)

file(GLOB SOURCES ../src/*.cpp)

add_executable(main ${SOURCES})

# Lier OpenCV à l'exécutable
target_link_libraries(main ${OpenCV_LIBS} -Ofast -lX11 -lpthread -static-libstdc++)
