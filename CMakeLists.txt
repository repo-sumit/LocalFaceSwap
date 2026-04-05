cmake_minimum_required(VERSION 3.24)

project(LocalFaceFilterNative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED CONFIG COMPONENTS core highgui imgproc video videoio)

add_executable(local_face_filter
    native/src/main.cpp
)

target_compile_definitions(local_face_filter PRIVATE
    NOMINMAX
)

if(MSVC)
    target_compile_options(local_face_filter PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(local_face_filter PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TARGET opencv_world)
    target_link_libraries(local_face_filter PRIVATE opencv_world)
else()
    target_link_libraries(local_face_filter PRIVATE
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_video
        opencv_videoio
    )
endif()
