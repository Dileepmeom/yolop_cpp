cmake_minimum_required(VERSION 3.10)
project(lite_yolop)
set(CMAKE_CXX_STANDARD 17)

# Set the path to lite.ai.toolkit
set(LITE_AI_TOOLKIT_DIR /home/ridebuddy/Documents/yolop_cpp/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64)

# Attempt to find lite.ai.toolkit
find_package(lite.ai.toolkit REQUIRED PATHS ${LITE_AI_TOOLKIT_DIR})

if (lite.ai.toolkit_FOUND)
    # If lite.ai.toolkitConfig.cmake is found, use it
    message(STATUS "lite.ai.toolkit_INCLUDE_DIRS: ${lite.ai.toolkit_INCLUDE_DIRS}")
    message(STATUS "        lite.ai.toolkit_LIBS: ${lite.ai.toolkit_LIBS}")
    message(STATUS "   lite.ai.toolkit_LIBS_DIRS: ${lite.ai.toolkit_LIBS_DIRS}")

    # Link and include directories from lite.ai.toolkit
    include_directories(${lite.ai.toolkit_INCLUDE_DIRS})
    link_directories(${lite.ai.toolkit_LIBS_DIRS})
else()
    # Fallback: Manually set include and library paths
    message(WARNING "lite.ai.toolkit not found via find_package. Using manual paths.")
    include_directories(${LITE_AI_TOOLKIT_DIR}/include)
    link_directories(${LITE_AI_TOOLKIT_DIR}/lib)
endif()

# Add the executable
add_executable(lite_yolop test_lite_yolop.cpp)

# Link lite.ai.toolkit libraries to the executable
target_link_libraries(lite_yolop ${lite.ai.toolkit_LIBS})

