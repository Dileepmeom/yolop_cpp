# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ridebuddy/Documents/yolop_cpp/yolop_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ridebuddy/Documents/yolop_cpp/yolop_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/lite_yolop.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lite_yolop.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lite_yolop.dir/flags.make

CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.o: CMakeFiles/lite_yolop.dir/flags.make
CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.o: ../test_lite_yolop.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ridebuddy/Documents/yolop_cpp/yolop_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.o -c /home/ridebuddy/Documents/yolop_cpp/yolop_cpp/test_lite_yolop.cpp

CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ridebuddy/Documents/yolop_cpp/yolop_cpp/test_lite_yolop.cpp > CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.i

CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ridebuddy/Documents/yolop_cpp/yolop_cpp/test_lite_yolop.cpp -o CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.s

# Object files for target lite_yolop
lite_yolop_OBJECTS = \
"CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.o"

# External object files for target lite_yolop
lite_yolop_EXTERNAL_OBJECTS =

lite_yolop: CMakeFiles/lite_yolop.dir/test_lite_yolop.cpp.o
lite_yolop: CMakeFiles/lite_yolop.dir/build.make
lite_yolop: CMakeFiles/lite_yolop.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ridebuddy/Documents/yolop_cpp/yolop_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lite_yolop"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lite_yolop.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lite_yolop.dir/build: lite_yolop

.PHONY : CMakeFiles/lite_yolop.dir/build

CMakeFiles/lite_yolop.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lite_yolop.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lite_yolop.dir/clean

CMakeFiles/lite_yolop.dir/depend:
	cd /home/ridebuddy/Documents/yolop_cpp/yolop_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ridebuddy/Documents/yolop_cpp/yolop_cpp /home/ridebuddy/Documents/yolop_cpp/yolop_cpp /home/ridebuddy/Documents/yolop_cpp/yolop_cpp/build /home/ridebuddy/Documents/yolop_cpp/yolop_cpp/build /home/ridebuddy/Documents/yolop_cpp/yolop_cpp/build/CMakeFiles/lite_yolop.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lite_yolop.dir/depend

