# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bobo/code/TLD

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bobo/code/TLD/build

# Include any dependencies generated for this target.
include src/CMakeFiles/LKTracker.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/LKTracker.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/LKTracker.dir/flags.make

src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o: src/CMakeFiles/LKTracker.dir/flags.make
src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o: ../src/LKTracker.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/bobo/code/TLD/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o"
	cd /home/bobo/code/TLD/build/src && g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/LKTracker.dir/LKTracker.cpp.o -c /home/bobo/code/TLD/src/LKTracker.cpp

src/CMakeFiles/LKTracker.dir/LKTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LKTracker.dir/LKTracker.cpp.i"
	cd /home/bobo/code/TLD/build/src && g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/bobo/code/TLD/src/LKTracker.cpp > CMakeFiles/LKTracker.dir/LKTracker.cpp.i

src/CMakeFiles/LKTracker.dir/LKTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LKTracker.dir/LKTracker.cpp.s"
	cd /home/bobo/code/TLD/build/src && g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/bobo/code/TLD/src/LKTracker.cpp -o CMakeFiles/LKTracker.dir/LKTracker.cpp.s

src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o.requires:
.PHONY : src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o.requires

src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o.provides: src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/LKTracker.dir/build.make src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o.provides.build
.PHONY : src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o.provides

src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o.provides.build: src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o

# Object files for target LKTracker
LKTracker_OBJECTS = \
"CMakeFiles/LKTracker.dir/LKTracker.cpp.o"

# External object files for target LKTracker
LKTracker_EXTERNAL_OBJECTS =

../lib/libLKTracker.a: src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o
../lib/libLKTracker.a: src/CMakeFiles/LKTracker.dir/build.make
../lib/libLKTracker.a: src/CMakeFiles/LKTracker.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../../lib/libLKTracker.a"
	cd /home/bobo/code/TLD/build/src && $(CMAKE_COMMAND) -P CMakeFiles/LKTracker.dir/cmake_clean_target.cmake
	cd /home/bobo/code/TLD/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LKTracker.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/LKTracker.dir/build: ../lib/libLKTracker.a
.PHONY : src/CMakeFiles/LKTracker.dir/build

src/CMakeFiles/LKTracker.dir/requires: src/CMakeFiles/LKTracker.dir/LKTracker.cpp.o.requires
.PHONY : src/CMakeFiles/LKTracker.dir/requires

src/CMakeFiles/LKTracker.dir/clean:
	cd /home/bobo/code/TLD/build/src && $(CMAKE_COMMAND) -P CMakeFiles/LKTracker.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/LKTracker.dir/clean

src/CMakeFiles/LKTracker.dir/depend:
	cd /home/bobo/code/TLD/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bobo/code/TLD /home/bobo/code/TLD/src /home/bobo/code/TLD/build /home/bobo/code/TLD/build/src /home/bobo/code/TLD/build/src/CMakeFiles/LKTracker.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/LKTracker.dir/depend

