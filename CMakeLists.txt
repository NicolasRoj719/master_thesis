cmake_minimum_required(VERSION 3.16)
project(MasterThesis VERSION 1.0 LANGUAGES CXX)

# Force C++ 20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# INTERFACE targets are ideal for header-only libraries.
add_library(master_thesis_lib INTERFACE)
target_include_directories(master_thesis_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

#-----------------------------------------------------------------------
# 2. COnfigure Unit Test Executable
#-----------------------------------------------------------------------
enable_testing()

#Finde all test .cpp files in test/unit/
file(GLOB TEST_SOURCES "tests/unit/*.cpp")

foreach(TEST_SOURCE ${TEST_SOURCES})

  # Extract executable name from filename (e.g. "tests/unit/example.cpp" -> "test_example")
  get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
  set(EXEC_NAME "test_${TEST_NAME}")

  # Build an executable for each test file
  add_executable(${EXEC_NAME} ${TEST_SOURCE})

  # Link against our library
  target_link_libraries(${EXEC_NAME} PRIVATE master_thesis_lib)

  # Allow tests to access tests/unit headers (e.g, class_test_example.hpp)
  target_include_directories(${EXEC_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit)

  # Automatically pass absolute path of test/fixture as a compile definition
  # Access FIXTURE_DIR in C++ directly as a string literal!
  target_compile_definitions(${EXEC_NAME} PRIVATE
      FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixture/"
  )

  # Register executable with CTest (CMake's built-in test runner)
  add_test(NAME ${EXEC_NAME} COMMAND ${EXEC_NAME})

endforeach()

# ---------------------------------------------------------------------
# 3. Setup Doxygen Target
# ---------------------------------------------------------------------
find_package(Doxygen)
if(DOXYGEN_FOUND)
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # Replace @VARIABLES@ in Doxyfile.in and output Doxyfile in build/
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  # Generate Doxyfile if it does not exist yet, or configure target
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen..."
    VERBATIM
  )

else()
  message(STATUS "Doxygen not found. 'doc' target will not be available.")
endif()

