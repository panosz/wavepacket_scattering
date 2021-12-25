cmake_minimum_required(VERSION 3.15...3.19)
include(CMakePrintHelpers)

project(scikit_build_example VERSION "0.0.1")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  cmake_print_variables(PYTHON_EXECUTABLE)
  cmake_print_variables(_tmp_dir)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
# endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)
find_package (Eigen3 REQUIRED)

pybind11_add_module(_wavepacket MODULE src/bindings.cpp)

add_library(mylibrary src/my_lib.hpp src/my_lib.cpp src/wavepacket.cpp src/wavepacket.hpp)

target_link_libraries(mylibrary PUBLIC Eigen3::Eigen)
target_link_libraries(_wavepacket PUBLIC mylibrary Eigen3::Eigen)

target_compile_definitions(_wavepacket PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_compile_features(mylibrary PUBLIC cxx_std_17)
target_compile_features(_wavepacket PUBLIC cxx_std_17)
install(TARGETS _wavepacket DESTINATION .)
