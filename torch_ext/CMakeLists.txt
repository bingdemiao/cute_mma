# uv venvs expose `python` rather than `python3`, so prefer that name
find_program(PYTHON_HINT NAMES python python3)
if(PYTHON_HINT)
    set(Python_EXECUTABLE "${PYTHON_HINT}" CACHE FILEPATH "Python interpreter")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Discover PyTorch cmake path from the Python interpreter
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FIND_RESULT
)
if(NOT TORCH_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to find PyTorch. Ensure torch is installed for ${Python_EXECUTABLE}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)

# Get Python extension suffix
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Build the extension as a shared library
add_library(cute_oft SHARED
    oft_torch_bind.cpp
    oft_torch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cute_oft_coop_pc.cu
)

# Use the Python extension suffix (e.g., .cpython-310-x86_64-linux-gnu.so)
set_target_properties(cute_oft PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
)

target_include_directories(cute_oft PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../default_configs
    ${CUTLASS_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

target_compile_definitions(cute_oft PRIVATE
    OFT_GROUP_SIZE=${GROUP_SIZE}
    OFT_RECONN_SIZE=${RECONN_SIZE}
    TORCH_EXTENSION_NAME=cute_oft
)

# Locate torch_python in the same directory as libtorch
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_link_libraries(cute_oft PRIVATE
    ${TORCH_LIBRARIES}
    "${TORCH_LIB_DIR}/libtorch_python.so"
    ${Python_LIBRARIES}
)

set_target_properties(cute_oft PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
