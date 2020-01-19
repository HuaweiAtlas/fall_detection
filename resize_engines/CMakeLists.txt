if(NOT DEFINED ENV{DDK_PATH})
	set(ENV{DDK_PATH} /home/pass/tools/che/ddk/ddk)
endif()

# CMAKE_CURRENT_LIST_DIR is the directory of current CMakeLists.txt file.
message("[device engine current dir]:" ${CMAKE_CURRENT_LIST_DIR})

# Compile options, please be noted that CPU_ONLY must be added.
add_compile_options(-std=c++11)
add_compile_options(-fPIC)
add_compile_options(-DCPU_ONLY)

# Set compiler
set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++")
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wall")

# Header path
include_directories(
		${PROJECT_SOURCE_DIR}/common/include
        ${CMAKE_CURRENT_LIST_DIR}
        $ENV{DDK_PATH}/include/inc
        $ENV{DDK_PATH}/include/third_party/opencv/include
        $ENV{DDK_PATH}/include/third_party/protobuf/include
        $ENV{DDK_PATH}/include/third_party/cereal/include
        $ENV{DDK_PATH}/include/libc_sec/include
)

# Link path
link_directories("$ENV{DDK_PATH}/device/lib/")
link_directories("${PROJECT_SOURCE_DIR}/out")
set(link_library 
    libDvpp_api.so
    libDvpp_api.so
    libDvpp_jpeg_decoder.so
    libDvpp_jpeg_encoder.so
    libDvpp_vpc.so
    libmedia_mini.so
    libhiai_server.so
    libidedaemon.so
    libhiai_common.so
    libopencv_world.so)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "${PROJECT_SOURCE_DIR}/out")
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/out")

# build device engine
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_LIST_DIR} DEVICE_SRC_LIST)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/common COMMON_SRC_LIST)

set(TARGET Resize)
# Specify .so file to be generated
ADD_LIBRARY(${TARGET} SHARED ${DEVICE_SRC_LIST} ${COMMON_SRC_LIST})

# Add link libraries
target_link_libraries(${TARGET} ${link_library})

