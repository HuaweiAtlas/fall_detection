# CMake lowest version requirement
cmake_minimum_required(VERSION 2.8)

if(NOT DEFINED ENV{DDK_PATH})
	set(ENV{DDK_PATH} /home/syw/tools/che/ddk/ddk)
endif()

#  project information
PROJECT(robot_dog)

# Compile options
add_compile_options(-std=c++11)
# Set compiler
set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++")

set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wall")

add_compile_options(-std=c++11)
add_compile_options(-fPIC)

# Header path
include_directories(
        ${PROJECT_SOURCE_DIR}
		${PROJECT_SOURCE_DIR}/common/include
        $ENV{DDK_PATH}/include/inc/
        $ENV{DDK_PATH}/include/third_party/protobuf/include
        $ENV{DDK_PATH}/include/third_party/cereal/include
        $ENV{DDK_PATH}/include/libc_sec/include
)

# Link path
link_directories("$ENV{DDK_PATH}/host/lib/")
link_directories("${PROJECT_SOURCE_DIR}/out")
set(link_library 
    libhiai_server.so
    libmatrixdaemon.so
    libome.so
    libfmk_common.so
    libdrvdevdrv.so
    libdrvaicpu.so
    libdrvhdc.so
    libdrvhdc_host.so
    libdrvtcp.so
    libmmpa.so
    libcrypto.so
    libc_sec.so
    libruntime.so
    libteec.so
    libpthread.so
    libcce.so
    libcce_aicpudev.so
    libcce_aicore.so
    libaicpu_engine.so
    libprotobuf.so
    librt.so
    libprofilerclient.so
    libslog.so
    libdl.so
    libmedia_mini.so
    libdevmm.so)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_SOURCE_DIR}/out")
SET(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/out")
ADD_SUBDIRECTORY("${PROJECT_SOURCE_DIR}/device_engines")
ADD_SUBDIRECTORY("${PROJECT_SOURCE_DIR}/host_engines")
ADD_SUBDIRECTORY("${PROJECT_SOURCE_DIR}/postprocess_engines")
ADD_SUBDIRECTORY("${PROJECT_SOURCE_DIR}/resize_engines")

# source code
message("[main current dir]:" ${CMAKE_CURRENT_LIST_DIR})
AUX_SOURCE_DIRECTORY(${CMAKE_CURRENT_LIST_DIR} SRC_LIST)

set(TARGET robot_dog)
# Specify executable file to be generated
add_executable(${TARGET} ${SRC_LIST})

# Add link libraries
target_link_libraries(${TARGET} ${link_library})
