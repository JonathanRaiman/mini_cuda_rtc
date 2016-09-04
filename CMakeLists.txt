cmake_minimum_required(VERSION 3.1)
project(runtime_compiler)

find_package(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -w -fPIC")
set(PROJECT_DIR "${PROJECT_SOURCE_DIR}")
set(GENERATED_FILES_DIR ${CMAKE_BINARY_DIR}/runtime_compiler_generated)
file(MAKE_DIRECTORY ${GENERATED_FILES_DIR})

set(SourceDir "src")
file(GLOB_RECURSE FilesInDir "${PROJECT_SOURCE_DIR}/${SourceDir}/*")
foreach(FilePath ${FilesInDir})
	get_filename_component(FileExt ${FilePath} EXT)
    get_filename_component(FileNameWe ${FilePath} NAME_WE)
	if (FileExt STREQUAL ".cpp")
		list(APPEND RTC_SOURCES "${FilePath}")
	elseif(FileExt STREQUAL ".cu")
		list(APPEND RTC_CUDA_SOURCES "${FilePath}")
	elseif(FileExt STREQUAL ".h")
		list(APPEND RTC_HEADERS "${FilePath}")
	elseif(FileExt STREQUAL ".h.in")
		configure_file("${FilePath}" "${GENERATED_FILES_DIR}/${FileNameWe}.h")
	endif()
endforeach()

add_library(rtc_lib ${RTC_SOURCES})
include_directories(${GENERATED_FILES_DIR})

cuda_add_executable(simple_rtc_nvcc ${RTC_CUDA_SOURCES})
cuda_include_directories(${GENERATED_FILES_DIR})
add_dependencies(simple_rtc_nvcc rtc_lib)

target_link_libraries(simple_rtc_nvcc rtc_lib)
