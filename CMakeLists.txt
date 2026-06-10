cmake_minimum_required(VERSION 3.16)

project(pp_doclayout_ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED GGML_DIR AND DEFINED ENV{GGML_DIR})
  set(GGML_DIR "$ENV{GGML_DIR}")
endif()

if(NOT GGML_DIR)
  message(FATAL_ERROR "Set GGML_DIR to a ggml checkout, for example: cmake -S native/pp-doclayout-ggml -B build -DGGML_DIR=/path/to/ggml")
endif()

set(GGML_DIR "${GGML_DIR}" CACHE PATH "Path to a ggml source checkout")

set(GGML_NATIVE OFF CACHE BOOL "Avoid host-specific flags in app builds" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)

add_compile_definitions(GGML_MAX_NAME=256)
add_subdirectory("${GGML_DIR}" ggml-build EXCLUDE_FROM_ALL)

add_library(pp_doclayout_ggml STATIC
  pp_doclayout_ggml.cpp
  pp_doclayout_ggml_ffi.cpp
)

target_include_directories(pp_doclayout_ggml PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(pp_doclayout_ggml PUBLIC ggml)

add_executable(pp-doclayout-ggml
  main.cpp
)

target_link_libraries(pp-doclayout-ggml PRIVATE pp_doclayout_ggml)

if(APPLE)
  target_link_libraries(pp_doclayout_ggml PUBLIC
    "-framework Foundation"
    "-framework Accelerate"
  )
endif()
