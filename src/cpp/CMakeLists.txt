cmake_minimum_required(VERSION 3.16)
project(mujoco_orbit_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MUJOCO_ORBIT_ENABLE_TSAN "Build C++ tests with ThreadSanitizer" OFF)

enable_testing()
find_package(Threads REQUIRED)

if(MUJOCO_ORBIT_ENABLE_TSAN)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
    check_cxx_source_compiles("int main() { return 0; }" MUJOCO_ORBIT_TSAN_AVAILABLE)
    unset(CMAKE_REQUIRED_FLAGS)
    if(NOT MUJOCO_ORBIT_TSAN_AVAILABLE)
        message(FATAL_ERROR "MUJOCO_ORBIT_ENABLE_TSAN=ON, but this compiler cannot link a ThreadSanitizer binary")
    endif()
endif()

function(mujoco_orbit_enable_tsan target)
    if(NOT MUJOCO_ORBIT_ENABLE_TSAN)
        return()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer -g)
        target_link_options(${target} PRIVATE -fsanitize=thread)
    else()
        message(WARNING "ThreadSanitizer is not configured for compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endfunction()

# Locate MuJoCo via the installed Python package. Wheel layout:
#   <site-packages>/mujoco/include/mujoco/*.h
#   <site-packages>/mujoco/libmujoco.so.<version>   (Linux)
#   <site-packages>/mujoco/libmujoco.<version>.dylib (macOS)
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
    "import mujoco, os, sys; sys.stdout.write(os.path.dirname(mujoco.__file__))"
    OUTPUT_VARIABLE MUJOCO_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE MUJOCO_QUERY_RESULT
    ERROR_VARIABLE MUJOCO_QUERY_ERROR
)
if(NOT MUJOCO_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate mujoco package via ${Python_EXECUTABLE}: ${MUJOCO_QUERY_ERROR}")
endif()
message(STATUS "mujoco_orbit: using mujoco from ${MUJOCO_PACKAGE_DIR}")

set(MUJOCO_INCLUDE_DIR "${MUJOCO_PACKAGE_DIR}/include")
if(NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
    message(FATAL_ERROR "mujoco.h not found at ${MUJOCO_INCLUDE_DIR}")
endif()

file(GLOB MUJOCO_LIBRARY_CANDIDATES
    "${MUJOCO_PACKAGE_DIR}/libmujoco${CMAKE_SHARED_LIBRARY_SUFFIX}"
    "${MUJOCO_PACKAGE_DIR}/libmujoco${CMAKE_SHARED_LIBRARY_SUFFIX}*"
    "${MUJOCO_PACKAGE_DIR}/libmujoco*${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
if(NOT MUJOCO_LIBRARY_CANDIDATES)
    message(FATAL_ERROR
        "MuJoCo shared library not found in ${MUJOCO_PACKAGE_DIR} "
        "(looked for libmujoco*${CMAKE_SHARED_LIBRARY_SUFFIX})"
    )
endif()
list(SORT MUJOCO_LIBRARY_CANDIDATES)
list(GET MUJOCO_LIBRARY_CANDIDATES 0 MUJOCO_LIBRARY)
message(STATUS "mujoco_orbit: linking against ${MUJOCO_LIBRARY}")
get_filename_component(MUJOCO_LIBRARY_BASENAME "${MUJOCO_LIBRARY}" NAME)

if(APPLE)
    set(MUJOCO_ORBIT_INSTALL_RPATH "@loader_path/../../mujoco")
    set(MUJOCO_ORBIT_MUJOCO_RUNTIME_NAME "@rpath/${MUJOCO_LIBRARY_BASENAME}")
    execute_process(
        COMMAND otool -D "${MUJOCO_LIBRARY}"
        OUTPUT_VARIABLE MUJOCO_DYLIB_ID_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE MUJOCO_DYLIB_ID_RESULT
    )
    if(NOT MUJOCO_DYLIB_ID_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to inspect MuJoCo dylib id with otool")
    endif()

    string(REPLACE "\n" ";" MUJOCO_DYLIB_ID_LINES "${MUJOCO_DYLIB_ID_OUTPUT}")
    unset(MUJOCO_DYLIB_ID)
    foreach(line IN LISTS MUJOCO_DYLIB_ID_LINES)
        if(line MATCHES "^@")
            if(DEFINED MUJOCO_DYLIB_ID AND NOT line STREQUAL MUJOCO_DYLIB_ID)
                message(FATAL_ERROR
                    "MuJoCo dylib has inconsistent architecture ids: ${MUJOCO_DYLIB_ID_OUTPUT}"
                )
            endif()
            set(MUJOCO_DYLIB_ID "${line}")
        endif()
    endforeach()
    if(NOT DEFINED MUJOCO_DYLIB_ID)
        message(FATAL_ERROR "Could not parse MuJoCo dylib id from: ${MUJOCO_DYLIB_ID_OUTPUT}")
    endif()
    message(STATUS "mujoco_orbit: MuJoCo dylib id is ${MUJOCO_DYLIB_ID}")
elseif(UNIX)
    set(MUJOCO_ORBIT_INSTALL_RPATH "$ORIGIN/../../mujoco")
else()
    set(MUJOCO_ORBIT_INSTALL_RPATH "")
endif()

function(mujoco_orbit_fix_macos_mujoco_id target)
    if(NOT APPLE)
        return()
    endif()
    if(MUJOCO_DYLIB_ID STREQUAL MUJOCO_ORBIT_MUJOCO_RUNTIME_NAME)
        return()
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_INSTALL_NAME_TOOL}
            -change "${MUJOCO_DYLIB_ID}"
                    "${MUJOCO_ORBIT_MUJOCO_RUNTIME_NAME}"
                    "$<TARGET_FILE:${target}>"
        COMMENT "Rewriting MuJoCo dylib install name for ${target}"
    )
endfunction()

set(ORBIT_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gravity.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/propagator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lvlh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environment.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/elements.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/orbit_cache.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/coupling_passive.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensors_plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rollout.cc
)

add_library(mujoco_orbit_plugin SHARED
    plugin/orbit_plugin.cc
    ${ORBIT_CORE_SOURCES}
)

target_include_directories(mujoco_orbit_plugin
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)

target_link_libraries(mujoco_orbit_plugin PRIVATE ${MUJOCO_LIBRARY})
mujoco_orbit_enable_tsan(mujoco_orbit_plugin)
mujoco_orbit_fix_macos_mujoco_id(mujoco_orbit_plugin)
set_target_properties(mujoco_orbit_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "mujoco_orbit_plugin"
    BUILD_RPATH "${MUJOCO_PACKAGE_DIR}"
    INSTALL_RPATH "${MUJOCO_ORBIT_INSTALL_RPATH}"
)

# scikit-build-core installs relative to the wheel root; land the .so inside the
# Python package so it is shipped alongside mujoco_orbit.
install(TARGETS mujoco_orbit_plugin
    LIBRARY DESTINATION mujoco_orbit/plugins
)

add_subdirectory(tests)
