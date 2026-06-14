cmake_minimum_required(VERSION 3.16)
project(mjorbit_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MJORBIT_ENABLE_TSAN "Build C++ tests with ThreadSanitizer" OFF)
option(MJORBIT_BUILD_TESTS "Build the C++ test executables" ON)

if(MJORBIT_BUILD_TESTS)
    enable_testing()
endif()
find_package(Threads REQUIRED)

if(MJORBIT_ENABLE_TSAN)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
    check_cxx_source_compiles("int main() { return 0; }" MJORBIT_TSAN_AVAILABLE)
    unset(CMAKE_REQUIRED_FLAGS)
    if(NOT MJORBIT_TSAN_AVAILABLE)
        message(FATAL_ERROR "MJORBIT_ENABLE_TSAN=ON, but this compiler cannot link a ThreadSanitizer binary")
    endif()
endif()

function(mjorbit_enable_tsan target)
    if(NOT MJORBIT_ENABLE_TSAN)
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
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
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
message(STATUS "mjorbit: using mujoco from ${MUJOCO_PACKAGE_DIR}")

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
message(STATUS "mjorbit: linking against ${MUJOCO_LIBRARY}")
get_filename_component(MUJOCO_LIBRARY_BASENAME "${MUJOCO_LIBRARY}" NAME)

execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NANOBIND_QUERY_RESULT
    ERROR_VARIABLE NANOBIND_QUERY_ERROR
)
if(NOT NANOBIND_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate nanobind via ${Python_EXECUTABLE}: ${NANOBIND_QUERY_ERROR}")
endif()
find_package(nanobind CONFIG REQUIRED)

if(APPLE)
    set(MJORBIT_INSTALL_RPATH "${MUJOCO_PACKAGE_DIR}")
    set(MJORBIT_PLUGIN_INSTALL_RPATH
        "${MUJOCO_PACKAGE_DIR};@loader_path/../../mujoco;@loader_path/../mujoco")
    set(MJORBIT_BINDINGS_INSTALL_RPATH
        "${MUJOCO_PACKAGE_DIR};@loader_path/../mujoco;@loader_path/../../mujoco")
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
    message(STATUS "mjorbit: MuJoCo dylib id is ${MUJOCO_DYLIB_ID}")
    set(MJORBIT_MUJOCO_RUNTIME_NAME "${MUJOCO_LIBRARY}")
elseif(UNIX)
    set(MJORBIT_INSTALL_RPATH "${MUJOCO_PACKAGE_DIR}")
    set(MJORBIT_PLUGIN_INSTALL_RPATH
        "${MUJOCO_PACKAGE_DIR};$ORIGIN/../../mujoco;$ORIGIN/../mujoco")
    set(MJORBIT_BINDINGS_INSTALL_RPATH
        "${MUJOCO_PACKAGE_DIR};$ORIGIN/../mujoco;$ORIGIN/../../mujoco")
else()
    set(MJORBIT_INSTALL_RPATH "")
    set(MJORBIT_PLUGIN_INSTALL_RPATH "")
    set(MJORBIT_BINDINGS_INSTALL_RPATH "")
endif()

function(mjorbit_fix_macos_mujoco_id target)
    if(NOT APPLE)
        return()
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_INSTALL_NAME_TOOL}
            -change "${MUJOCO_DYLIB_ID}"
                    "${MJORBIT_MUJOCO_RUNTIME_NAME}"
                    "$<TARGET_FILE:${target}>"
        COMMAND ${CMAKE_INSTALL_NAME_TOOL}
            -change "@rpath/${MUJOCO_LIBRARY_BASENAME}"
                    "${MJORBIT_MUJOCO_RUNTIME_NAME}"
                    "$<TARGET_FILE:${target}>"
        COMMAND /usr/bin/codesign
            --force --sign - "$<TARGET_FILE:${target}>"
        COMMENT "Rewriting MuJoCo dylib install name for ${target}"
    )
endfunction()

set(ORBIT_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gravity.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/propagator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/orbit_schedule.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lvlh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environment.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/elements.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/orbit_cache.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/coupling_passive.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensors_plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rollout.cc
)

add_library(mjorbit_core STATIC
    ${ORBIT_CORE_SOURCES}
)

target_include_directories(mjorbit_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
)

target_link_libraries(mjorbit_core PUBLIC ${MUJOCO_LIBRARY})
mjorbit_enable_tsan(mjorbit_core)
set_target_properties(mjorbit_core PROPERTIES
    BUILD_RPATH "${MUJOCO_PACKAGE_DIR}"
    INSTALL_RPATH "${MJORBIT_INSTALL_RPATH}"
)

add_library(mjorbit_plugin SHARED
    plugin/orbit_plugin.cc
)

target_include_directories(mjorbit_plugin
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)

target_link_libraries(mjorbit_plugin PRIVATE mjorbit_core ${MUJOCO_LIBRARY})
mjorbit_enable_tsan(mjorbit_plugin)
mjorbit_fix_macos_mujoco_id(mjorbit_plugin)
set_target_properties(mjorbit_plugin PROPERTIES
    PREFIX ""
    OUTPUT_NAME "mjorbit_plugin"
    BUILD_RPATH "${MUJOCO_PACKAGE_DIR}"
    INSTALL_RPATH "${MJORBIT_PLUGIN_INSTALL_RPATH}"
)

# scikit-build-core installs relative to the wheel root; land the .so inside the
# Python package so it is shipped alongside mjorbit.
install(TARGETS mjorbit_plugin
    LIBRARY DESTINATION mjorbit/plugins
)

nanobind_add_module(mjorbit_bindings
    bindings/module.cc
    src/runtime.cc
    src/runtime_model.cc
    src/runtime_sensors.cc
    src/runtime_spec.cc
    src/runtime_state.cc
)

target_include_directories(mjorbit_bindings
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/plugin
)
target_link_libraries(mjorbit_bindings PRIVATE mjorbit_core ${MUJOCO_LIBRARY})
mjorbit_fix_macos_mujoco_id(mjorbit_bindings)
set_target_properties(mjorbit_bindings PROPERTIES
    OUTPUT_NAME "_bindings"
    BUILD_RPATH "${MUJOCO_PACKAGE_DIR}"
    INSTALL_RPATH "${MJORBIT_BINDINGS_INSTALL_RPATH}"
)
install(TARGETS mjorbit_bindings
    LIBRARY DESTINATION mjorbit
)

if(APPLE)
    install(CODE "
        file(GLOB _mjorbit_binding_paths \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/mjorbit/_bindings*.so\")
        foreach(_mjorbit_path IN LISTS _mjorbit_binding_paths)
            execute_process(COMMAND \"${CMAKE_INSTALL_NAME_TOOL}\"
                -change \"@rpath/${MUJOCO_LIBRARY_BASENAME}\"
                        \"@loader_path/../mujoco/${MUJOCO_LIBRARY_BASENAME}\"
                        \"\${_mjorbit_path}\" ERROR_QUIET)
            execute_process(COMMAND \"${CMAKE_INSTALL_NAME_TOOL}\"
                -change \"${MUJOCO_DYLIB_ID}\"
                        \"@loader_path/../mujoco/${MUJOCO_LIBRARY_BASENAME}\"
                        \"\${_mjorbit_path}\" ERROR_QUIET)
            execute_process(COMMAND \"${CMAKE_INSTALL_NAME_TOOL}\"
                -change \"${MUJOCO_LIBRARY}\"
                        \"@loader_path/../mujoco/${MUJOCO_LIBRARY_BASENAME}\"
                        \"\${_mjorbit_path}\" ERROR_QUIET)
            execute_process(COMMAND \"${CMAKE_INSTALL_NAME_TOOL}\"
                -add_rpath \"@loader_path/../mujoco\"
                           \"\${_mjorbit_path}\" ERROR_QUIET)
            execute_process(COMMAND /usr/bin/codesign
                --force --sign - \"\${_mjorbit_path}\" ERROR_QUIET)
        endforeach()

        file(GLOB _mjorbit_plugin_paths \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/mjorbit/plugins/mjorbit_plugin*.dylib\")
        foreach(_mjorbit_path IN LISTS _mjorbit_plugin_paths)
            execute_process(COMMAND \"${CMAKE_INSTALL_NAME_TOOL}\"
                -change \"@rpath/${MUJOCO_LIBRARY_BASENAME}\"
                        \"@loader_path/../../mujoco/${MUJOCO_LIBRARY_BASENAME}\"
                        \"\${_mjorbit_path}\" ERROR_QUIET)
            execute_process(COMMAND \"${CMAKE_INSTALL_NAME_TOOL}\"
                -change \"${MUJOCO_DYLIB_ID}\"
                        \"@loader_path/../../mujoco/${MUJOCO_LIBRARY_BASENAME}\"
                        \"\${_mjorbit_path}\" ERROR_QUIET)
            execute_process(COMMAND \"${CMAKE_INSTALL_NAME_TOOL}\"
                -change \"${MUJOCO_LIBRARY}\"
                        \"@loader_path/../../mujoco/${MUJOCO_LIBRARY_BASENAME}\"
                        \"\${_mjorbit_path}\" ERROR_QUIET)
            execute_process(COMMAND \"${CMAKE_INSTALL_NAME_TOOL}\"
                -add_rpath \"@loader_path/../../mujoco\"
                           \"\${_mjorbit_path}\" ERROR_QUIET)
            execute_process(COMMAND /usr/bin/codesign
                --force --sign - \"\${_mjorbit_path}\" ERROR_QUIET)
        endforeach()
    ")
endif()

if(MJORBIT_BUILD_TESTS)
    add_subdirectory(tests)
endif()
