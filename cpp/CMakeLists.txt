cmake_minimum_required(VERSION 3.16)
project(mujoco_orbit_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

# Locate MuJoCo via the installed Python package. Wheel layout:
#   <site-packages>/mujoco/include/mujoco/*.h
#   <site-packages>/mujoco/libmujoco.so.<version>
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

file(GLOB MUJOCO_SO_CANDIDATES "${MUJOCO_PACKAGE_DIR}/libmujoco.so*")
if(NOT MUJOCO_SO_CANDIDATES)
    message(FATAL_ERROR "libmujoco.so not found in ${MUJOCO_PACKAGE_DIR}")
endif()
list(GET MUJOCO_SO_CANDIDATES 0 MUJOCO_SO)
message(STATUS "mujoco_orbit: linking against ${MUJOCO_SO}")

set(ORBIT_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gravity.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/propagator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lvlh.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environment.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/elements.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/coupling_passive.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sensors_plugin.cc
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

target_link_libraries(mujoco_orbit_plugin PRIVATE ${MUJOCO_SO})
set_target_properties(mujoco_orbit_plugin PROPERTIES
    PREFIX ""                       # so file is "mujoco_orbit_plugin.so", not "libmujoco_orbit_plugin.so"
    OUTPUT_NAME "mujoco_orbit_plugin"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/../../mujoco"  # find libmujoco.so at runtime relative to the plugin
)

# scikit-build-core installs relative to the wheel root; land the .so inside the
# Python package so it is shipped alongside mujoco_orbit.
install(TARGETS mujoco_orbit_plugin
    LIBRARY DESTINATION mujoco_orbit/plugins
)

add_subdirectory(tests)
