cmake_minimum_required(VERSION 3.18)
project(mujoco_orbit_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate MuJoCo via the installed Python package. Wheel layout:
#   <site-packages>/mujoco/include/mujoco/*.h
#   <site-packages>/mujoco/libmujoco.so.<version>
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
    "import mujoco, os, sys; sys.stdout.write(os.path.dirname(mujoco.__file__))"
    OUTPUT_VARIABLE MUJOCO_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
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

add_library(mujoco_orbit_plugin SHARED
    plugin/orbit_plugin.cc
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
