add_executable(test_orbit_core
    ${CMAKE_CURRENT_SOURCE_DIR}/test_orbit_core.cc
    ${ORBIT_CORE_SOURCES}
)

target_include_directories(test_orbit_core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_test(NAME test_orbit_core COMMAND test_orbit_core)
