add_executable(test_orbit_core
    ${CMAKE_CURRENT_SOURCE_DIR}/test_orbit_core.cc
    ${ORBIT_CORE_SOURCES}
)

target_include_directories(test_orbit_core
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugin
)

target_link_libraries(test_orbit_core PRIVATE ${MUJOCO_SO})

add_test(NAME test_orbit_core COMMAND $<TARGET_FILE:test_orbit_core>)
