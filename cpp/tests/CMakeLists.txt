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
mujoco_orbit_enable_tsan(test_orbit_core)

add_test(NAME test_orbit_core COMMAND $<TARGET_FILE:test_orbit_core>)

add_executable(test_threaded_plugin
    ${CMAKE_CURRENT_SOURCE_DIR}/test_threaded_plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/orbit_plugin.cc
    ${ORBIT_CORE_SOURCES}
)

target_include_directories(test_threaded_plugin
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugin
)

target_link_libraries(test_threaded_plugin PRIVATE ${MUJOCO_SO} Threads::Threads)
mujoco_orbit_enable_tsan(test_threaded_plugin)

add_test(NAME test_threaded_plugin COMMAND $<TARGET_FILE:test_threaded_plugin>)
