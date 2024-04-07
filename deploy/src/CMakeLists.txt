
file(GLOB_RECURSE SRC_FILES "api/*.cpp??" "app/*.cpp??" "codec/*.c??" "common/*.c??" "core/util/*.c??" "core/*.c??" "model/*.c??")

# message(${SRC_FILES})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/api)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/register)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tracker)