# 1. sample 
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sample DESTINATION ${CMAKE_INSTALL_PREFIX}/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sample/CMakeLists.txt DESTINATION ${CMAKE_INSTALL_PREFIX}/)

# 2. script
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/script)

# 3. dockerfile 
# common install
install(DIRECTORY ${CMAKE_SOURCE_DIR}/docker/common/ DESTINATION ${CMAKE_INSTALL_PREFIX}/docker/common)
install(FILES ${CMAKE_SOURCE_DIR}/docker/install_build.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/docker)
install(FILES ${CMAKE_SOURCE_DIR}/docker/install/install_common.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/docker/install)

# bmnn 
if(BUILD_TARGET_CHIP STREQUAL "bmnn")
    install(FILES ${CMAKE_SOURCE_DIR}/docker/dockerfile/bmnn_build.Dockerfile DESTINATION ${CMAKE_INSTALL_PREFIX}/docker/)
    install(FILES ${CMAKE_SOURCE_DIR}/docker/dockerfile/bmnn_runtime.Dockerfile DESTINATION ${CMAKE_INSTALL_PREFIX}/docker/)
    install(FILES ${CMAKE_SOURCE_DIR}/docker/install/install_bmnn.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/docker/install)
endif()

# tingsmicro 
if(BUILD_TARGET_CHIP STREQUAL "ts")
    install(FILES ${CMAKE_SOURCE_DIR}/docker/dockerfile/ts.Dockerfile DESTINATION ${CMAKE_INSTALL_PREFIX}/docker/)
    install(FILES ${CMAKE_SOURCE_DIR}/docker/dockerfile/ts.Dockerfile DESTINATION ${CMAKE_INSTALL_PREFIX}/docker/)
    # install(FILES ${CMAKE_SOURCE_DIR}/docker/install/install_ts.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/docker/install)
endif()

# 4. docs
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs/ DESTINATION ${CMAKE_INSTALL_PREFIX}/docs)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md DESTINATION ${CMAKE_INSTALL_PREFIX}/)