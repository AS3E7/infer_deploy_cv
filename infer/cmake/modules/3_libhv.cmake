find_package(libhv QUIET)

if (libhv_FOUND)
    message(STATUS "Found libhv: ${libhv_CONFIG} (found version \"${libhv_VERSION}\")")
else()

if (openssl_FOUND)
    ExternalProject_Add(
        libhv_external
        GIT_REPOSITORY https://github.com/ithewei/libhv.git
        GIT_TAG v1.2.6
        PREFIX ${EXTERNAL_INSTALL_LOCATION}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DCMAKE_FIND_ROOT_PATH=${CMAKE_BINARY_DIR}/external
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS="-I${CMAKE_BINARY_DIR}/external/include" -DBUILD_SHARED=OFF -DWITH_OPENSSL=ON
            -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib/libhv_static.a
    )
else()
    ExternalProject_Add(
        libhv_external
        DEPENDS openssl_external
        GIT_REPOSITORY https://github.com/ithewei/libhv.git
        GIT_TAG v1.2.6
        PREFIX ${EXTERNAL_INSTALL_LOCATION}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -DCMAKE_FIND_ROOT_PATH=${CMAKE_BINARY_DIR}/external
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_C_FLAGS="-I${CMAKE_BINARY_DIR}/external/include" -DBUILD_SHARED=OFF -DWITH_OPENSSL=ON
            -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib/libhv_static.a
    )
endif()

    add_library(hv_static STATIC IMPORTED)
    add_dependencies(hv_static libhv_external)
    set_target_properties(hv_static PROPERTIES
        IMPORTED_LOCATION "${EXTERNAL_INSTALL_LOCATION}/lib/libhv_static.a"
        INTERFACE_LINK_LIBRARIES "crypto;ssl")
endif()

file(GLOB CodeFiles
    src/inference_server/*.c??
    src/modules/network/downloader.cpp
)

set(LibFiles "${LibFiles};${CodeFiles}")
set(LinkLibraries "${LinkLibraries};hv_static")