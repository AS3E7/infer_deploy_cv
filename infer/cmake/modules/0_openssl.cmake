pkg_check_modules(zlib QUIET IMPORTED_TARGET zlib=1.2.13)

if (zlib_FOUND)
    message(STATUS "Found zlib: ${zlib_CONFIG} (found version \"${zlib_VERSION}\")")
else()
    ExternalProject_Add(
        zlib_external
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v1.2.13
        PREFIX ${EXTERNAL_INSTALL_LOCATION}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED=OFF
            -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib/libz.so
    )
    add_library(z STATIC IMPORTED)
    add_dependencies(z zlib_external)
    set_target_properties(z PROPERTIES IMPORTED_LOCATION "${EXTERNAL_INSTALL_LOCATION}/lib/libz.so")
endif()

pkg_check_modules(openssl QUIET libcrypto=1.1.1n libssl=1.1.1n)

if (openssl_FOUND)
    message(STATUS "Found openssl: ${openssl_CONFIG} (found version \"${openssl_VERSION}\")")
else()
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7")
        set(PLATFORM_NAME "linux-armv4")
    else()
        string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME)
        set(PLATFORM_NAME ${SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})
    endif()

    if (zlib_FOUND)
        ExternalProject_Add(
            openssl_external
            GIT_REPOSITORY https://github.com/openssl/openssl.git
            GIT_TAG OpenSSL_1_1_1n
            PREFIX ${EXTERNAL_INSTALL_LOCATION}
            BUILD_IN_SOURCE 1
            CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
                ./Configure ${PLATFORM_NAME} --prefix=${EXTERNAL_INSTALL_LOCATION}
            BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib/libcrypto.so ${EXTERNAL_INSTALL_LOCATION}/lib/libssl.so
        )
    else()
        ExternalProject_Add(
            openssl_external
            GIT_REPOSITORY https://github.com/openssl/openssl.git
            GIT_TAG OpenSSL_1_1_1n
            DEPENDS zlib_external
            PREFIX ${EXTERNAL_INSTALL_LOCATION}
            BUILD_IN_SOURCE 1
            CONFIGURE_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
                ./Configure ${SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR} --prefix=${EXTERNAL_INSTALL_LOCATION}
            BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib/libcrypto.so ${EXTERNAL_INSTALL_LOCATION}/lib/libssl.so
        )
    endif()

    add_library(crypto STATIC IMPORTED)
    add_dependencies(crypto openssl_external)
    set_target_properties(crypto PROPERTIES
        IMPORTED_LOCATION "${EXTERNAL_INSTALL_LOCATION}/lib/libcrypto.so"
        INTERFACE_LINK_LIBRARIES "ssl")
    add_library(ssl STATIC IMPORTED)
    add_dependencies(ssl openssl_external)
    set_target_properties(ssl PROPERTIES IMPORTED_LOCATION "${EXTERNAL_INSTALL_LOCATION}/lib/libssl.so")
endif()

add_library(OpenSSL::Crypto STATIC IMPORTED)
add_dependencies(OpenSSL::Crypto openssl_external)
set_target_properties(OpenSSL::Crypto PROPERTIES
    IMPORTED_LOCATION "${EXTERNAL_INSTALL_LOCATION}/lib/libcrypto.so"
    INTERFACE_LINK_LIBRARIES "ssl")
add_library(OpenSSL::SSL STATIC IMPORTED)
add_dependencies(OpenSSL::SSL openssl_external)
set_target_properties(OpenSSL::SSL PROPERTIES IMPORTED_LOCATION "${EXTERNAL_INSTALL_LOCATION}/lib/libssl.so")

set(LinkLibraries "${LinkLibraries};crypto;ssl")