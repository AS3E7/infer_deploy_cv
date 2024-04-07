pkg_check_modules(curl QUIET IMPORTED_TARGET libcurl)

if (curl_FOUND)
    message(STATUS "Found CURL: ${CURL_CONFIG} (found version \"${CURL_VERSION}\")")
else()
    ExternalProject_Add(
        curl_external
        URL http://cacher.devops.io/api/cacher/files/f98bdb06c0f52bdd19e63c4a77b5eb19b243bcbbd0f5b002b9f3cba7295a3a42
        DEPENDS openssl_external
        PREFIX ${EXTERNAL_INSTALL_LOCATION}
        BUILD_IN_SOURCE 1
        BUILD_COMMAND CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} PKG_CONFIG_PATH=${EXTERNAL_INSTALL_LOCATION}/lib/pkgconfig ./configure --prefix=${EXTERNAL_INSTALL_LOCATION} --host=${CMAKE_SYSTEM_PROCESSOR} --with-openssl
        INSTALL_COMMAND make install
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib/libcurl.a
    )
    add_library(curl STATIC IMPORTED)
    add_dependencies(curl curl_external)
    set_target_properties(curl PROPERTIES
        IMPORTED_LOCATION "${EXTERNAL_INSTALL_LOCATION}/lib/libcurl.a"
        INTERFACE_LINK_LIBRARIES "crypto;ssl;z")
endif()

set(LinkLibraries "${LinkLibraries};curl;crypto;ssl;z")