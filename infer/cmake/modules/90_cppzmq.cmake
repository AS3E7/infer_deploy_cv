find_package(ZeroMQ QUIET)

if (ZeroMQ_FOUND)
    message(STATUS "Found ZeroMQ: ${ZeroMQ_CONFIG} (found version \"${ZeroMQ_VERSION}\")")
    add_library(cppzmq::cppzmq INTERFACE IMPORTED)
    set_target_properties(cppzmq::cppzmq PROPERTIES INTERFACE_LINK_LIBRARIES "${EXTERNAL_INSTALL_LOCATION}/lib/libzmq.a")
else()
    ExternalProject_Add(
        zmq_external
        GIT_REPOSITORY https://github.com/zeromq/libzmq.git
        GIT_TAG v4.3.4
        PREFIX ${EXTERNAL_INSTALL_LOCATION}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED=OFF -DWITH_LIBSODIUM=OFF -DWITH_LIBBSD=OFF -DWITH_TLS=OFF
            -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        BUILD_BYPRODUCTS ${EXTERNAL_INSTALL_LOCATION}/lib/libzmq.a
    )

    ExternalProject_Add(
        cppzmq_external
        DEPENDS zmq_external
        GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
        GIT_TAG v4.10.0
        PREFIX ${EXTERNAL_INSTALL_LOCATION}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_CXX_FLAGS="-I${EXTERNAL_INSTALL_LOCATION}/include" -DCPPZMQ_BUILD_TESTS=OFF
            -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    )

    add_library(cppzmq::cppzmq INTERFACE IMPORTED)
    add_dependencies(cppzmq::cppzmq cppzmq_external)
    set_target_properties(cppzmq::cppzmq PROPERTIES INTERFACE_LINK_LIBRARIES "${EXTERNAL_INSTALL_LOCATION}/lib/libzmq.a;pthread")
endif()

file(GLOB ZMQ_M_FILES "src/modules/zmq_cv_post/*.c??")
set(LibFiles "${LibFiles};${ZMQ_M_FILES}")
set(LinkLibraries "${LinkLibraries};cppzmq::cppzmq")