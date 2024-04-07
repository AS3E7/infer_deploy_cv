find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(clipper QUIET IMPORTED_TARGET Clipper2)

    if (clipper_FOUND)
        include_directories(${clipper_INCLUDE_DIRS})
        link_directories(${clipper_LIBRARY_DIRS})
        set(LinkLibraries "${LinkLibraries};Clipper2")
    endif()
endif()