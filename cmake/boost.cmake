# Build Core Boost components

if(NOT USE_SEQ AND NOT USE_MPI AND NOT USE_OMP AND NOT USE_TBB AND NOT USE_STL)
    return()
endif()

SUBDIRLIST(subdirs ${CMAKE_SOURCE_DIR}/3rdparty/boost/libs)
foreach(subd ${subdirs})
        include_directories(${CMAKE_SOURCE_DIR}/3rdparty/boost/libs/${subd}/include)
endforeach()

include(ExternalProject)
ExternalProject_Add(ppc_boost
        SOURCE_DIR        "${CMAKE_SOURCE_DIR}/3rdparty/boost"
        PREFIX            "${CMAKE_CURRENT_BINARY_DIR}/ppc_boost"
        BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/ppc_boost/build"
        INSTALL_DIR       "${CMAKE_CURRENT_BINARY_DIR}/ppc_boost/install"
        CONFIGURE_COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_SOURCE_DIR}/3rdparty/boost/" -B "${CMAKE_CURRENT_BINARY_DIR}/ppc_boost/build/"
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -G${CMAKE_GENERATOR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBOOST_ENABLE_MPI=ON
        -D MPI_INCLUDE_PATH=${MPI_INCLUDE_PATH} -D MPI_LIBRARIES=${MPI_LIBRARIES} -D MPI_COMPILE_FLAGS=${MPI_COMPILE_FLAGS} -D MPI_LINK_FLAGS=${MPI_LINK_FLAGS}
        -DBOOST_INCLUDE_LIBRARIES=mpi
        -D CMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER} -D CMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        BUILD_COMMAND     "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}/ppc_boost/build" --config ${CMAKE_BUILD_TYPE} --parallel
        INSTALL_COMMAND   "${CMAKE_COMMAND}" --install "${CMAKE_CURRENT_BINARY_DIR}/ppc_boost/build" --prefix "${CMAKE_CURRENT_BINARY_DIR}/ppc_boost/install"
        TEST_COMMAND      "")
