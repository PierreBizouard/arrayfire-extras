CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ArrayFire-Extras-Tests)

FIND_PACKAGE(ArrayFire REQUIRED)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

FILE(GLOB CPP_FILES "*.cpp")

IF(ArrayFire_Unified_FOUND)
    ADD_EXECUTABLE(test_opencv ${CPP_FILES})
    TARGET_LINK_LIBRARIES(test_opencv ${ArrayFire_Unified_LIBRARIES}
                                      ${OpenCV_LIBS}
                                      ${GTEST_LIBRARIES}
                                      ${CMAKE_THREAD_LIBS_INIT}
                                      )
ENDIF()


