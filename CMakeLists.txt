cmake_minimum_required(VERSION 3.21)

project(minstd_rand)

include(cmake/defaults.cmake)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        main.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        .
)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
        minstd_rand-defaults
)
