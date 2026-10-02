cmake_minimum_required(VERSION 3.16)
project(zero_engine VERSION 0.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ZERO_SOURCES
    src/main.cpp
    src/uci.cpp
    src/bitboard.cpp
    src/position.cpp
    src/movegen.cpp
    src/movepick.cpp
    src/zobrist.cpp
    src/tt.cpp
    src/search.cpp
    src/qsearch.cpp
    src/search_root.cpp
    src/thread.cpp
    src/evaluation.cpp
    src/evaluation/material.cpp
    src/pst.cpp
)

add_executable(zero ${ZERO_SOURCES})
target_include_directories(zero PRIVATE src)
target_compile_options(zero PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
    add_executable(perft_test tests/perft.cpp src/bitboard.cpp src/position.cpp src/movegen.cpp src/zobrist.cpp)
    target_include_directories(perft_test PRIVATE src)
    target_compile_options(perft_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME perft COMMAND perft_test)

    add_executable(search_smoke_test
        tests/search_smoke.cpp src/bitboard.cpp src/position.cpp src/movegen.cpp src/movepick.cpp
        src/zobrist.cpp src/tt.cpp src/search.cpp src/qsearch.cpp src/search_root.cpp src/evaluation.cpp
        src/evaluation/material.cpp src/pst.cpp
    )
    target_include_directories(search_smoke_test PRIVATE src)
    target_compile_options(search_smoke_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME search_smoke COMMAND search_smoke_test)

    add_executable(position_integrity_test
        tests/position_integrity.cpp src/bitboard.cpp src/position.cpp src/movegen.cpp src/zobrist.cpp
    )
    target_include_directories(position_integrity_test PRIVATE src)
    target_compile_options(position_integrity_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME position_integrity COMMAND position_integrity_test)
endif()
