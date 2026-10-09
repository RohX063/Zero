cmake_minimum_required(VERSION 3.16)
project(zero_engine VERSION 0.2 LANGUAGES CXX)

# Default to an optimized build; a bare `cmake ..` used to produce -O0.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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
    src/countermove.cpp
    src/history.cpp
    src/killer.cpp
    src/zobrist.cpp
    src/tt.cpp
    src/search.cpp
    src/search_helpers.cpp
    src/search_time.cpp
    src/search_mistake.cpp
    src/qsearch.cpp
    src/search_root.cpp
    src/thread.cpp
    src/repetition.cpp
    src/evaluation.cpp
    src/evaluation/material.cpp
    src/evaluation/pawns.cpp
    src/evaluation/mobility.cpp
    src/evaluation/king.cpp
    src/evaluation/threats.cpp
    src/pst.cpp
)

# Cross-platform warning policy: MSVC uses /W4, GCC/Clang use -Wall/-Wextra/-Wpedantic.
function(zero_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_executable(zero ${ZERO_SOURCES})
target_include_directories(zero PRIVATE src)
zero_enable_warnings(zero)

include(CTest)
if(BUILD_TESTING)
    add_executable(perft_test tests/perft.cpp src/bitboard.cpp src/position.cpp src/movegen.cpp src/zobrist.cpp)
    target_include_directories(perft_test PRIVATE src)
    zero_enable_warnings(perft_test)
    add_test(NAME perft COMMAND perft_test)

    add_executable(search_smoke_test
        tests/search_smoke.cpp
        src/bitboard.cpp src/position.cpp src/movegen.cpp src/movepick.cpp
        src/countermove.cpp src/history.cpp src/killer.cpp
        src/zobrist.cpp src/tt.cpp src/search.cpp src/search_helpers.cpp src/search_mistake.cpp src/qsearch.cpp src/search_root.cpp src/search_time.cpp
        src/thread.cpp src/repetition.cpp src/evaluation.cpp
        src/evaluation/material.cpp src/evaluation/pawns.cpp src/evaluation/mobility.cpp
        src/evaluation/king.cpp src/evaluation/threats.cpp src/pst.cpp
    )
    target_include_directories(search_smoke_test PRIVATE src)
    zero_enable_warnings(search_smoke_test)
    add_test(NAME search_smoke COMMAND search_smoke_test)

    add_executable(repetition_search_test
        tests/repetition_search_test.cpp
        src/bitboard.cpp src/position.cpp src/movegen.cpp src/movepick.cpp
        src/zobrist.cpp src/tt.cpp src/search.cpp src/qsearch.cpp src/search_root.cpp src/search_time.cpp
        src/repetition.cpp src/history.cpp src/countermove.cpp src/killer.cpp
        src/search_helpers.cpp src/search_mistake.cpp
        src/evaluation.cpp src/evaluation/material.cpp src/evaluation/pawns.cpp
        src/evaluation/mobility.cpp src/evaluation/king.cpp src/evaluation/threats.cpp src/pst.cpp
    )
    target_include_directories(repetition_search_test PRIVATE src)
    zero_enable_warnings(repetition_search_test)
    add_test(NAME repetition_search COMMAND repetition_search_test)

    add_executable(position_integrity_test
        tests/position_integrity.cpp src/bitboard.cpp src/position.cpp src/movegen.cpp src/zobrist.cpp
    )
    target_include_directories(position_integrity_test PRIVATE src)
    zero_enable_warnings(position_integrity_test)
    add_test(NAME position_integrity COMMAND position_integrity_test)

    add_executable(search_time_test tests/search_time_test.cpp src/search_time.cpp)
    target_include_directories(search_time_test PRIVATE src)
    target_compile_options(search_time_test PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME search_time COMMAND search_time_test)

    add_executable(mistake_punisher_test
        tests/mistake_punisher.cpp
        src/search_mistake.cpp
    )
    target_include_directories(mistake_punisher_test PRIVATE src)
    zero_enable_warnings(mistake_punisher_test)
    add_test(NAME mistake_punisher COMMAND mistake_punisher_test)
endif()

# Python tuning harness is intentionally runtime-only and does not add Python as
# a build dependency. The target documents the command used by CI/dev scripts.
add_custom_target(tune_lmr_smoke
    COMMAND ${CMAKE_COMMAND} -E echo "Run: python3 tuning/spsa_lmr.py --engine <build/zero> --iterations 2 --games 2 --movetime 50 --dry-run"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
)
