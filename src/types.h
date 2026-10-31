#ifndef ZERO_TYPES_H
#define ZERO_TYPES_H

#include <cstdint>

namespace Zero {

enum Color : uint8_t { WHITE, BLACK, COLOR_NB };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : uint8_t {
    NO_PIECE_TYPE,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    PIECE_TYPE_NB
};

using Bitboard = std::uint64_t;
using Square   = std::uint8_t;
using Value    = int;
using Depth    = int;
using Key      = std::uint64_t;

constexpr Square SQ_NONE = 64;
constexpr int SQUARE_NB = 64;
constexpr Value VALUE_DRAW    = 0;
constexpr Value VALUE_MATE    = 100000;
constexpr Value VALUE_INFINITE = 1000000;
constexpr int MAX_PLY = 128;

constexpr Bitboard EMPTY_BB = 0ULL;

inline constexpr Bitboard square_bb(Square sq) {
    return sq < SQUARE_NB ? (Bitboard(1) << sq) : 0ULL;
}

inline constexpr int file_of(Square sq) { return int(sq) & 7; }
inline constexpr int rank_of(Square sq) { return int(sq) >> 3; } // 0..7 = rank 1..8
inline constexpr Square make_square(int file, int rank) {
    return Square(rank * 8 + file);
}

inline int popcount(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(b);
#else
    int count = 0;
    while (b) { b &= b - 1; ++count; }
    return count;
#endif
}

inline Square lsb(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
    return Square(__builtin_ctzll(b));
#else
    Square s = 0;
    while ((b & 1ULL) == 0) { b >>= 1; ++s; }
    return s;
#endif
}

inline Square msb(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
    return Square(63 - __builtin_clzll(b));
#else
    Square s = 63;
    while ((b & (Bitboard(1) << 63)) == 0) { b <<= 1; --s; }
    return s;
#endif
}

inline Square pop_lsb(Bitboard& b) {
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

}

#endif
