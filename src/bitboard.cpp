#include "bitboard.h"

#include <algorithm>

namespace Zero::Bitboards {

Bitboard KnightAttacks[SQUARE_NB]{};
Bitboard KingAttacks[SQUARE_NB]{};
Bitboard PawnAttacks[COLOR_NB][SQUARE_NB]{};

const Bitboard FileBB[8] = {
    0x0101010101010101ULL, 0x0202020202020202ULL,
    0x0404040404040404ULL, 0x0808080808080808ULL,
    0x1010101010101010ULL, 0x2020202020202020ULL,
    0x4040404040404040ULL, 0x8080808080808080ULL
};

const Bitboard RankBB[8] = {
    0x00000000000000FFULL, 0x000000000000FF00ULL,
    0x0000000000FF0000ULL, 0x00000000FF000000ULL,
    0x000000FF00000000ULL, 0x0000FF0000000000ULL,
    0x00FF000000000000ULL, 0xFF00000000000000ULL
};

namespace {

enum Direction : int { NORTH, SOUTH, EAST, WEST, NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST };

Bitboard RayBB[SQUARE_NB][8]{};

constexpr int knightDelta[8][2] = {
    { 1, 2}, { 2, 1}, { 2,-1}, { 1,-2},
    {-1,-2}, {-2,-1}, {-2, 1}, {-1, 2}
};

constexpr int kingDelta[8][2] = {
    {-1,-1}, {-1,0}, {-1,1}, {0,-1},
    {0,1}, {1,-1}, {1,0}, {1,1}
};

constexpr int dirDelta[8][2] = {
    {0, 1}, {0,-1}, {1, 0}, {-1, 0},
    {1, 1}, {-1, 1}, {1,-1}, {-1,-1}
};

Bitboard leaper_attacks(Square sq, const int (*delta)[2], int count) {
    Bitboard result = 0;
    const int f = file_of(sq), r = rank_of(sq);
    for (int i = 0; i < count; ++i) {
        const int nf = f + delta[i][0];
        const int nr = r + delta[i][1];
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
            result |= square_bb(make_square(nf, nr));
    }
    return result;
}

void init_rays() {
    for (Square sq = 0; sq < SQUARE_NB; ++sq) {
        const int f = file_of(sq), r = rank_of(sq);
        for (int d = 0; d < 8; ++d) {
            Bitboard ray = 0;
            int nf = f + dirDelta[d][0];
            int nr = r + dirDelta[d][1];
            while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                ray |= square_bb(make_square(nf, nr));
                nf += dirDelta[d][0];
                nr += dirDelta[d][1];
            }
            RayBB[sq][d] = ray;
        }
    }
}

inline Bitboard ray_attacks(Square sq, Bitboard occupied, Direction d, bool increasing) {
    const Bitboard ray = RayBB[sq][d];
    const Bitboard blockers = ray & occupied;
    if (!blockers)
        return ray;

    const Square blocker = increasing ? lsb(blockers) : msb(blockers);
    return ray ^ RayBB[blocker][d];
}

}

void init() {
    init_rays();

    for (Square sq = 0; sq < SQUARE_NB; ++sq) {
        KnightAttacks[sq] = leaper_attacks(sq, knightDelta, 8);
        KingAttacks[sq] = leaper_attacks(sq, kingDelta, 8);

        const int f = file_of(sq), r = rank_of(sq);
        Bitboard white = 0, black = 0;
        if (r < 7) {
            if (f > 0) white |= square_bb(make_square(f - 1, r + 1));
            if (f < 7) white |= square_bb(make_square(f + 1, r + 1));
        }
        if (r > 0) {
            if (f > 0) black |= square_bb(make_square(f - 1, r - 1));
            if (f < 7) black |= square_bb(make_square(f + 1, r - 1));
        }
        PawnAttacks[WHITE][sq] = white;
        PawnAttacks[BLACK][sq] = black;
    }
}

Bitboard bishop_attacks(Square sq, Bitboard occupied) {
    return ray_attacks(sq, occupied, NORTH_EAST, true)
         | ray_attacks(sq, occupied, NORTH_WEST, true)
         | ray_attacks(sq, occupied, SOUTH_EAST, false)
         | ray_attacks(sq, occupied, SOUTH_WEST, false);
}

Bitboard rook_attacks(Square sq, Bitboard occupied) {
    return ray_attacks(sq, occupied, NORTH, true)
         | ray_attacks(sq, occupied, SOUTH, false)
         | ray_attacks(sq, occupied, EAST, true)
         | ray_attacks(sq, occupied, WEST, false);
}

Bitboard queen_attacks(Square sq, Bitboard occupied) {
    return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied);
}

}
