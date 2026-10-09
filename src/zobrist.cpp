#include "zobrist.h"

namespace Zero::Zobrist {

Bitboard psq[PIECE_NB][SQUARE_NB]{};
Key side = 0;
Key castling[16]{};
Key enPassantFile[8]{};

namespace {

constexpr Key kSeed = 0x9E3779B97F4A7C15ULL;

constexpr Key mix(Key x) {
    x += kSeed;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

}

void init() {
    Key state = kSeed;
    for (int pc = 0; pc < PIECE_NB; ++pc)
        for (int sq = 0; sq < SQUARE_NB; ++sq) {
            state = mix(state);
            psq[pc][sq] = state;
        }

    state = mix(state);
    side = state;

    for (int rights = 0; rights < 16; ++rights) {
        state = mix(state);
        castling[rights] = state;
    }

    for (int file = 0; file < 8; ++file) {
        state = mix(state);
        enPassantFile[file] = state;
    }
}

} // namespace Zero::Zobrist
