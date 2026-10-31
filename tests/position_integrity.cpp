#include "position.h"
#include "movegen.h"
#include "piece.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {

bool consistent(const Position& pos) {
    using namespace Zero;
    Bitboard rebuilt = 0;
    std::array<Bitboard, PIECE_NB> byPiece{};
    std::array<Bitboard, COLOR_NB> byColor{};
    std::array<Bitboard, PIECE_TYPE_NB> byType{};

    for (Square sq = 0; sq < SQUARE_NB; ++sq) {
        const Piece pc = pos.piece_on(sq);
        if (pc == EMPTY) continue;
        const Bitboard b = square_bb(sq);
        rebuilt |= b;
        byPiece[pc] |= b;
        byColor[color_of(pc)] |= b;
        byType[type_of(pc)] |= b;
    }

    if (rebuilt != pos.pieces()) return false;
    for (int pc = WHITE_PAWN; pc < PIECE_NB; ++pc)
        if (byPiece[pc] != pos.pieces(static_cast<Piece>(pc))) return false;
    for (int c = 0; c < COLOR_NB; ++c)
        if (byColor[c] != pos.pieces(static_cast<Zero::Color>(c))) return false;
    for (int pt = PAWN; pt < PIECE_TYPE_NB; ++pt)
        if (byType[pt] != pos.pieces(static_cast<Zero::PieceType>(pt))) return false;

    return true;
}

}

int main() {
    Position::init();
    std::mt19937 rng(0x5EED1234u);

    for (int game = 0; game < 100; ++game) {
        Position pos;
        const Position initial = pos;
        std::array<StateInfo, Zero::MAX_PLY + 1> states{};
        std::array<Move, Zero::MAX_PLY> moves{};
        int plies = 0;

        while (plies < Zero::MAX_PLY) {
            if (!consistent(pos)) return 1;
            const auto legal = generateLegalMoves(pos, pos.isWhiteToMove());
            if (legal.empty()) break;
            const Move move = legal[rng() % legal.size()];
            const Position before = pos;
            pos.doMove(move, states[plies]);
            moves[plies] = move;
            ++plies;
            if (!consistent(pos)) return 2;

            pos.undoMove(move);
            if (!pos.equals(before)) return 3;
            if (!consistent(pos)) return 4;
            pos.doMove(move, states[plies - 1]);
        }

        while (plies > 0) {
            pos.undoMove(moves[plies - 1]);
            --plies;
        }
        if (!pos.equals(initial)) return 5;
    }

    std::cout << "random position/make-undo integrity: PASS\n";
    return 0;
}
