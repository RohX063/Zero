#include "evaluation.h"
#include "evaluation/material.h"
#include "evaluation/pawns.h"
#include "evaluation/mobility.h"
#include "evaluation/king.h"
#include "position.h"
#include "pst.h"
#include "piece.h"

int evaluatePosition(const Position& position)
{
    using namespace Zero;
    int score = Evaluation::evaluateMaterial(position);
    score += Evaluation::evaluatePawns(position);
    score += Evaluation::evaluateMobility(position);
    score += Evaluation::evaluateKingSafety(position);

    Bitboard pieces = position.pieces();
    while (pieces) {
        const Square sq = pop_lsb(pieces);
        const Piece piece = position.piece_on(sq);
        const int file = file_of(sq);
        const int whiteRow = 7 - rank_of(sq);
        const int pawnWhiteRow = rank_of(sq);
        const int pawnBlackRow = 7 - rank_of(sq);
        const int blackRow = rank_of(sq);

        switch(piece) {
            case WHITE_PAWN:   score += pawnTable[pawnWhiteRow][file]; break;
            case WHITE_KNIGHT: score += knightTable[whiteRow][file]; break;
            case WHITE_BISHOP: score += bishopTable[whiteRow][file]; break;
            case WHITE_ROOK:   score += rookTable[whiteRow][file]; break;
            case WHITE_QUEEN:  score += queenTable[whiteRow][file]; break;
            case WHITE_KING:   score += kingTable[whiteRow][file]; break;
            case BLACK_PAWN:   score -= pawnTable[pawnBlackRow][file]; break;
            case BLACK_KNIGHT: score -= knightTable[blackRow][file]; break;
            case BLACK_BISHOP: score -= bishopTable[blackRow][file]; break;
            case BLACK_ROOK:   score -= rookTable[blackRow][file]; break;
            case BLACK_QUEEN:  score -= queenTable[blackRow][file]; break;
            case BLACK_KING:   score -= kingTable[blackRow][file]; break;
            default: break;
        }
    }
    return score;
}
