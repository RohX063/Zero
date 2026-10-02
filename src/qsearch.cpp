#include "qsearch.h"

#include <algorithm>

#include "evaluation.h"
#include "movegen.h"
#include "movepick.h"
#include "position.h"

namespace Zero::Search {

int QMovePicker::scoreCapture(const Position& position, const Move& move) {
    const Piece attacker = position.piece_on(move.from_sq());
    Piece victim = position.piece_on(move.to_sq());
    if (move.isEnPassant())
        victim = make_piece(position.isWhiteToMove() ? BLACK : WHITE, PAWN);

    return 100000 + pieceValue(victim) * 10 - pieceValue(attacker);
}

QMovePicker::QMovePicker(const Position& position, bool whiteToMove) {
    const TacticalMoveList list = generateTacticalMoves(position, whiteToMove);
    count_ = list.count;
    for (std::size_t i = 0; i < count_; ++i) {
        moves_[i].move = list.moves[i];
        moves_[i].score = scoreCapture(position, list.moves[i]);
    }
}

Move QMovePicker::next_move() {
    if (cursor_ >= count_)
        return Move::none();

    std::size_t best = cursor_;
    for (std::size_t i = cursor_ + 1; i < count_; ++i)
        if (moves_[i].score > moves_[best].score)
            best = i;

    std::swap(moves_[cursor_], moves_[best]);
    return moves_[cursor_++].move;
}

Value QSearch::run(Position& position, Stack* ss, Value alpha, Value beta) {
    worker_.visitNode();
    if (worker_.shouldStop())
        return VALUE_DRAW;

    if (ss->ply >= MAX_PLY)
        return position.isWhiteToMove() ? evaluatePosition(position) : -evaluatePosition(position);

    ss->moveCount = 0;
    const bool inCheck = position.isKingInCheck(position.isWhiteToMove());
    ss->inCheck = inCheck;

    if (!inCheck) {
        const Value standPat = position.isWhiteToMove() ? evaluatePosition(position)
                                                        : -evaluatePosition(position);
        ss->staticEval = standPat;
        if (standPat >= beta) return standPat;
        if (standPat > alpha) alpha = standPat;
    }

    if (inCheck) {
        FixedMoveList moves;
        generateLegalMoves(position, position.isWhiteToMove(), moves);
        if (moves.empty())
            return -VALUE_MATE + ss->ply;

        MovePicker picker(position, moves);
        while (true) {
            if (worker_.shouldStop())
                return VALUE_DRAW;

            const Move move = picker.next_move();
            if (!move.isOk()) break;

            ss->currentMove = move;
            ++ss->moveCount;

            StateInfo newState;
            position.doMove(move, newState);

            Stack* child = ss + 1;
            child->ply = ss->ply + 1;
            child->moveCount = 0;
            child->staticEval = 0;
            child->currentMove = Move::none();
            child->inCheck = false;

            const Value score = -run(position, child, -beta, -alpha);
            position.undoMove(move);
            if (worker_.shouldStop())
                return VALUE_DRAW;

            if (score > alpha) {
                alpha = score;
                if (alpha >= beta) break;
            }
        }
        return alpha;
    }

    QMovePicker picker(position, position.isWhiteToMove());
    while (true) {
        if (worker_.shouldStop())
            return VALUE_DRAW;

        const Move move = picker.next_move();
        if (!move.isOk()) break;

        ss->currentMove = move;
        ++ss->moveCount;

        StateInfo newState;
        position.doMove(move, newState);

        // Tactical list is pseudo-legal; validate after make to handle pins.
        if (position.isKingInCheck(!position.isWhiteToMove())) {
            position.undoMove(move);
            continue;
        }

        Stack* child = ss + 1;
        child->ply = ss->ply + 1;
        child->moveCount = 0;
        child->staticEval = 0;
        child->currentMove = Move::none();
        child->inCheck = false;

        const Value score = -run(position, child, -beta, -alpha);
        position.undoMove(move);
        if (worker_.shouldStop())
            return VALUE_DRAW;

        if (score > alpha) {
            alpha = score;
            if (alpha >= beta) break;
        }
    }

    return alpha;
}

}
