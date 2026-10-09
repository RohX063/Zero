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

QMovePicker::QMovePicker(Position& position, bool whiteToMove) {
    const TacticalMoveList captures = generateTacticalMoves(position, whiteToMove);

    count_ = captures.count;
    for (std::size_t i = 0; i < captures.count; ++i) {
        moves_[i].move = captures.moves[i];
        moves_[i].score = scoreCapture(position, captures.moves[i]);
    }

    TacticalMoveList checks;
    // QSearch must see quiet checking moves as well as captures/promotions.
    // Without them, a quiet queen/rook/bishop check at the horizon can remain
    // invisible until the next full-depth iteration.
    generateQuietChecks(position, whiteToMove, checks);

    for (std::size_t i = 0; i < checks.count && count_ < moves_.size(); ++i) {
        moves_[count_].move = checks.moves[i];
        // Quiet checks outrank ordinary quiet geometry inside QSearch but remain
        // below the strongest captures.
        moves_[count_].score = 95000;
        ++count_;
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
    worker_.visitNode(ss->ply);
    if (worker_.shouldStop())
        return VALUE_DRAW;

    if (ss->ply >= MAX_PLY)
        return position.isWhiteToMove() ? evaluatePosition(position)
                                        : -evaluatePosition(position);

    ss->moveCount = 0;
    ss->evalValid = false;
    const bool inCheck = position.isKingInCheck(position.isWhiteToMove());
    ss->inCheck = inCheck;

    if (!inCheck) {
        const Value standPat = position.isWhiteToMove()
            ? evaluatePosition(position)
            : -evaluatePosition(position);

        ss->staticEval = standPat;
        ss->evalValid = true;

        if (standPat >= beta)
            return standPat;

        if (standPat > alpha)
            alpha = standPat;
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
            if (!move.isOk())
                break;

            ss->currentMove = move;
            ++ss->moveCount;

            StateInfo newState;
            position.doMove(move, newState);

            Stack* child = ss + 1;
            child->ply = ss->ply + 1;
            child->moveCount = 0;
            child->staticEval = 0;
            child->evalValid = false;
            child->currentMove = move;
            child->movedPiece = position.piece_on(move.to_sq());
            child->inCheck = false;
            child->canNullMove = false;

            const Value score = -run(position, child, -beta, -alpha);

            position.undoMove(move);

            if (worker_.shouldStop())
                return VALUE_DRAW;

            if (score > alpha) {
                alpha = score;
                if (alpha >= beta)
                    break;
            }
        }

        return alpha;
    }

    QMovePicker picker(position, position.isWhiteToMove());

    while (true) {
        if (worker_.shouldStop())
            return VALUE_DRAW;

        const Move move = picker.next_move();
        if (!move.isOk())
            break;

        ss->currentMove = move;
        ++ss->moveCount;

        StateInfo newState;
        position.doMove(move, newState);

        // The QMovePicker contains pseudo-legal tactical/check moves; validate
        // the side-to-move king after making the move.
        if (position.isKingInCheck(!position.isWhiteToMove())) {
            position.undoMove(move);
            continue;
        }

        Stack* child = ss + 1;
        child->ply = ss->ply + 1;
        child->moveCount = 0;
        child->staticEval = 0;
            child->evalValid = false;
        child->currentMove = move;
        child->movedPiece = position.piece_on(move.to_sq());
        child->inCheck = false;
        child->canNullMove = false;

        const Value score = -run(position, child, -beta, -alpha);

        position.undoMove(move);

        if (worker_.shouldStop())
            return VALUE_DRAW;

        if (score > alpha) {
            alpha = score;
            if (alpha >= beta)
                break;
        }
    }

    return alpha;
}

}
