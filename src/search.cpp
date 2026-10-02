#include "search.h"

#include "evaluation.h"
#include "movegen.h"
#include "movepick.h"
#include "qsearch.h"
#include "position.h"

namespace Zero::Search {

Worker::Worker(Position& position) : position_(position), tt_(TranspositionTable::DEFAULT_HASH_MB) {}

Move Worker::counterMoveFor(Piece previousPiece, Square previousTo) const
{
    return counterMoves_.probe(previousPiece, previousTo);
}

Move Worker::killerMoveFor(Depth ply, std::size_t slot) const
{
    if (slot == 0) return killerMoves_.first(ply);
    if (slot == 1) return killerMoves_.second(ply);
    return Move::none();
}

Value Worker::search(Position& position, Stack* ss, Depth depth, Value alpha, Value beta)
{
    visitNode();
    if (shouldStop())
        return VALUE_DRAW;

    ss->inCheck = position.isKingInCheck(position.isWhiteToMove());

    if (depth <= 0)
        return QSearch(*this).run(position, ss, alpha, beta);

    const Key key = position.key();
    const Value alphaOriginal = alpha;
    const TTData ttData = tt_.probe(key);
    const Value ttValue = ttData.hit ? TranspositionTable::value_from_tt(ttData.value, ss->ply)
                                     : VALUE_DRAW;

    if (ttData.hit && ttData.depth >= depth) {
        if (ttData.bound == BOUND_EXACT)
            return ttValue;
        if (ttData.bound == BOUND_LOWER && ttValue >= beta)
            return ttValue;
        if (ttData.bound == BOUND_UPPER && ttValue <= alpha)
            return ttValue;
    }

    FixedMoveList moves;
    generateLegalMoves(position, position.isWhiteToMove(), moves);
    if (moves.empty()) {
        const Value terminal = ss->inCheck ? -VALUE_MATE + ss->ply : VALUE_DRAW;
        tt_.store(key, TranspositionTable::value_to_tt(terminal, ss->ply), BOUND_EXACT, depth,
                  Move::none());
        return terminal;
    }

    const Move previousMove = ss->ply > 0 ? (ss - 1)->currentMove : Move::none();
    const Move counterMove = previousMove.isOk()
        ? counterMoves_.probe(position.piece_on(previousMove.to_sq()), previousMove.to_sq())
        : Move::none();

    MovePicker picker(position, moves,
                      ttData.hit ? ttData.move : Move::none(),
                      counterMove,
                      killerMoves_.first(ss->ply),
                      killerMoves_.second(ss->ply));
    Value bestScore = -VALUE_INFINITE;
    Move bestMove = Move::none();
    int moveCount = 0;

    for (;;) {
        if (shouldStop())
            return VALUE_DRAW;

        const Move move = picker.next_move();
        if (!move.isOk()) break;

        ++moveCount;
        ss->moveCount = moveCount;
        ss->currentMove = move;

        StateInfo newState;
        position.doMove(move, newState);
        Stack* child = ss + 1;
        *child = Stack{};
        child->ply = ss->ply + 1;
        const Value score = -search(position, child, depth - 1, -beta, -alpha);
        position.undoMove(move);
        if (shouldStop())
            return VALUE_DRAW;

        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
        if (score > alpha) alpha = score;
        if (alpha >= beta) {
            const bool quiet = position.piece_on(move.to_sq()) == EMPTY
                            && !move.isPromotion()
                            && !move.isEnPassant();
            if (quiet)
                killerMoves_.update(ss->ply, move);
            break;
        }
    }

    Bound bound = BOUND_EXACT;
    if (bestScore <= alphaOriginal)
        bound = BOUND_UPPER;
    else if (bestScore >= beta)
        bound = BOUND_LOWER;

    tt_.store(key, TranspositionTable::value_to_tt(bestScore, ss->ply), bound, depth, bestMove,
              beta - alphaOriginal > 1);

    // Only learn from a completed node. The best move found here is the reply
    // to the move that brought us into this position.
    if (previousMove.isOk() && bestMove.isOk())
        counterMoves_.update(position.piece_on(previousMove.to_sq()), previousMove.to_sq(), bestMove);

    return bestScore;
}

}
