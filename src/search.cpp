#include "search.h"

#include <algorithm>
#include <array>

#include "evaluation.h"
#include "movegen.h"
#include "movepick.h"
#include "qsearch.h"
#include "position.h"

namespace Zero::Search {

namespace {

struct TriedMove {
    Move move{};
    Piece attacker = EMPTY;
    Piece victim = EMPTY;
    bool quiet = false;
};

bool isQuietMove(const Move& move, Piece victim)
{
    return victim == EMPTY && !move.isPromotion() && !move.isEnPassant();
}

Piece capturedPiece(const Position& position, const Move& move)
{
    if (move.isEnPassant())
        return make_piece(position.isWhiteToMove() ? BLACK : WHITE, PAWN);

    return position.piece_on(move.to_sq());
}

int historyBonus(Depth depth)
{
    const int d = std::max(1, depth);
    return std::clamp(16 + 8 * d, 16, 512);
}

} // namespace

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

    const Color us = position.isWhiteToMove() ? WHITE : BLACK;
    const Move previousMove = ss->ply > 0 ? ss->currentMove : Move::none();
    const Piece previousPiece = ss->ply > 0 ? ss->movedPiece : EMPTY;
    const Square previousTo = previousMove.isOk() ? previousMove.to_sq() : SQ_NONE;

    const Move counterMove = previousMove.isOk()
        ? counterMoves_.probe(previousPiece, previousTo)
        : Move::none();

    MovePicker picker(position, moves,
                      ttData.hit ? ttData.move : Move::none(),
                      counterMove,
                      killerMoves_.first(ss->ply),
                      killerMoves_.second(ss->ply),
                      &history_,
                      us,
                      previousPiece,
                      previousTo);

    std::array<TriedMove, MAX_MOVES> tried{};
    std::size_t triedCount = 0;

    Value bestScore = -VALUE_INFINITE;
    Move bestMove = Move::none();
    int moveCount = 0;

    for (;;) {
        if (shouldStop())
            return VALUE_DRAW;

        const Move move = picker.next_move();
        if (!move.isOk())
            break;

        ++moveCount;
        ss->moveCount = moveCount;
        ss->currentMove = move;

        const Piece attacker = position.piece_on(move.from_sq());
        const Piece victim = capturedPiece(position, move);
        const bool quiet = isQuietMove(move, victim);

        if (triedCount < tried.size())
            tried[triedCount++] = {move, attacker, victim, quiet};

        StateInfo newState;
        position.doMove(move, newState);

        Stack* child = ss + 1;
        *child = Stack{};
        child->ply = ss->ply + 1;
        child->currentMove = move;
        child->movedPiece = attacker;

        const Value score = -search(position, child, depth - 1, -beta, -alpha);
        position.undoMove(move);

        if (shouldStop())
            return VALUE_DRAW;

        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
        if (score > alpha)
            alpha = score;

        if (alpha >= beta) {
            if (quiet)
                killerMoves_.update(ss->ply, move);
            break;
        }
    }

    const bool cutoff = alpha >= beta;
    const int baseBonus = historyBonus(depth);
    const int winnerBonus = cutoff ? std::min(512, baseBonus * 2) : baseBonus;
    const int loserPenalty = std::max(8, baseBonus / 2);

    // A completed node teaches the move-ordering subsystem from the result of
    // the node. The successful move is reinforced; alternatives that were
    // actually searched are discounted. Continuation history receives the
    // same contextual signal for the previous-move -> current-move sequence.
    for (std::size_t i = 0; i < triedCount; ++i) {
        const TriedMove& tm = tried[i];

        if (tm.quiet) {
            const int bonus = (tm.move == bestMove)
                ? winnerBonus
                : -loserPenalty;

            history_.updateQuiet(us,
                                 tm.attacker,
                                 tm.move.from_sq(),
                                 tm.move.to_sq(),
                                 previousPiece,
                                 previousTo,
                                 bonus);
        } else {
            const int bonus = (tm.move == bestMove)
                ? winnerBonus
                : -loserPenalty;

            if (tm.victim != EMPTY)
                history_.updateCapture(tm.attacker, tm.victim, tm.move.to_sq(), bonus);
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
        counterMoves_.update(previousPiece, previousTo, bestMove);

    return bestScore;
}

}
