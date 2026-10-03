#include "search.h"

#include <algorithm>
#include <array>

#include "evaluation.h"
#include "movegen.h"
#include "movepick.h"
#include "qsearch.h"
#include "position.h"
#include "repetition.h"

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

int lmrReduction(Depth depth, int moveCount, int historyScore, bool inCheck)
{
    constexpr int MIN_DEPTH = 3;
    constexpr int MIN_MOVES = 4;

    if (inCheck || depth < MIN_DEPTH || moveCount < MIN_MOVES)
        return 0;

    int reduction = 1;

    if (depth >= 6)
        ++reduction;
    if (moveCount >= 8)
        ++reduction;

    if (historyScore < 0)
        ++reduction;
    else if (historyScore > 2000 && reduction > 1)
        --reduction;

    return std::min(reduction, depth - 2);
}

// A small positive margin prevents null-move pruning from firing on positions
// that barely meet beta. This is deliberately conservative while ZERO's
// selective search is still being validated for tactical correctness.
constexpr Value NULL_MOVE_MARGIN = 64;

} // namespace

Worker::Worker(Position& position)
    : position_(position),
      tt_(TranspositionTable::DEFAULT_HASH_MB)
{
}

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

    if (Rules::isAutomaticDraw(position))
        return VALUE_DRAW;

    ss->inCheck = position.isKingInCheck(position.isWhiteToMove());

    if (depth <= 0)
        return QSearch(*this).run(position, ss, alpha, beta);

    const bool pvNode = alpha + 1 < beta;

    const Key key = position.key();
    const Value alphaOriginal = alpha;
    const TTData ttData = tt_.probe(key);
    const Value ttValue = ttData.hit
        ? TranspositionTable::value_from_tt(ttData.value, ss->ply)
        : VALUE_DRAW;

    if (ttData.hit && ttData.depth >= depth) {
        if (ttData.bound == BOUND_EXACT)
            return ttValue;
        if (ttData.bound == BOUND_LOWER && ttValue >= beta)
            return ttValue;
        if (ttData.bound == BOUND_UPPER && ttValue <= alpha)
            return ttValue;
    }

    if (!ss->inCheck && depth >= 3) {
        ss->staticEval = position.isWhiteToMove()
            ? evaluatePosition(position)
            : -evaluatePosition(position);
    }

    // Null-move pruning is deliberately disabled at PV nodes. A full-window
    // node is where ZERO must prove the principal variation rather than infer
    // a cutoff from the right to move. The additional margin also reduces
    // tactical false positives while the pruning subsystem is being validated.
    if (depth >= 4
        && !pvNode
        && ss->canNullMove
        && !ss->inCheck
        && ss->staticEval >= beta + NULL_MOVE_MARGIN)
    {
        const Color us = position.isWhiteToMove() ? WHITE : BLACK;
        const Bitboard nonPawnMaterial =
            position.pieces(us)
            & ~position.pieces(PAWN)
            & ~position.pieces(KING);

        if (nonPawnMaterial != 0) {
            const Depth reduction = 2 + depth / 4;
            const Depth nullDepth = depth - 1 - reduction;

            if (nullDepth >= 0) {
                StateInfo nullState;
                position.doNullMove(nullState);

                Stack* child = ss + 1;
                *child = Stack{};
                child->ply = ss->ply + 1;
                child->canNullMove = false;

                ++stats_.nullMoveSearches;
                const Value nullScore =
                    -search(position, child, nullDepth, -beta, -beta + 1);

                position.undoNullMove();

                if (shouldStop())
                    return VALUE_DRAW;

                if (nullScore >= beta) {
                    ++stats_.nullMoveCutoffs;
                    tt_.store(key,
                              TranspositionTable::value_to_tt(nullScore, ss->ply),
                              BOUND_LOWER,
                              depth,
                              Move::none());
                    return nullScore;
                }
            }
        }
    }

    FixedMoveList moves;
    generateLegalMoves(position, position.isWhiteToMove(), moves);
    if (moves.empty()) {
        const Value terminal = ss->inCheck
            ? -VALUE_MATE + ss->ply
            : VALUE_DRAW;

        tt_.store(key,
                  TranspositionTable::value_to_tt(terminal, ss->ply),
                  BOUND_EXACT,
                  depth,
                  Move::none());
        return terminal;
    }

    const Color us = position.isWhiteToMove() ? WHITE : BLACK;
    const Move previousMove = ss->ply > 0 ? ss->currentMove : Move::none();
    const Piece previousPiece = ss->ply > 0 ? ss->movedPiece : EMPTY;
    const Square previousTo = previousMove.isOk()
        ? previousMove.to_sq()
        : SQ_NONE;

    const Move counterMove = previousMove.isOk()
        ? counterMoves_.probe(previousPiece, previousTo)
        : Move::none();

    MovePicker picker(position,
                      moves,
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
    bool firstMove = true;

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

        int quietHistory = 0;
        if (quiet) {
            quietHistory = history_.quietScore(
                us,
                attacker,
                move.from_sq(),
                move.to_sq(),
                previousPiece,
                previousTo);
        }

        StateInfo newState;
        position.doMove(move, newState);

        Stack* child = ss + 1;
        *child = Stack{};
        child->ply = ss->ply + 1;
        child->currentMove = move;
        child->movedPiece = attacker;
        child->canNullMove = true;

        int extension = 0;
        bool givesCheck = false;

        if (ss->ply + 2 < MAX_PLY) {
            givesCheck = position.isKingInCheck(position.isWhiteToMove());

            if (givesCheck) {
                extension++;
                ++stats_.checkExtensions;
            }

            if (move.isPromotion()) {
                extension++;
                ++stats_.promotionExtensions;
            }
        }

        const Depth fullDepth = std::max(0, depth - 1 + extension);

        int reduction = 0;

        // Critical tactical-stability rule:
        // checking moves are never LMR candidates. If ZERO has just created a
        // check, that move must get the full tactical horizon immediately;
        // otherwise a reduced probe can incorrectly fail low and suppress a
        // forcing line before the full-window verification ever happens.
        if (!firstMove
            && quiet
            && !move.isPromotion()
            && !givesCheck)
        {
            reduction = lmrReduction(depth,
                                     moveCount,
                                     quietHistory,
                                     ss->inCheck);

            if (reduction > 0)
                ++stats_.lmrReductions;
        }

        Value score;

        if (firstMove) {
            score = -search(position,
                            child,
                            fullDepth,
                            -beta,
                            -alpha);
            firstMove = false;
        } else {
            const Depth probeDepth = std::max(0, fullDepth - reduction);

            score = -search(position,
                            child,
                            probeDepth,
                            -alpha - 1,
                            -alpha);

            if (score > alpha) {
                ++stats_.pvsReSearches;
                score = -search(position,
                                child,
                                fullDepth,
                                -beta,
                                -alpha);
            }
        }

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
    const int winnerBonus = cutoff
        ? std::min(512, baseBonus * 2)
        : baseBonus;
    const int loserPenalty = std::max(8, baseBonus / 2);

    for (std::size_t i = 0; i < triedCount; ++i) {
        const TriedMove& tm = tried[i];
        const int bonus = (tm.move == bestMove)
            ? winnerBonus
            : -loserPenalty;

        if (tm.quiet) {
            history_.updateQuiet(us,
                                 tm.attacker,
                                 tm.move.from_sq(),
                                 tm.move.to_sq(),
                                 previousPiece,
                                 previousTo,
                                 bonus);
        } else if (tm.victim != EMPTY) {
            history_.updateCapture(tm.attacker,
                                   tm.victim,
                                   tm.move.to_sq(),
                                   bonus);
        }
    }

    Bound bound = BOUND_EXACT;
    if (bestScore <= alphaOriginal)
        bound = BOUND_UPPER;
    else if (bestScore >= beta)
        bound = BOUND_LOWER;

    tt_.store(key,
              TranspositionTable::value_to_tt(bestScore, ss->ply),
              bound,
              depth,
              bestMove,
              pvNode);

    if (previousMove.isOk() && bestMove.isOk())
        counterMoves_.update(previousPiece, previousTo, bestMove);

    return bestScore;
}

}
