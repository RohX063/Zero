#include "search.h"
#include "search_helpers.h"
#include "search_mistake.h"
#include "search_params.h"

#include <algorithm>
#include <array>

#include "evaluation.h"
#include "movegen.h"
#include "movepick.h"
#include "qsearch.h"
#include "position.h"
#include "repetition.h"

namespace Zero::Search {

Worker::Worker(Position& position)
    : position_(position),
      tt_(TranspositionTable::DEFAULT_HASH_MB),
      history_(std::make_unique<HistoryTables>())
{
    rebuildLMRTable();
}

void Worker::rebuildLMRTable()
{
    for (auto& row : lmrTable_)
        row.fill(0);

    for (int depth = 1; depth <= MAX_PLY; ++depth)
        for (int moveCount = 1; moveCount <= int(MAX_MOVES); ++moveCount)
            lmrTable_[depth][moveCount] = evaluateLMRFormula(lmrParams_, depth, moveCount);
}

void Worker::setLMRLogScale(int value)
{
    lmrParams_.logScale = std::clamp(value, 0, 1200);
    rebuildLMRTable();
}

bool Worker::setParam(const std::string& name, int value)
{
    if (name == "LMRLogScale") {
        setLMRLogScale(value);
        return true;
    }
    if (name == "Move Overhead") {
        params_.moveOverheadMs = std::clamp(value, 0, 2000);
        return true;
    }
    for (const ParamDef& def : searchParamTable()) {
        if (name == def.name) {
            params_.*(def.field) = std::clamp(value, def.lo, def.hi);
            return true;
        }
    }
    return false;
}

void Worker::setLMRBase(int value)
{
    lmrParams_.base = std::clamp(value, -4096, 4096);
    rebuildLMRTable();
}

void Worker::setLMRDepthCoeff(int value)
{
    lmrParams_.depthCoeff = std::clamp(value, -2048, 2048);
    rebuildLMRTable();
}

void Worker::setLMRMoveCoeff(int value)
{
    lmrParams_.moveCoeff = std::clamp(value, -2048, 2048);
    rebuildLMRTable();
}

void Worker::setLMRPVAdjust(int value)
{
    lmrParams_.pvAdjust = std::clamp(value, -2048, 2048);
}

void Worker::setLMRCutAdjust(int value)
{
    lmrParams_.cutAdjust = std::clamp(value, -2048, 2048);
}

void Worker::setLMRTTAdjust(int value)
{
    lmrParams_.ttAdjust = std::clamp(value, -4096, 1024);
}

void Worker::setLMRHistoryCoeff(int value)
{
    lmrParams_.historyCoeff = std::clamp(value, -1024, 1024);
}

void Worker::setLMRContinuationCoeff(int value)
{
    lmrParams_.continuationCoeff = std::clamp(value, -1024, 1024);
}

void Worker::setLMRImprovingAdjust(int value)
{
    lmrParams_.improvingAdjust = std::clamp(value, -2048, 1024);
}

void Worker::setLMRTacticalSafetyAdjust(int value)
{
    lmrParams_.tacticalSafetyAdjust = std::clamp(value, -2048, 1024);
}

int Worker::lmrReduction(Depth depth,
                         int moveCount,
                         int historyScore,
                         int continuationHistoryScore,
                         bool improving,
                         bool pvNode,
                         bool cutNode,
                         bool ttMove,
                         bool tacticalSafety,
                         int alphaBetaWindow) const
{
    if (depth < 3 || moveCount < 2)
        return 0;

    const int d = std::clamp(depth, 1, MAX_PLY);
    const int m = std::clamp(moveCount, 1, int(MAX_MOVES));
    int reduction1024 = lmrTable_[d][m];

    // Contextual LMR model. Every coefficient is independently exposed to
    // SPSA, while the logarithmic depth*move interaction remains frozen.
    if (pvNode)
        reduction1024 += lmrParams_.pvAdjust;
    if (cutNode)
        reduction1024 += lmrParams_.cutAdjust;
    if (ttMove)
        reduction1024 += lmrParams_.ttAdjust;
    if (improving)
        reduction1024 += lmrParams_.improvingAdjust;
    if (tacticalSafety)
        reduction1024 += lmrParams_.tacticalSafetyAdjust;

    reduction1024 += std::clamp(historyScore, -32767, 32767)
                   * lmrParams_.historyCoeff / 1024;
    reduction1024 += std::clamp(continuationHistoryScore, -32767, 32767)
                   * lmrParams_.continuationCoeff / 1024;

    // Narrow windows are expected-cut/expected-fail-low probes: reduce a bit
    // less there than inside wide PV windows.
    reduction1024 -= std::clamp(alphaBetaWindow, 1, 512) * 128 / 512;

    return std::max(0, reduction1024);
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

std::uint64_t Worker::elapsedMs() const
{
    if (searchStart_ == std::chrono::steady_clock::time_point{})
        return 0;

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - searchStart_).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

std::uint64_t Worker::nps() const
{
    const std::uint64_t ms = std::max<std::uint64_t>(1, elapsedMs());
    if (nodes_ == 0)
        return 0;
    return (nodes_ * 1000ULL) / ms;
}

Value Worker::search(Position& position, Stack* ss, Depth depth, Value alpha, Value beta, bool cutNode)
{
    visitNode(ss->ply);
    if (shouldStop())
        return VALUE_DRAW;

    // Draws are scored before anything else, including at the horizon, so a
    // repetition one ply from the leaf is never hidden by qsearch.
    if (ss->ply > 0 && Rules::isDrawInSearch(position, ss->ply))
        return VALUE_DRAW;

    ss->inCheck = position.isKingInCheck(position.isWhiteToMove());

    if (depth <= 0)
        return QSearch(*this).run(position, ss, alpha, beta);

    // Mate-distance clamp: no line from here can be better than mating on the
    // next ply or worse than being mated right now.
    if (ss->ply > 0) {
        alpha = std::max(alpha, Value(-VALUE_MATE + ss->ply));
        beta  = std::min(beta,  Value(VALUE_MATE - ss->ply - 1));
        if (alpha >= beta)
            return alpha;
    }

    const bool pvNode = alpha + 1 < beta;

    // The parent publishes the reduction it applied to reach this node in its
    // own stack slot; read it once and clear it so a later full-depth
    // re-search from the parent never inherits a stale value.
    const int priorReduction = ss->ply >= 1 ? (ss - 1)->reduction : 0;
    if (ss->ply >= 1)
        (ss - 1)->reduction = 0;

    const Key key = position.key();
    const Value alphaOriginal = alpha;
    const TTData ttData = tt_.probe(key);
    const Value ttValue = ttData.hit
        ? TranspositionTable::value_from_tt(ttData.value, ss->ply)
        : VALUE_DRAW;

    // TT cutoffs only at non-PV nodes (PV nodes keep their exact window), and
    // never close to the fifty-move horizon where stored scores are unreliable.
    if (!pvNode && ttData.hit && ttData.depth >= depth
        && position.halfmove_clock() < 90) {
        if (ttData.bound == BOUND_EXACT)
            return ttValue;
        if (ttData.bound == BOUND_LOWER && ttValue >= beta)
            return ttValue;
        if (ttData.bound == BOUND_UPPER && ttValue <= alpha)
            return ttValue;
    }

    ss->evalValid = false;
    if (!ss->inCheck && depth >= 3) {
        ss->staticEval = position.isWhiteToMove()
            ? evaluatePosition(position)
            : -evaluatePosition(position);
        ss->evalValid = true;
    }

    const bool improving = ss->ply >= 2 && ss->evalValid && (ss - 2)->evalValid
        && ss->staticEval > (ss - 2)->staticEval;

    const bool haveParentEval = ss->ply >= 1 && ss->evalValid && (ss - 1)->evalValid;

    // Hindsight depth correction (see search_mistake.h).
    if (haveParentEval && ss->ply >= 2 && depth >= 3) {
        HindsightParams hp;
        hp.recoverReduction = params_.hindsightRecoverReduction;
        hp.trimReduction    = params_.hindsightTrimReduction;
        hp.trimEvalSum      = params_.hindsightTrimEvalSum;
        depth = std::min(Depth(MAX_PLY - 1),
                         hindsightDepthAdjustment(
                             depth, priorReduction, *ss, *(ss - 1), hp));
    }

    const bool opponentWorsened = haveParentEval && opponentWorsening(*ss, *(ss - 1));

    // Null-move pruning at expected cut nodes.
    //  * margin and reduction come from SearchParams (ZERO-tuned)
    //  * a null cutoff is a reduced proof, so it never writes a TT entry
    //  * deep cutoffs are verified with null moves disabled underneath
    if (depth >= std::max(3, params_.nmpMinDepth)
        && !pvNode
        && cutNode
        && !ss->inCheck
        && ss->evalValid
        && ss->canNullMove
        && !ss->isNullMove
        && ss->ply >= ss->nmpMinPly
        && ss->staticEval >= beta + params_.nmpMarginBase
                             - params_.nmpMarginPerDepth * depth
        && beta > -VALUE_MATE + 1000)
    {
        const Color us = position.isWhiteToMove() ? WHITE : BLACK;
        const Bitboard nonPawnMaterial =
            position.pieces(us)
            & ~(position.pieces(PAWN) | position.pieces(KING));

        if (nonPawnMaterial != 0) {
            const int evalGap = std::max(0, ss->staticEval - beta);
            const Depth reduction = std::min(
                depth - 1,
                params_.nmpReductionBase + depth / params_.nmpDepthDivisor
                    + std::min(evalGap / params_.nmpEvalDivisor, params_.nmpEvalCap));
            const Depth nullDepth = depth - 1 - reduction;

            if (nullDepth >= 0) {
                StateInfo nullState;
                position.doNullMove(nullState);

                Stack* child = ss + 1;
                *child = Stack{};
                child->ply = ss->ply + 1;
                child->canNullMove = false;
                child->isNullMove = true;
                child->nmpMinPly = ss->nmpMinPly;

                ++stats_.nullMoveSearches;
                const Value nullScore =
                    -search(position, child, nullDepth, -beta, -beta + 1, false);

                position.undoNullMove();

                if (shouldStop())
                    return VALUE_DRAW;

                if (nullScore >= beta && std::abs(nullScore) < VALUE_MATE - 1000) {
                    ++stats_.nullMoveCutoffs;

                    if (depth >= params_.nmpVerifyDepth && nullDepth > 0
                        && ss->nmpMinPly == 0) {
                        const Stack savedStack = *ss;
                        const int verifyUntil = ss->ply
                            + std::max(1, nullDepth * params_.nmpVerifyPercent / 100);
                        ss->nmpMinPly = verifyUntil;

                        ++stats_.nullMoveVerifications;
                        const Value verify =
                            search(position, ss, nullDepth, beta - 1, beta, false);

                        *ss = savedStack;

                        if (verify >= beta)
                            return nullScore;
                    } else {
                        return nullScore;
                    }
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
                      history_.get(),
                      us,
                      previousPiece,
                      previousTo);

    std::array<TriedMove, MAX_MOVES> tried{};
    std::size_t triedCount = 0;

    Value bestScore = -VALUE_INFINITE;
    Move bestMove = Move::none();
    int moveCount = 0;
    bool firstMove = true;

    // Extensions are rationed relative to the iteration depth so chains of
    // checks cannot blow the tree up.
    const bool extensionsAllowed = ss->ply + 2 < MAX_PLY
        && ss->ply < std::max(8, rootDepth_ * params_.extensionPlyLimitPct / 100);

    // Child node types: the first child of a PV node is a PV node; every other
    // zero-window child of a PV node is expected to fail high (cut node);
    // below non-PV nodes the type simply alternates.
    const bool nonFirstChildCut = pvNode ? true : !cutNode;

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
        int continuationHistory = 0;
        if (quiet) {
            quietHistory = history_->quietScore(
                us, attacker, move.from_sq(), move.to_sq(), previousPiece, previousTo);

            if (previousPiece != EMPTY && previousTo != SQ_NONE) {
                continuationHistory = history_->continuation(
                    previousPiece, previousTo, attacker, move.to_sq());
            }
        }

        StateInfo newState;
        position.doMove(move, newState);

        Stack* child = ss + 1;
        *child = Stack{};
        child->ply = ss->ply + 1;
        child->currentMove = move;
        child->movedPiece = attacker;
        child->canNullMove = true;
        child->nmpMinPly = ss->nmpMinPly;
        child->isNullMove = false;

        const bool givesCheck = position.isKingInCheck(position.isWhiteToMove());

        int extension = 0;
        if (extensionsAllowed) {
            if (givesCheck && params_.checkExtension) {
                extension++;
                ++stats_.checkExtensions;
            }
            if (move.isPromotion() && params_.promotionExtension) {
                extension++;
                ++stats_.promotionExtensions;
            }
        }

        const Depth fullDepth = std::max(0, depth - 1 + extension);

        int reduction = 0;

        // LMR applies to quiet, non-checking late moves.
        if (!firstMove
            && quiet
            && !ss->inCheck
            && !move.isPromotion()
            && !givesCheck)
        {
            const bool isKiller =
                move == killerMoves_.first(ss->ply)
                || move == killerMoves_.second(ss->ply);
            const bool isCounter = move == counterMove;
            const bool tacticalSafety = isKiller || isCounter;

            const int reduction1024 = lmrReduction(
                depth,
                moveCount,
                quietHistory,
                continuationHistory,
                improving,
                pvNode,
                cutNode,
                ttData.hit && move == ttData.move,
                tacticalSafety,
                beta - alpha);

            reduction = std::min(
                std::max(0, fullDepth - 1),
                (reduction1024 + 512) / 1024);

            reduction = punishOpponentWorsening(
                reduction, opponentWorsened, moveCount, pvNode);

            if (reduction > 0)
                ++stats_.lmrReductions;
        }

        Value score;

        if (firstMove) {
            score = -search(position, child, fullDepth, -beta, -alpha, pvNode ? false : !cutNode);
            firstMove = false;
        } else {
            const Depth probeDepth = std::max(0, fullDepth - reduction);

            // Publish the reduction for the child's hindsight correction.
            ss->reduction = reduction;
            score = -search(position, child, probeDepth, -alpha - 1, -alpha,
                            reduction > 0 ? true : nonFirstChildCut);
            ss->reduction = 0;

            // A reduced probe that beats alpha is verified at full depth with
            // the same zero window; only PV nodes then widen the window.
            if (score > alpha && reduction > 0) {
                ++stats_.lmrResearches;
                score = -search(position, child, fullDepth, -alpha - 1, -alpha,
                                nonFirstChildCut);
            }

            if (pvNode && score > alpha && score < beta) {
                ++stats_.pvsReSearches;
                score = -search(position, child, fullDepth, -beta, -alpha, false);
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

    const bool cutoff = bestScore >= beta;

    // Learning signal: only when some move actually improved on the window.
    // A node where every move failed low carries no information about which
    // quiet move is good, so it does not reward `bestMove`.
    if (bestScore > alphaOriginal && bestMove.isOk()) {
        const int baseBonus = historyBonus(params_, depth);
        const int winnerBonus = cutoff ? baseBonus : baseBonus / 2;
        const int loserPenalty =
            std::max(4, baseBonus * params_.historyMalusPercent / 100);

        for (std::size_t i = 0; i < triedCount; ++i) {
            const TriedMove& tm = tried[i];
            const int bonus = (tm.move == bestMove) ? winnerBonus : -loserPenalty;

            if (tm.quiet) {
                history_->updateQuiet(us, tm.attacker, tm.move.from_sq(), tm.move.to_sq(),
                                      previousPiece, previousTo, bonus);
            } else if (tm.victim != EMPTY) {
                history_->updateCapture(tm.attacker, tm.victim, tm.move.to_sq(), bonus);
            }
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

    // Counter-move table: only a quiet move that caused a beta cutoff refutes
    // the previous move.
    if (cutoff && previousMove.isOk() && bestMove.isOk()
        && isQuietMove(bestMove, capturedPiece(position, bestMove)))
        counterMoves_.update(previousPiece, previousTo, bestMove);

    return bestScore;
}

}
