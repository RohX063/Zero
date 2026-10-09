#include "search_time.h"

#include <algorithm>
#include <cstdlib>

namespace Zero::Search {

namespace {
constexpr int DEFAULT_MOVES_TO_GO = 30;
constexpr int BULLET_MOVES_TO_GO = 20;
constexpr int VERY_FAST_CLOCK_MS = 3000;
constexpr int FAST_CLOCK_MS = 15000;
constexpr int STABLE_SCORE_DELTA = 30;
constexpr int UNSTABLE_SCORE_DELTA = 80;
}

TimeBudget computeTimeBudget(const Limits& limits)
{
    if (limits.movetimeMs > 0)
        return {limits.movetimeMs, limits.movetimeMs};

    if (limits.sideTimeMs <= 0)
        return {};

    int moves = limits.movesToGo > 0
        ? limits.movesToGo
        : (limits.sideTimeMs <= VERY_FAST_CLOCK_MS ? BULLET_MOVES_TO_GO
                                                    : DEFAULT_MOVES_TO_GO);

    moves = std::max(1, moves);

    // Safety reserve never drops below the configured GUI/IO overhead.
    const int reserve = std::max(std::max(0, limits.moveOverheadMs),
                                 std::clamp(limits.sideTimeMs / 100, 8, 150));
    const int base = limits.sideTimeMs / moves;
    const int incrementPart = (3 * limits.incrementMs) / 4;

    const int maxByReserve = std::max(1, limits.sideTimeMs - reserve);

    // An increment bigger than the remaining clock must not push the target
    // past what is actually safe to spend.
    const int optimum = std::clamp(base + incrementPart - reserve, 1,
                                   std::max(1, maxByReserve * 4 / 5));

    // Give unstable positions extra room, but keep a hard reserve for future
    // moves. Under extreme clock pressure the multiplier is deliberately more
    // conservative so a single long think cannot lose the game on time.
    const double multiplier = limits.sideTimeMs <= VERY_FAST_CLOCK_MS
        ? 1.50
        : (limits.sideTimeMs <= FAST_CLOCK_MS ? 1.75 : 2.00);

    const int maximum = std::min(
        maxByReserve,
        std::max(optimum, static_cast<int>(optimum * multiplier)));

    return {optimum, std::max(optimum, maximum)};
}

int updateStableIterations(const Move& previousBestMove,
                           const Move& currentBestMove,
                           Value previousScore,
                           Value currentScore,
                           int stableIterations)
{
    if (!previousBestMove.isOk() || !currentBestMove.isOk())
        return 0;

    const bool sameMove = previousBestMove == currentBestMove;
    const int scoreDelta = std::abs(int(currentScore - previousScore));

    if (sameMove && scoreDelta <= STABLE_SCORE_DELTA)
        return std::min(stableIterations + 1, 8);

    return 0;
}

bool shouldContinueAfterIteration(const TimeBudget& budget,
                                  int elapsedMs,
                                  const Move& previousBestMove,
                                  const Move& currentBestMove,
                                  Value previousScore,
                                  Value currentScore,
                                  int stableIterations)
{
    // No clock at all (go depth / go nodes / go infinite): the depth limit in
    // the caller is the only stopping rule, so always keep deepening.
    if (budget.optimumMs <= 0)
        return true;

    if (elapsedMs < budget.optimumMs)
        return true;

    if (elapsedMs >= budget.maximumMs)
        return false;

    const bool bestMoveChanged =
        previousBestMove.isOk() && currentBestMove.isOk()
        && previousBestMove != currentBestMove;

    const int scoreDelta = std::abs(int(currentScore - previousScore));
    const bool scoreUnstable = scoreDelta >= UNSTABLE_SCORE_DELTA;

    // Stable PV + stable score: the next iteration is unlikely to change the
    // decision enough to justify spending the extra clock time.
    if (!bestMoveChanged && !scoreUnstable && stableIterations >= 2)
        return false;

    // Otherwise the position is still moving. Keep searching until the hard
    // budget is reached.
    return true;
}

} // namespace Zero::Search
