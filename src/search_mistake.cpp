#include "search_mistake.h"

#include <algorithm>

namespace Zero::Search {

namespace {
constexpr int EARLY_PUNISH_MOVE_LIMIT = 6;
}

bool opponentWorsening(const Stack& current, const Stack& previous)
{
    // Both evaluations are kept in the node's own negamax perspective.
    return current.staticEval > -previous.staticEval;
}

Depth hindsightDepthAdjustment(Depth depth,
                               int priorReduction,
                               const Stack& current,
                               const Stack& previous,
                               const HindsightParams& params)
{
    int adjusted = depth;
    const bool worsened = opponentWorsening(current, previous);

    if (priorReduction >= params.recoverReduction && !worsened)
        ++adjusted;

    if (priorReduction >= params.trimReduction
        && adjusted >= 2
        && current.staticEval + previous.staticEval > params.trimEvalSum)
        --adjusted;

    return std::max<Depth>(1, adjusted);
}

Depth hindsightDepthAdjustment(Depth depth,
                               int priorReduction,
                               const Stack& current,
                               const Stack& previous)
{
    return hindsightDepthAdjustment(depth, priorReduction, current, previous,
                                    HindsightParams{});
}

int punishOpponentWorsening(int reduction,
                            bool opponentWorsened,
                            int moveCount,
                            bool pvNode)
{
    if (!opponentWorsened || pvNode || reduction <= 0)
        return std::max(0, reduction);

    if (moveCount <= EARLY_PUNISH_MOVE_LIMIT)
        return std::max(0, reduction - 1);

    return reduction;
}

} // namespace Zero::Search
