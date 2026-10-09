#ifndef ZERO_SEARCH_MISTAKE_H
#define ZERO_SEARCH_MISTAKE_H

#include "search_stack.h"
#include "types.h"

namespace Zero::Search {

// Search-context signal: the opponent's last move "worsened" their own position
// when our static evaluation now exceeds the negation of the evaluation they had
// one ply earlier. Pure function so it can be tested without a Position/Worker.
bool opponentWorsening(const Stack& current, const Stack& previous);

// Thresholds are SPSA-tunable (see SearchParams); defaults are ZERO-native.
struct HindsightParams {
    int recoverReduction = 3;   // prior reduction needed to give one ply back
    int trimReduction    = 2;   // prior reduction needed before a ply can be taken away
    int trimEvalSum      = 150; // current+previous static eval above which a ply is taken
};

// Hindsight correction for a node that was reached through a reduced search:
// if the reduction was large and the position did not get worse for the
// opponent, the reduction was probably too aggressive (+1 ply); if the node
// looks clearly winning for the side to move after a reduction, trim a ply.
Depth hindsightDepthAdjustment(Depth depth,
                               int priorReduction,
                               const Stack& current,
                               const Stack& previous,
                               const HindsightParams& params);

// Convenience overload using the default thresholds.
Depth hindsightDepthAdjustment(Depth depth,
                               int priorReduction,
                               const Stack& current,
                               const Stack& previous);

// Conservative ZERO-native punishment hook: when the opponent's move has
// worsened the static evaluation, reduce the LMR reduction by at most one on
// the first few candidate moves. This spends a little more search where a
// concrete mistake is most likely to be punishable, without creating a new
// search mode or persistent state field.
int punishOpponentWorsening(int reduction,
                            bool opponentWorsened,
                            int moveCount,
                            bool pvNode);

} // namespace Zero::Search

#endif
