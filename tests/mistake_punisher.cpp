#include <cassert>

#include "search_mistake.h"

using namespace Zero;
using namespace Zero::Search;

int main()
{
    Stack previous;
    Stack current;

    // Sign convention: current > -previous marks the opponent-worsening context.
    previous.staticEval = 100;
    current.staticEval = -50;
    assert(opponentWorsening(current, previous));

    previous.staticEval = -200;
    current.staticEval = 100;
    assert(!opponentWorsening(current, previous));

    // Opponent worsened: no +1 hindsight recovery.
    previous.staticEval = 100;
    current.staticEval = -50;
    assert(hindsightDepthAdjustment(8, 3, current, previous) == 8);

    // Opponent did not worsen and the prior search was heavily reduced:
    // recover one ply.
    previous.staticEval = -200;
    current.staticEval = 100;
    assert(!opponentWorsening(current, previous));
    assert(hindsightDepthAdjustment(8, 3, current, previous) == 9);

    // Large positive combined evaluation triggers the conservative -1 rule.
    previous.staticEval = 100;
    current.staticEval = 100;
    assert(hindsightDepthAdjustment(8, 2, current, previous) == 7);

    assert(punishOpponentWorsening(3, true, 4, false) == 2);
    assert(punishOpponentWorsening(3, true, 8, false) == 3);
    assert(punishOpponentWorsening(3, false, 4, false) == 3);
    assert(punishOpponentWorsening(3, true, 4, true) == 3);

    return 0;
}
