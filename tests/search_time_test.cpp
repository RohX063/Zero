#include <cstdlib>
#include "search_time.h"

using namespace Zero::Search;

namespace {
void require(bool condition)
{
    if (!condition)
        std::abort();
}
}

int main()
{
    Limits l;
    l.sideTimeMs = 60000;
    const TimeBudget normal = computeTimeBudget(l);
    require(normal.optimumMs > 0);
    require(normal.maximumMs >= normal.optimumMs);

    l.sideTimeMs = 1000;
    const TimeBudget bullet = computeTimeBudget(l);
    require(bullet.optimumMs > 0);
    require(bullet.maximumMs >= bullet.optimumMs);
    require(bullet.maximumMs <= 1000);

    // Regression: an increment larger than the whole remaining clock must not
    // produce a target beyond what is safe to spend.
    Limits tight;
    tight.sideTimeMs = 500;
    tight.incrementMs = 3000;
    const TimeBudget capped = computeTimeBudget(tight);
    require(capped.optimumMs > 0);
    require(capped.maximumMs < 500);
    require(capped.optimumMs <= capped.maximumMs);

    // Regression: configured GUI overhead is respected as a reserve.
    Limits overhead;
    overhead.sideTimeMs = 2000;
    overhead.moveOverheadMs = 400;
    require(computeTimeBudget(overhead).maximumMs <= 1600);

    // Regression: with no clock (go depth / go infinite) deepening never stops
    // on its own; the depth limit is the only rule.
    require(shouldContinueAfterIteration(TimeBudget{}, 5000, Move::none(), Move::none(), 0, 0, 0));

    Move a{};
    a.from = Zero::make_square(4, 1);
    a.to   = Zero::make_square(4, 3);
    Move b = a;
    Move c{};
    c.from = Zero::make_square(6, 0);
    c.to   = Zero::make_square(5, 2);

    require(updateStableIterations(Move::none(), a, 0, 0, 0) == 0);
    require(updateStableIterations(a, b, 100, 110, 0) == 1);
    require(updateStableIterations(a, c, 100, 110, 1) == 0);

    const TimeBudget budget{100, 200};
    require(!shouldContinueAfterIteration(budget, 120, a, a, 100, 115, 2));
    require(shouldContinueAfterIteration(budget, 120, a, c, 100, 110, 0));
    require(!shouldContinueAfterIteration(budget, 210, a, c, 100, 110, 0));

    return 0;
}
