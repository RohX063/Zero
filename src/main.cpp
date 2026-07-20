#include "uci.h"
#include "zobrist.h"
#include "tt.h"

int main()
{
    initializeZobrist();

    TT.clear();

    uciLoop();

    return 0;
}