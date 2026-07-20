#include "moveordering.h"
#include "move.h"
#include "evaluation.h"
#include "killer.h"
#include "history.h"

int scoreMove(
    const Board& board,
    const Move& move,
    int ply
)
{
    Piece attacker =
        board.getPiece(
            move.fromRow,
            move.fromCol
        );

    Piece victim =
        board.getPiece(
            move.toRow,
            move.toCol
        );

    //--------------------------------------------------
    // Captures (MVV-LVA)
    //--------------------------------------------------

    if(victim != EMPTY)
    {
        return
            100000
            +
            pieceValue(victim) * 10
            -
            pieceValue(attacker);
    }

    //--------------------------------------------------
    // Killer Moves
    //--------------------------------------------------

    if(isKillerMove(move, ply))
    {
        return 90000;
    }

    //--------------------------------------------------
    // History Heuristic
    //--------------------------------------------------

    return getHistoryScore(move);
}