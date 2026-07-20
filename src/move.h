#ifndef MOVE_H
#define MOVE_H

#include "piece.h"

struct Move
{
    int fromRow;
    int fromCol;

    int toRow;
    int toCol;

    Piece movedPiece;
    Piece capturedPiece;
    bool isPromotion = false;
    Piece promotionPiece = EMPTY;

    int score = 0;
};

#endif
