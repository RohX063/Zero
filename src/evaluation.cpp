#include "evaluation.h"
#include "pst.h"
#include <iostream>
int evaluatePosition(const Board& board)
{
    int score = 0;

    for(int row = 0; row < 8; row++)
    {
        for(int col = 0; col < 8; col++)
        {
            Piece piece = board.getPiece(row, col);

            switch(piece)
            {
                //----------------------------
                // White Pieces
                //----------------------------

                case WHITE_PAWN:
                    score += 100;
                    score += pawnTable[7 - row][col];
                    break;

                case WHITE_KNIGHT:
                    score += 320;
                    score += knightTable[row][col];
                    break;

                case WHITE_BISHOP:
                    score += 330;
                    score += bishopTable[row][col];
                    break;

                case WHITE_ROOK:
                    score += 500;
                    score += rookTable[row][col];
                    break;

                case WHITE_QUEEN:
                    score += 900;
                    score += queenTable[row][col];
                    break;

                case WHITE_KING:
                    score += kingTable[row][col];
                    break;
                    

                //----------------------------
                // Black Pieces
                //----------------------------

                case BLACK_PAWN:
                    score -= 100;
                    score -= pawnTable[row][col];;
                    break;

                case BLACK_KNIGHT:
                    score -= 320;
                    score -= knightTable[7-row][col];
                    break;

                case BLACK_BISHOP:
                    score -= 330;
                    score -= bishopTable[7-row][col];
                    break;

                case BLACK_ROOK:
                    score -= 500;
                    score -= rookTable[7-row][col];
                    break;

                case BLACK_QUEEN:
                    score -= 900;
                    score -= queenTable[7-row][col];
                    break;

                case BLACK_KING:
                    score -= kingTable[7-row][col];
                    break;
                

                default:
                    break;
            }
        }
    }

    return score;
}
int pieceValue(Piece piece)
{
    switch(piece)
    {
        case WHITE_PAWN:
        case BLACK_PAWN:
            return 100;

        case WHITE_KNIGHT:
        case BLACK_KNIGHT:
            return 320;

        case WHITE_BISHOP:
        case BLACK_BISHOP:
            return 330;

        case WHITE_ROOK:
        case BLACK_ROOK:
            return 500;

        case WHITE_QUEEN:
        case BLACK_QUEEN:
            return 900;

        case WHITE_KING:
        case BLACK_KING:
            return 20000;

        default:
            return 0;
    }
}