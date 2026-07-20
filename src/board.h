#include "move.h"
#ifndef BOARD_H
#define BOARD_H
#include <string>

class Board
{
public:

    Board();

    void initialize();

    void printBoard() const;

    void makeMove(int fromRow, int fromCol,
              int toRow, int toCol);
    
    void makeMove(Move &move);
    void undoMove(const Move &move);
    Piece getPiece(int row, int col) const;

    bool isSquareAttacked(int row,
                      int col,
                      bool byWhite) const;

    bool isKingInCheck(bool whiteKing) const;

    void clearBoard();
    void setPiece(int row, int col, Piece piece);

    bool isWhiteToMove() const;

    void setSideToMove(bool white);

    void switchSide();

    bool loadFEN(const std::string& fen);

    void makeNullMove();
    void undoNullMove();

private:

    int board[8][8];

    bool whiteToMove;
};

#endif