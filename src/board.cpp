#include <iostream>
#include "board.h"
#include "piece.h"
#include <sstream>
#include <cctype>

using namespace std;

//------------------------------------------------------------
// Constructor
//------------------------------------------------------------

Board::Board()
{
    initialize();
    whiteToMove = true;
}

void Board::clearBoard()
{
    for(int row = 0; row < 8; row++)
    {
        for(int col = 0; col < 8; col++)
        {
            board[row][col] = EMPTY;
        }
    }
}

void Board::setPiece(int row, int col, Piece piece)
{
    board[row][col] = piece;
}

bool Board::loadFEN(const std::string& fen)
{
    clearBoard();

    std::stringstream ss(fen);

    std::string boardPart;
    std::string sidePart;

    ss >> boardPart >> sidePart;

    int row = 0;
    int col = 0;

    for(char ch : boardPart)
    {
        if(ch == '/')
        {
            row++;
            col = 0;
            continue;
        }

        if(std::isdigit(ch))
        {
            col += ch - '0';
            continue;
        }

        Piece piece = EMPTY;

        switch(ch)
        {
            case 'P': piece = WHITE_PAWN;   break;
            case 'N': piece = WHITE_KNIGHT; break;
            case 'B': piece = WHITE_BISHOP; break;
            case 'R': piece = WHITE_ROOK;   break;
            case 'Q': piece = WHITE_QUEEN;  break;
            case 'K': piece = WHITE_KING;   break;

            case 'p': piece = BLACK_PAWN;   break;
            case 'n': piece = BLACK_KNIGHT; break;
            case 'b': piece = BLACK_BISHOP; break;
            case 'r': piece = BLACK_ROOK;   break;
            case 'q': piece = BLACK_QUEEN;  break;
            case 'k': piece = BLACK_KING;   break;

            default:
                return false;
        }

        board[row][col] = piece;
        col++;
    }

    if(sidePart == "w")
        whiteToMove = true;
    else if(sidePart == "b")
        whiteToMove = false;
    else
        return false;

    return true;
}

//------------------------------------------------------------
// Initialize Chess Board
//------------------------------------------------------------

void Board::initialize()
{
    // Clear Board
    for(int row = 0; row < 8; row++)
    {
        for(int col = 0; col < 8; col++)
        {
            board[row][col] = EMPTY;
        }
    }

    // Black Major Pieces
    board[0][0] = BLACK_ROOK;
    board[0][1] = BLACK_KNIGHT;
    board[0][2] = BLACK_BISHOP;
    board[0][3] = BLACK_QUEEN;
    board[0][4] = BLACK_KING;
    board[0][5] = BLACK_BISHOP;
    board[0][6] = BLACK_KNIGHT;
    board[0][7] = BLACK_ROOK;

    // Black Pawns
    for(int col = 0; col < 8; col++)
        board[1][col] = BLACK_PAWN;

    // White Pawns
    for(int col = 0; col < 8; col++)
        board[6][col] = WHITE_PAWN;

    // White Major Pieces
    board[7][0] = WHITE_ROOK;
    board[7][1] = WHITE_KNIGHT;
    board[7][2] = WHITE_BISHOP;
    board[7][3] = WHITE_QUEEN;
    board[7][4] = WHITE_KING;
    board[7][5] = WHITE_BISHOP;
    board[7][6] = WHITE_KNIGHT;
    board[7][7] = WHITE_ROOK;
}

//------------------------------------------------------------
// Convert Piece -> Character
//------------------------------------------------------------

char pieceToChar(Piece piece)
{
    switch(piece)
    {
        case EMPTY:          return '.';

        case WHITE_PAWN:     return 'P';
        case WHITE_KNIGHT:   return 'N';
        case WHITE_BISHOP:   return 'B';
        case WHITE_ROOK:     return 'R';
        case WHITE_QUEEN:    return 'Q';
        case WHITE_KING:     return 'K';

        case BLACK_PAWN:     return 'p';
        case BLACK_KNIGHT:   return 'n';
        case BLACK_BISHOP:   return 'b';
        case BLACK_ROOK:     return 'r';
        case BLACK_QUEEN:    return 'q';
        case BLACK_KING:     return 'k';

        default:             return '?';
    }
}

//------------------------------------------------------------
// Print Board
//------------------------------------------------------------

void Board::printBoard() const
{
    cout << "\n";
    cout << "    a b c d e f g h\n";
    cout << "   -----------------\n";

    for(int row = 0; row < 8; row++)
    {
        cout << 8 - row << " | ";

        for(int col = 0; col < 8; col++)
        {
            cout << pieceToChar((Piece)board[row][col]) << " ";
        }

        cout << "| " << 8 - row << endl;
    }

    cout << "   -----------------\n";
    cout << "    a b c d e f g h\n";
}
void Board::makeMove(Move &move)
{
    move.movedPiece = (Piece)board[move.fromRow][move.fromCol];
    move.capturedPiece = (Piece)board[move.toRow][move.toCol];

    board[move.toRow][move.toCol] =
        board[move.fromRow][move.fromCol];

    board[move.fromRow][move.fromCol] = EMPTY;
    switchSide();
}
void Board::undoMove(const Move &move)
{
    board[move.fromRow][move.fromCol] =
        move.movedPiece;

    board[move.toRow][move.toCol] =
        move.capturedPiece;
        switchSide();
}
Piece Board::getPiece(int row, int col) const
{
    return (Piece)board[row][col];
}
bool Board::isWhiteToMove() const
{
    return whiteToMove;
}

void Board::setSideToMove(bool white)
{
    whiteToMove = white;
}

void Board::switchSide()
{
    whiteToMove = !whiteToMove;
}

const int knightOffsets[8][2] =
{
    {-2,-1},
    {-2, 1},

    {-1,-2},
    {-1, 2},

    { 1,-2},
    { 1, 2},

    { 2,-1},
    { 2, 1}
};

const int bishopDirections[4][2] =
{
    {-1,-1},
    {-1, 1},
    { 1,-1},
    { 1, 1}
};

const int rookDirections[4][2] =
{
    {-1,0},
    { 1,0},
    {0,-1},
    {0, 1}
};

const int kingDirections[8][2] =
{
    {-1,-1},
    {-1, 0},
    {-1, 1},

    { 0,-1},
    { 0, 1},

    { 1,-1},
    { 1, 0},
    { 1, 1}
};

bool Board::isSquareAttacked(int row,
                             int col,
                             bool byWhite) const
{
    //--------------------------------------------------
    // White Pawn Attacks
    //--------------------------------------------------

    if(byWhite)
    {
        if(row + 1 < 8)
        {
            if(col - 1 >= 0 &&
               getPiece(row + 1, col - 1) == WHITE_PAWN)
                return true;

            if(col + 1 < 8 &&
               getPiece(row + 1, col + 1) == WHITE_PAWN)
                return true;
        }
    }

    //--------------------------------------------------
    // Black Pawn Attacks
    //--------------------------------------------------

    else
    {
        if(row - 1 >= 0)
        {
            if(col - 1 >= 0 &&
               getPiece(row - 1, col - 1) == BLACK_PAWN)
                return true;

            if(col + 1 < 8 &&
               getPiece(row - 1, col + 1) == BLACK_PAWN)
                return true;
        }
    }

    //--------------------------------------------------
    // Knight Attacks
    //--------------------------------------------------

    for(int i = 0; i < 8; i++)
    {
        int r = row + knightOffsets[i][0];
        int c = col + knightOffsets[i][1];

     if(r < 0 || r >= 8 ||
        c < 0 || c >= 8)
         continue;

         Piece piece = getPiece(r, c);

     if(byWhite && piece == WHITE_KNIGHT)
        return true;

     if(!byWhite && piece == BLACK_KNIGHT)
        return true;
    }
    //--------------------------------------------------
    // Bishop Attacks
    //--------------------------------------------------

    for(int dir = 0; dir < 4; dir++)
    {
        int r = row + bishopDirections[dir][0];
        int c = col + bishopDirections[dir][1];

       while(r >= 0 && r < 8 &&
             c >= 0 && c < 8)
    {
            Piece piece = getPiece(r,c);

          if(piece != EMPTY)
         {
             if(byWhite &&
              (piece == WHITE_BISHOP ||
               piece == WHITE_QUEEN))
                return true;

             if(!byWhite &&
              (piece == BLACK_BISHOP ||
               piece == BLACK_QUEEN))
                return true;

             break;
        }

             r += bishopDirections[dir][0];
             c += bishopDirections[dir][1];
    }
    }
    //--------------------------------------------------
    // Rook Attacks
    //--------------------------------------------------

    for(int dir = 0; dir < 4; dir++)
    {
        int r = row + rookDirections[dir][0];
        int c = col + rookDirections[dir][1];

       while(r >= 0 && r < 8 &&
             c >= 0 && c < 8)
        {
            Piece piece = getPiece(r,c);

          if(piece != EMPTY)
         {
             if(byWhite &&
              (piece == WHITE_ROOK ||
               piece == WHITE_QUEEN))
                return true;

             if(!byWhite &&
              (piece == BLACK_ROOK ||
               piece == BLACK_QUEEN))
                return true;

            break;
        }

            r += rookDirections[dir][0];
            c += rookDirections[dir][1];
        }
        }
        //--------------------------------------------------
        // King Attacks
        //--------------------------------------------------

        for(int i = 0; i < 8; i++)
        {
            int r = row + kingDirections[i][0];
            int c = col + kingDirections[i][1];

            if(r < 0 || r >= 8 ||
               c < 0 || c >= 8)
               continue;

            Piece piece = getPiece(r,c);

            if(byWhite && piece == WHITE_KING)
            return true;

            if(!byWhite && piece == BLACK_KING)
            return true;
        }


    return false;
}

bool Board::isKingInCheck(bool whiteKing) const
{
    int kingRow = -1;
    int kingCol = -1;

    Piece kingPiece =
        whiteKing ? WHITE_KING : BLACK_KING;

    for(int row = 0; row < 8; row++)
    {
        for(int col = 0; col < 8; col++)
        {
            if(getPiece(row, col) == kingPiece)
            {
                kingRow = row;
                kingCol = col;
                break;
            }
        }

        if(kingRow != -1)
            break;
    }

    if(kingRow == -1)
        return false;

    return isSquareAttacked(
        kingRow,
        kingCol,
        !whiteKing
    );
}

void Board::makeNullMove()
{
    switchSide();
}

void Board::undoNullMove()
{
    switchSide();
}