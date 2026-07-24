#include "movegen.h"
#include "piece.h"
#include <iostream>

using namespace std;

inline bool isWhitePiece(Piece piece)
{
    return piece >= WHITE_PAWN && piece <= WHITE_KING;
}

inline bool isBlackPiece(Piece piece)
{
    return piece >= BLACK_PAWN && piece <= BLACK_KING;
}

const int knightOffsets[8][2] = {
    {-2, -1},
    {-2, 1},
    {-1, -2},
    {-1, 2},
    {1, -2},
    {1, 2},
    {2, -1},
    {2, 1}
};

const int bishopDirections[4][2] = {
    {-1, -1},
    {-1, 1},
    {1, -1},
    {1, 1}
};

const int rookDirections[4][2] = {
    {-1, 0},
    {1, 0},
    {0, -1},
    {0, 1}
};

const int queenDirections[8][2] = {
    {-1, -1},
    {-1, 1},
    {1, -1},
    {1, 1},
    {-1, 0},
    {1, 0},
    {0, -1},
    {0, 1}
};

const int kingOffsets[8][2] = {
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, -1},
    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1}
};

void generateSlidingMoves(
    const Board& board,
    std::vector<Move>& moves,
    Piece piece,
    int row,
    int col,
    const int directions[][2],
    int directionCount)
{
    for (int d = 0; d < directionCount; ++d) {
        int dr = directions[d][0];
        int dc = directions[d][1];

        int r = row + dr;
        int c = col + dc;

        while (r >= 0 && r < 8 && c >= 0 && c < 8) {
            Piece target = board.getPiece(r, c);

            bool whitePiece = (piece >= WHITE_PAWN && piece <= WHITE_KING);
            bool blackPiece = (piece >= BLACK_PAWN && piece <= BLACK_KING);

            // Friendly piece blocks movement
            if (whitePiece && target >= WHITE_PAWN && target <= WHITE_KING) {
                break;
            }

            if (blackPiece && target >= BLACK_PAWN && target <= BLACK_KING) {
                break;
            }

            Move move;
            move.fromRow = row;
            move.fromCol = col;
            move.toRow = r;
            move.toCol = c;
            moves.push_back(move);

            // Enemy piece can be captured, then stop
            if (target != EMPTY) {
                break;
            }

            r += dr;
            c += dc;
        }
    }
}

void addPromotionMoves(
    std::vector<Move>& moves,
    int fromRow,
    int fromCol,
    int toRow,
    int toCol,
    bool white)
{
    Piece pieces[4];

    if (white)
    {
        pieces[0] = WHITE_QUEEN;
        pieces[1] = WHITE_ROOK;
        pieces[2] = WHITE_BISHOP;
        pieces[3] = WHITE_KNIGHT;
    }
    else
    {
        pieces[0] = BLACK_QUEEN;
        pieces[1] = BLACK_ROOK;
        pieces[2] = BLACK_BISHOP;
        pieces[3] = BLACK_KNIGHT;
    }

    for (int i = 0; i < 4; i++)
    {
        Move move;

        move.fromRow = fromRow;
        move.fromCol = fromCol;
        move.toRow = toRow;
        move.toCol = toCol;

        move.isPromotion = true;
        move.promotionPiece = pieces[i];

        moves.push_back(move);
    }
}

std::vector<Move> generateCaptureMoves(Board& board, bool whiteToMove)
{
    std::vector<Move> captures;
    std::vector<Move> moves = generateLegalMoves(board, whiteToMove);

    for (const Move& move : moves) {
        Piece target = board.getPiece(move.toRow, move.toCol);

        if (target != EMPTY) {
            captures.push_back(move);
        }
    }

    return captures;
}

std::vector<Move> generateAllMoves(const Board& board, bool whiteToMove)
{
    std::vector<Move> moves;

    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            Piece piece = board.getPiece(row, col);

            if (whiteToMove) {
                if (piece >= BLACK_PAWN && piece <= BLACK_KING) {
                    continue;
                }
            } else {
                if (piece >= WHITE_PAWN && piece <= WHITE_KING) {
                    continue;
                }
            }

            switch (piece) {
                case WHITE_KNIGHT:
                case BLACK_KNIGHT: {
                    for (int i = 0; i < 8; ++i) {
                        int newRow = row + knightOffsets[i][0];
                        int newCol = col + knightOffsets[i][1];

                        if (newRow < 0 || newRow >= 8 || newCol < 0 || newCol >= 8) {
                            continue;
                        }

                        Piece target = board.getPiece(newRow, newCol);

                        if (piece == WHITE_KNIGHT && target >= WHITE_PAWN && target <= WHITE_KING) {
                            continue;
                        }

                        if (piece == BLACK_KNIGHT && target >= BLACK_PAWN && target <= BLACK_KING) {
                            continue;
                        }

                        Move move;
                        move.fromRow = row;
                        move.fromCol = col;
                        move.toRow = newRow;
                        move.toCol = newCol;
                        moves.push_back(move);
                    }
                    break;
                }

                case WHITE_BISHOP:
                case BLACK_BISHOP: {
                    generateSlidingMoves(board, moves, piece, row, col, bishopDirections, 4);
                    break;
                }

                case WHITE_ROOK:
                case BLACK_ROOK: {
                    generateSlidingMoves(board, moves, piece, row, col, rookDirections, 4);
                    break;
                }

                case WHITE_QUEEN:
                case BLACK_QUEEN: {
                    generateSlidingMoves(board, moves, piece, row, col, queenDirections, 8);
                    break;
                }

                case WHITE_KING:
                case BLACK_KING: {
                    for (int i = 0; i < 8; ++i) {
                        int newRow = row + kingOffsets[i][0];
                        int newCol = col + kingOffsets[i][1];

                        if (newRow < 0 || newRow >= 8 || newCol < 0 || newCol >= 8) {
                            continue;
                        }

                        Piece target = board.getPiece(newRow, newCol);

                        if (piece == WHITE_KING && target >= WHITE_PAWN && target <= WHITE_KING) {
                            continue;
                        }

                        if (piece == BLACK_KING && target >= BLACK_PAWN && target <= BLACK_KING) {
                            continue;
                        }

                        Move move;
                        move.fromRow = row;
                        move.fromCol = col;
                        move.toRow = newRow;
                        move.toCol = newCol;
                        moves.push_back(move);
                    }

                    //--------------------------------------------------
                    // White kingside castle
                    //--------------------------------------------------
                    if (piece == WHITE_KING &&
                        row == 7 && col == 4 &&
                        board.canWhiteCastleKingSide() &&
                        board.getPiece(7, 5) == EMPTY &&
                        board.getPiece(7, 6) == EMPTY &&
                        board.getPiece(7, 7) == WHITE_ROOK &&
                        !board.isSquareAttacked(7, 4, false) &&
                        !board.isSquareAttacked(7, 5, false) &&
                        !board.isSquareAttacked(7, 6, false)) {
                        Move castle;
                        castle.fromRow = 7;
                        castle.fromCol = 4;
                        castle.toRow = 7;
                        castle.toCol = 6;
                        castle.isCastle = true;
                        castle.isKingSideCastle = true;
                        moves.push_back(castle);
                    }

                    //--------------------------------------------------
                    // White queenside castle
                    //--------------------------------------------------
                    if (piece == WHITE_KING &&
                        row == 7 && col == 4 &&
                        board.canWhiteCastleQueenSide() &&
                        board.getPiece(7, 3) == EMPTY &&
                        board.getPiece(7, 2) == EMPTY &&
                        board.getPiece(7, 1) == EMPTY &&
                        board.getPiece(7, 0) == WHITE_ROOK &&
                        !board.isSquareAttacked(7, 4, false) &&
                        !board.isSquareAttacked(7, 3, false) &&
                        !board.isSquareAttacked(7, 2, false)) {
                        Move castle;
                        castle.fromRow = 7;
                        castle.fromCol = 4;
                        castle.toRow = 7;
                        castle.toCol = 2;
                        castle.isCastle = true;
                        castle.isQueenSideCastle = true;
                        moves.push_back(castle);
                    }

                    //--------------------------------------------------
                    // Black kingside castle
                    //--------------------------------------------------

                    if (piece == BLACK_KING &&
                        row == 0 && col == 4 &&
                        board.canBlackCastleKingSide() &&
                        board.getPiece(0, 5) == EMPTY &&
                        board.getPiece(0, 6) == EMPTY &&
                        board.getPiece(0, 7) == BLACK_ROOK &&
                        !board.isSquareAttacked(0, 4, true) &&
                        !board.isSquareAttacked(0, 5, true) &&
                        !board.isSquareAttacked(0, 6, true)) {
                        Move castle;
                        castle.fromRow = 0;
                        castle.fromCol = 4;
                        castle.toRow = 0;
                        castle.toCol = 6;
                        castle.isCastle = true;
                        castle.isKingSideCastle = true;
                        moves.push_back(castle);
                    }

                    //--------------------------------------------------
                    // Black queenside castle
                    //--------------------------------------------------
                    if (piece == BLACK_KING &&
                        row == 0 && col == 4 &&
                        board.canBlackCastleQueenSide() &&
                        board.getPiece(0, 1) == EMPTY &&
                        board.getPiece(0, 2) == EMPTY &&
                        board.getPiece(0, 3) == EMPTY &&
                        board.getPiece(0, 0) == BLACK_ROOK &&
                        !board.isSquareAttacked(0, 4, true) &&
                        !board.isSquareAttacked(0, 3, true) &&
                        !board.isSquareAttacked(0, 2, true)) {
                        Move castle;
                        castle.fromRow = 0;
                        castle.fromCol = 4;
                        castle.toRow = 0;
                        castle.toCol = 2;
                        castle.isCastle = true;
                        castle.isQueenSideCastle = true;
                        moves.push_back(castle);
                    }

                    break;
                }

                case WHITE_PAWN: {
                    // One square
                    if (row > 0 && board.getPiece(row - 1, col) == EMPTY) {
                        if (row == 1) {
                            Piece promotions[] = {
                                WHITE_QUEEN,
                                WHITE_ROOK,
                                WHITE_BISHOP,
                                WHITE_KNIGHT
                            };

                            for (Piece piece : promotions) {
                                Move move;

                                move.fromRow = row;
                                move.fromCol = col;
                                move.toRow = 0;
                                move.toCol = col;

                                move.isPromotion = true;
                                move.promotionPiece = piece;

                                moves.push_back(move);
                            }
                        } else {
                            Move move;

                            move.fromRow = row;
                            move.fromCol = col;
                            move.toRow = row - 1;
                            move.toCol = col;

                            moves.push_back(move);
                        }

                        // Two squares
                        if (row == 6 && board.getPiece(row - 2, col) == EMPTY) {
                            Move move2;
                            move2.fromRow = row;
                            move2.fromCol = col;
                            move2.toRow = row - 2;
                            move2.toCol = col;
                            moves.push_back(move2);
                        }
                    }

                    // Capture left
                    if (row > 0 && col > 0) {
                        Piece target = board.getPiece(row - 1, col - 1);

                        if (target != EMPTY && isBlackPiece(target)) {
                            if (row == 1) {
                                Piece promotions[] = {
                                    WHITE_QUEEN,
                                    WHITE_ROOK,
                                    WHITE_BISHOP,
                                    WHITE_KNIGHT
                                };

                                for (Piece piece : promotions) {
                                    Move move;

                                    move.fromRow = row;
                                    move.fromCol = col;
                                    move.toRow = 0;
                                    move.toCol = col - 1;

                                    move.isPromotion = true;
                                    move.promotionPiece = piece;

                                    moves.push_back(move);
                                }
                            } else {
                                Move move;

                                move.fromRow = row;
                                move.fromCol = col;
                                move.toRow = row - 1;
                                move.toCol = col - 1;

                                moves.push_back(move);
                            }
                        }
                    }

                    // Capture right
                    if (row > 0 && col < 7) {
                        Piece target = board.getPiece(row - 1, col + 1);

                        if (target != EMPTY && isBlackPiece(target)) {
                            if (row == 1) {
                                Piece promotions[] = {
                                    WHITE_QUEEN,
                                    WHITE_ROOK,
                                    WHITE_BISHOP,
                                    WHITE_KNIGHT
                                };

                                for (Piece piece : promotions) {
                                    Move move;

                                    move.fromRow = row;
                                    move.fromCol = col;
                                    move.toRow = 0;
                                    move.toCol = col + 1;

                                    move.isPromotion = true;
                                    move.promotionPiece = piece;

                                    moves.push_back(move);
                                }
                            } else {
                                Move move;

                                move.fromRow = row;
                                move.fromCol = col;
                                move.toRow = row - 1;
                                move.toCol = col + 1;

                                moves.push_back(move);
                            }
                        }
                    }

                    // En Passant
                    if (row == 3) {
                        if (col > 0 &&
                            board.getEnPassantRow() == 2 &&
                            board.getEnPassantCol() == col - 1) {
                            Move move;
                            move.fromRow = row;
                            move.fromCol = col;
                            move.toRow = 2;
                            move.toCol = col - 1;
                            move.isEnPassant = true;
                            moves.push_back(move);
                        }

                        if (col < 7 &&
                            board.getEnPassantRow() == 2 &&
                            board.getEnPassantCol() == col + 1) {
                            Move move;
                            move.fromRow = row;
                            move.fromCol = col;
                            move.toRow = 2;
                            move.toCol = col + 1;
                            move.isEnPassant = true;
                            moves.push_back(move);
                        }
                    }

                    break;
                }

                case BLACK_PAWN: {
                    // One square
                    if (row < 7 && board.getPiece(row + 1, col) == EMPTY) {
                        if (row == 6) {
                            Piece promotions[] = {
                                BLACK_QUEEN,
                                BLACK_ROOK,
                                BLACK_BISHOP,
                                BLACK_KNIGHT
                            };

                            for (Piece piece : promotions) {
                                Move move;

                                move.fromRow = row;
                                move.fromCol = col;
                                move.toRow = 7;
                                move.toCol = col;

                                move.isPromotion = true;
                                move.promotionPiece = piece;

                                moves.push_back(move);
                            }
                        } else {
                            Move move;

                            move.fromRow = row;
                            move.fromCol = col;
                            move.toRow = row + 1;
                            move.toCol = col;

                            moves.push_back(move);

                            // Two squares
                            if (row == 1 && board.getPiece(row + 2, col) == EMPTY) {
                                Move move2;

                                move2.fromRow = row;
                                move2.fromCol = col;
                                move2.toRow = row + 2;
                                move2.toCol = col;

                                moves.push_back(move2);
                            }
                        }
                    }

                    // Capture left
                    if (row < 7 && col > 0) {
                        Piece target = board.getPiece(row + 1, col - 1);

                        if (target != EMPTY && isWhitePiece(target)) {
                            if (row == 6) {
                                Piece promotions[] = {
                                    BLACK_QUEEN,
                                    BLACK_ROOK,
                                    BLACK_BISHOP,
                                    BLACK_KNIGHT
                                };

                                for (Piece piece : promotions) {
                                    Move move;

                                    move.fromRow = row;
                                    move.fromCol = col;
                                    move.toRow = 7;
                                    move.toCol = col - 1;

                                    move.isPromotion = true;
                                    move.promotionPiece = piece;

                                    moves.push_back(move);
                                }
                            } else {
                                Move move;

                                move.fromRow = row;
                                move.fromCol = col;
                                move.toRow = row + 1;
                                move.toCol = col - 1;

                                moves.push_back(move);
                            }
                        }
                    }

                    // Capture right
                    if (row < 7 && col < 7) {
                        Piece target = board.getPiece(row + 1, col + 1);

                        if (target != EMPTY && isWhitePiece(target)) {
                            if (row == 6) {
                                Piece promotions[] = {
                                    BLACK_QUEEN,
                                    BLACK_ROOK,
                                    BLACK_BISHOP,
                                    BLACK_KNIGHT
                                };

                                for (Piece piece : promotions) {
                                    Move move;

                                    move.fromRow = row;
                                    move.fromCol = col;
                                    move.toRow = 7;
                                    move.toCol = col + 1;

                                    move.isPromotion = true;
                                    move.promotionPiece = piece;

                                    moves.push_back(move);
                                }
                            } else {
                                Move move;

                                move.fromRow = row;
                                move.fromCol = col;
                                move.toRow = row + 1;
                                move.toCol = col + 1;

                                moves.push_back(move);
                            }
                        }
                    }

                    // En Passant
                    if (row == 4) {
                        if (col > 0 &&
                            board.getEnPassantRow() == 5 &&
                            board.getEnPassantCol() == col - 1) {
                            Move move;

                            move.fromRow = row;
                            move.fromCol = col;
                            move.toRow = 5;
                            move.toCol = col - 1;
                            move.isEnPassant = true;

                            moves.push_back(move);
                        }

                        if (col < 7 &&
                            board.getEnPassantRow() == 5 &&
                            board.getEnPassantCol() == col + 1) {
                            Move move;

                            move.fromRow = row;
                            move.fromCol = col;
                            move.toRow = 5;
                            move.toCol = col + 1;
                            move.isEnPassant = true;

                            moves.push_back(move);
                        }
                    }

                    break;
                }

                default:
                    break;
            }
        }
    }

    return moves;
}

std::vector<Move> generateLegalMoves(Board& board, bool whiteToMove)
{
    bool sideBefore = board.isWhiteToMove();

    std::vector<Move> legalMoves;
    std::vector<Move> moves = generateAllMoves(board, whiteToMove);

    for (Move& move : moves) {
        board.makeMove(move);

        bool kingInCheck = board.isKingInCheck(whiteToMove);

        std::string u;

        u += char('a' + move.fromCol);
        u += char('8' - move.fromRow);
        u += char('a' + move.toCol);
        u += char('8' - move.toRow);

        if (!kingInCheck) {
            legalMoves.push_back(move);
        }

        board.undoMove(move);
    }

    return legalMoves;
}

bool isCheckmate(Board& board, bool whiteToMove)
{
    if (!board.isKingInCheck(whiteToMove)) {
        return false;
    }

    std::vector<Move> legalMoves = generateLegalMoves(board, whiteToMove);
    return legalMoves.empty();
}
