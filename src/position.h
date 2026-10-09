#ifndef ZERO_POSITION_H
#define ZERO_POSITION_H

#include <array>
#include <string>
#include "bitboard.h"
#include "move.h"

struct StateInfo {
    StateInfo* previous = nullptr;
    Zero::Key key = 0;

    bool whiteKingSideCastle = false;
    bool whiteQueenSideCastle = false;
    bool blackKingSideCastle = false;
    bool blackQueenSideCastle = false;

    Zero::Square epSquare = Zero::SQ_NONE;
    bool whiteToMove = true;
    Piece capturedPiece = EMPTY;
    Piece movedPiece = EMPTY;

    // FIDE fifty/seventy-five-move counter. It advances on reversible moves
    // and resets after pawn moves or captures.
    int halfmoveClock = 0;

    // Null moves are search-only and must not participate in repetition scans.
    bool isNullMove = false;
};

class Position
{
public:
    Position();
    Position(const Position& other);
    Position& operator=(const Position& other);

    static void init();

    void initialize();
    void printBoard() const;

    void doMove(const Move& move, StateInfo& newState);
    void undoMove(const Move& move);

    Piece piece_on(Zero::Square sq) const { return board_[sq]; }
    Piece getPiece(int row, int col) const;
    Piece getPiece(Zero::Square sq) const { return piece_on(sq); }

    Zero::Bitboard pieces() const { return occupancy_; }
    Zero::Bitboard pieces(Zero::Color c) const { return byColor_[c]; }
    Zero::Bitboard pieces(Zero::PieceType pt) const { return byType_[pt]; }
    Zero::Bitboard pieces(Zero::Color c, Zero::PieceType pt) const { return byColor_[c] & byType_[pt]; }
    Zero::Bitboard pieces(Piece pc) const { return byPiece_[pc]; }
    Zero::Key key() const { return state_->key; }
    Zero::Key recomputeKey() const;

    Zero::Square king_square(Zero::Color c) const;
    Zero::Bitboard attackers_to(Zero::Square sq, Zero::Color by) const;
    bool isSquareAttacked(int row, int col, bool byWhite) const;
    bool isSquareAttacked(Zero::Square sq, Zero::Color by) const;
    bool isKingInCheck(bool whiteKing) const;

    void clearBoard();
    void setPiece(int row, int col, Piece piece);
    void setPiece(Zero::Square sq, Piece piece);

    bool isWhiteToMove() const;
    void setSideToMove(bool white);
    void switchSide();

    bool loadFEN(const std::string& fen);
    bool equals(const Position& other) const;

    bool canWhiteCastleKingSide() const;
    bool canWhiteCastleQueenSide() const;
    bool canBlackCastleKingSide() const;
    bool canBlackCastleQueenSide() const;

    Zero::Square ep_square() const { return state_->epSquare; }
    int getEnPassantRow() const;
    int getEnPassantCol() const;

    int halfmove_clock() const { return state_->halfmoveClock; }

    void doNullMove(StateInfo& newState);
    void undoNullMove();

    StateInfo* state() { return state_; }
    const StateInfo* state() const { return state_; }

private:
    void putPiece(Zero::Square sq, Piece piece);
    void removePiece(Zero::Square sq, Piece piece);
    void movePiece(Zero::Square from, Zero::Square to, Piece piece);
    void setCastlingRightsForRookSquare(Zero::Square sq);
    void clearCastlingRightsForCapturedRook(Zero::Square sq, Piece captured);
    int castlingRightsMask() const;

    std::array<Piece, Zero::SQUARE_NB> board_{};
    std::array<Zero::Bitboard, PIECE_NB> byPiece_{};
    std::array<Zero::Bitboard, Zero::COLOR_NB> byColor_{};
    std::array<Zero::Bitboard, Zero::PIECE_TYPE_NB> byType_{};
    Zero::Bitboard occupancy_ = 0;

    StateInfo rootState_{};
    StateInfo* state_ = &rootState_;
};

#endif
