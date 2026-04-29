#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <chrono>
#include <climits>
#include <cctype>
#include <cmath>

// ============================================================
// BASIC TYPES
// ============================================================

using U64  = unsigned long long;
using Move = unsigned int;

// ============================================================
// PIECES / COLORS
// ============================================================

enum Piece {
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE
};

enum Color {
    WHITE, BLACK, BOTH
};

// ============================================================
// SQUARES
// ============================================================

enum Square {
    A1,B1,C1,D1,E1,F1,G1,H1,
    A2,B2,C2,D2,E2,F2,G2,H2,
    A3,B3,C3,D3,E3,F3,G3,H3,
    A4,B4,C4,D4,E4,F4,G4,H4,
    A5,B5,C5,D5,E5,F5,G5,H5,
    A6,B6,C6,D6,E6,F6,G6,H6,
    A7,B7,C7,D7,E7,F7,G7,H7,
    A8,B8,C8,D8,E8,F8,G8,H8,
    NO_SQ
};

// ============================================================
// CASTLING RIGHTS
// ============================================================

enum Castling {
    WK = 1,
    WQ = 2,
    BK = 4,
    BQ = 8
};

// ============================================================
// MOVE ENCODING
// from(6) to(6) promo(3) cap dp ep castle
// ============================================================

inline Move encodeMove(int from, int to, int promo = 0, int cap = 0,
                       int dp = 0, int ep = 0, int cast = 0)
{
    return from |
          (to    << 6) |
          (promo << 12) |
          (cap   << 15) |
          (dp    << 16) |
          (ep    << 17) |
          (cast  << 18);
}

inline int  moveFrom(Move m)     { return  m        & 0x3F; }
inline int  moveTo(Move m)       { return (m >> 6 ) & 0x3F; }
inline int  movePromo(Move m)    { return (m >> 12) & 0x7;  }
inline bool moveCapture(Move m)  { return (m >> 15) & 1;    }
inline bool moveDP(Move m)       { return (m >> 16) & 1;    }
inline bool moveEP(Move m)       { return (m >> 17) & 1;    }
inline bool moveCastle(Move m)   { return (m >> 18) & 1;    }

// ============================================================
// BITBOARD HELPERS
// ============================================================

inline int popcount(U64 b)   { return __builtin_popcountll(b); }
inline int lsb(U64 b)        { return __builtin_ctzll(b); }

inline U64 popLSB(U64 &b)
{
    U64 bit = b & -b;
    b &= (b - 1);
    return bit;
}

inline int popLSBIdx(U64 &b)
{
    int sq = lsb(b);
    b &= (b - 1);
    return sq;
}

#define setBit(bb,sq)   ((bb) |=  (1ULL << (sq)))
#define clearBit(bb,sq) ((bb) &= ~(1ULL << (sq)))
#define getBit(bb,sq)   ((bb) &   (1ULL << (sq)))

// ============================================================
// FILE / RANK MASKS
// ============================================================

const U64 FILE_A = 0x0101010101010101ULL;
const U64 FILE_H = 0x8080808080808080ULL;

const U64 RANK_1 = 0x00000000000000FFULL;
const U64 RANK_2 = 0x000000000000FF00ULL;
const U64 RANK_7 = 0x00FF000000000000ULL;
const U64 RANK_8 = 0xFF00000000000000ULL;

// ============================================================
// ATTACK TABLES
// ============================================================

extern U64 pawnAttacks[2][64];
extern U64 knightAttacks[64];
extern U64 kingAttacks[64];

extern U64 bishopMasks[64];
extern U64 rookMasks[64];

extern U64 bishopAttacks[64][512];
extern U64 rookAttacks[64][4096];

extern U64 bishopMagics[64];
extern U64 rookMagics[64];

extern int bishopBits[64];
extern int rookBits[64];

void initAttacks();

U64 getBishopAttacks(int sq, U64 occ);
U64 getRookAttacks(int sq, U64 occ);
U64 getQueenAttacks(int sq, U64 occ);

// ============================================================
// FORWARD DECLARATION
// ============================================================

struct Board;

// ============================================================
// UNDO INFO
// ============================================================

struct UndoInfo
{
    U64 pieces[2][6];
    U64 occupancy[3];

    int pieceOn[64];
    int colorOn[64];

    int side;
    int enPassant;
    int castling;
    int halfMove;
    int fullMove;

    U64 zobristHash;
    int capturedPiece;

    void save(const Board& b, Move m, int cap);
    void restore(Board& b);
};

// ============================================================
// BOARD
// ============================================================

struct Board
{
    U64 pieces[2][6];
    U64 occupancy[3];

    int side;
    int enPassant;
    int castling;
    int halfMove;
    int fullMove;

    int pieceOn[64];
    int colorOn[64];

    U64 zobristHash;

    Board() { reset(); }

    void reset();
    void recalcOccupancy();

    void setFromFEN(const std::string& fen);
    std::string toFEN() const;
    void print() const;

    bool isSquareAttacked(int sq, int by) const;

    bool makeMove(Move mv, UndoInfo& undo);
    void unmakeMove(Move mv, UndoInfo& undo);

    bool inCheck() const
    {
        U64 kingBB = pieces[side][KING];
        if (!kingBB) return false;
        return isSquareAttacked(lsb(kingBB), side ^ 1);
    }
};

// ============================================================
// MOVE LIST
// ============================================================

struct MoveList
{
    Move moves[256];
    int count = 0;

    void add(Move m) { moves[count++] = m; }
    void clear()     { count = 0; }
};

void generateMoves(const Board& b, MoveList& ml);
void scoreAndSortMoves(const Board& b, MoveList& ml,
                       Move pvMove = 0, int* scores = nullptr);

// ============================================================
// EVALUATION
// ============================================================

int evaluate(const Board& b);

// ============================================================
// SEARCH
// ============================================================

struct SearchInfo
{
    int depth = 0;
    long long nodes = 0;
    bool stop = false;

    std::chrono::steady_clock::time_point startTime;
    int timeLimit = -1; // ms
};

int alphaBeta(Board& b, int alpha, int beta, int depth,
              SearchInfo& info, bool nullMoveAllowed = true);

int quiescence(Board& b, int alpha, int beta,
               SearchInfo& info, int qdepth = 0);

Move bestMove(Board& b, int depth, int timeLimitMs = -1);

// ============================================================
// TESTING
// ============================================================

unsigned long long perft(Board& b, int depth);
void divide(Board& b, int depth);

// ============================================================
// UCI
// ============================================================

std::string squareName(int sq);
std::string moveToStr(Move m);
Move strToMove(const Board& b, const std::string& str);
void uciLoop()
