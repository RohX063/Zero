#include "chess.h"
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>

// ============================================================
//  TYPES & CONSTANTS
// ============================================================

// ============================================================
//  ATTACK TABLES (Magic Bitboards)
// ============================================================

U64 pawnAttacks[2][64];
U64 knightAttacks[64];
U64 kingAttacks[64];
U64 bishopMasks[64];
U64 rookMasks[64];
U64 bishopAttacks[64][512];
U64 rookAttacks[64][4096];
U64 bishopMagics[64];
U64 rookMagics[64];
int bishopBits[64];
int rookBits[64];

static U64 BISHOP_MAGIC[64] = {
    0x0002020202020200ULL,0x0002020202020000ULL,0x0004010202000000ULL,0x0004040080000000ULL,
    0x0001104000000000ULL,0x0000821040000000ULL,0x0000410410400000ULL,0x0000104104104000ULL,
    0x0000040404040400ULL,0x0000020202020200ULL,0x0000040102020000ULL,0x0000040400800000ULL,
    0x0000011040000000ULL,0x0000008210400000ULL,0x0000004104104000ULL,0x0000002082082000ULL,
    0x0004000808080800ULL,0x0002000404040400ULL,0x0001000202020200ULL,0x0000800802004000ULL,
    0x0000800400A00000ULL,0x0000200100884000ULL,0x0000400082082000ULL,0x0000200041041000ULL,
    0x0002080010101000ULL,0x0001040008080800ULL,0x0000208004010400ULL,0x0000404004010200ULL,
    0x0000840000802000ULL,0x0000404002011000ULL,0x0000808001041000ULL,0x0000404000820800ULL,
    0x0001041000202000ULL,0x0000820800101000ULL,0x0000104400080800ULL,0x0000020080080080ULL,
    0x0000404040040100ULL,0x0000808100020100ULL,0x0001010100020800ULL,0x0000808080010400ULL,
    0x0000820820004000ULL,0x0000410410002000ULL,0x0000082088001000ULL,0x0000002011000800ULL,
    0x0000080100400400ULL,0x0001010101000200ULL,0x0002020202000400ULL,0x0001010101000200ULL,
    0x0000410410400000ULL,0x0000208208200000ULL,0x0000002084000000ULL,0x0000000020880000ULL,
    0x0000001002020000ULL,0x0000040408020000ULL,0x0004040404040000ULL,0x0002020202020000ULL,
    0x0000104104104000ULL,0x0000002082082000ULL,0x0000000020841000ULL,0x0000000008422000ULL,
    0x0000000000208800ULL,0x0000000100200400ULL,0x0000002040800200ULL,0x0000004081020100ULL
};

static U64 ROOK_MAGIC[64] = {
    0x8a80104000800020ULL, 0x140002000100040ULL,  0x2801880a0017001ULL,  0x100081001000420ULL,
    0x200020010080420ULL,  0x3001c0002010008ULL,  0x8480008002000100ULL, 0x2080088004402900ULL,
    0x800098204000ULL,     0x2024401000200040ULL, 0x100802000801000ULL,  0x120800800801000ULL,
    0x208808088000400ULL,  0x2802200800400ULL,    0x2200800100020080ULL, 0x801000060821100ULL,
    0x80044006422000ULL,   0x100808020004000ULL,  0x12108a0010204200ULL, 0x140848010000802ULL,
    0x481828014002800ULL,  0x8094004002004100ULL, 0x4010040010010802ULL, 0x20008806104ULL,
    0x100400080208000ULL,  0x2040002120081000ULL, 0x21200680100081ULL,   0x20100080080080ULL,
    0x2000a00200410ULL,    0x20080800400ULL,      0x80088400100102ULL,   0x80004600042881ULL,
    0x4040008040800020ULL, 0x440003000200801ULL,  0x4200011004500ULL,    0x188020010100100ULL,
    0x14800401802800ULL,   0x2080040080800200ULL, 0x124080204001001ULL,  0x200046502000484ULL,
    0x480400080088020ULL,  0x1000422010034000ULL, 0x30200100110040ULL,   0x100021010009ULL,
    0x2002080100110004ULL, 0x202008004008002ULL,  0x20020004010100ULL,   0x2048440040820001ULL,
    0x101002200408200ULL,  0x40802000401080ULL,   0x4008142004410100ULL, 0x2060820c0120200ULL,
    0x1001004080100ULL,    0x20c020080040080ULL,  0x2935610830022400ULL, 0x44440041009200ULL,
    0x280001040802101ULL,  0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
    0x20030a0244872ULL,    0x12001008414402ULL,   0x2006104900a0804ULL,  0x1004081002402ULL
};

static int BISHOP_BITS_TABLE[64] = {
    6,5,5,5,5,5,5,6, 5,5,5,5,5,5,5,5, 5,5,7,7,7,7,5,5, 5,5,7,9,9,7,5,5,
    5,5,7,9,9,7,5,5, 5,5,7,7,7,7,5,5, 5,5,5,5,5,5,5,5, 6,5,5,5,5,5,5,6
};

static int ROOK_BITS_TABLE[64] = {
    12,11,11,11,11,11,11,12, 11,10,10,10,10,10,10,11, 11,10,10,10,10,10,10,11, 11,10,10,10,10,10,10,11,
    11,10,10,10,10,10,10,11, 11,10,10,10,10,10,10,11, 11,10,10,10,10,10,10,11, 12,11,11,11,11,11,11,12
};

static U64 bishopAttackOTF(int sq, U64 occ) {
    U64 atk = 0;
    int r = sq/8, f = sq%8, tr, tf;
    for(tr=r+1,tf=f+1; tr<8&&tf<8; tr++,tf++) { atk |= 1ULL<<(tr*8+tf); if(occ & (1ULL<<(tr*8+tf))) break; }
    for(tr=r+1,tf=f-1; tr<8&&tf>=0; tr++,tf--) { atk |= 1ULL<<(tr*8+tf); if(occ & (1ULL<<(tr*8+tf))) break; }
    for(tr=r-1,tf=f+1; tr>=0&&tf<8; tr--,tf++) { atk |= 1ULL<<(tr*8+tf); if(occ & (1ULL<<(tr*8+tf))) break; }
    for(tr=r-1,tf=f-1; tr>=0&&tf>=0; tr--,tf--) { atk |= 1ULL<<(tr*8+tf); if(occ & (1ULL<<(tr*8+tf))) break; }
    return atk;
}

static U64 rookAttackOTF(int sq, U64 occ) {
    U64 atk = 0;
    int r = sq/8, f = sq%8, tr, tf;
    for(tr=r+1; tr<8; tr++) { atk |= 1ULL<<(tr*8+f); if(occ & (1ULL<<(tr*8+f))) break; }
    for(tr=r-1; tr>=0; tr--) { atk |= 1ULL<<(tr*8+f); if(occ & (1ULL<<(tr*8+f))) break; }
    for(tf=f+1; tf<8; tf++) { atk |= 1ULL<<(r*8+tf); if(occ & (1ULL<<(r*8+tf))) break; }
    for(tf=f-1; tf>=0; tf--) { atk |= 1ULL<<(r*8+tf); if(occ & (1ULL<<(r*8+tf))) break; }
    return atk;
}

static U64 bishopMaskGen(int sq) {
    U64 atk = 0;
    int r = sq/8, f = sq%8, tr, tf;
    for(tr=r+1,tf=f+1; tr<7&&tf<7; tr++,tf++) atk |= 1ULL<<(tr*8+tf);
    for(tr=r+1,tf=f-1; tr<7&&tf>0; tr++,tf--) atk |= 1ULL<<(tr*8+tf);
    for(tr=r-1,tf=f+1; tr>0&&tf<7; tr--,tf++) atk |= 1ULL<<(tr*8+tf);
    for(tr=r-1,tf=f-1; tr>0&&tf>0; tr--,tf--) atk |= 1ULL<<(tr*8+tf);
    return atk;
}

static U64 rookMaskGen(int sq) {
    U64 atk = 0;
    int r = sq/8, f = sq%8, tr, tf;
    for(tr=r+1; tr<=6; tr++) atk |= 1ULL<<(tr*8+f);
    for(tr=r-1; tr>=1; tr--) atk |= 1ULL<<(tr*8+f);
    for(tf=f+1; tf<=6; tf++) atk |= 1ULL<<(r*8+tf);
    for(tf=f-1; tf>=1; tf--) atk |= 1ULL<<(r*8+tf);
    return atk;
}

static U64 setOccupancy(int idx, int bits, U64 mask) {
    U64 occ = 0;
    for(int i = 0; i < bits; i++) {
        int sq = popLSBIdx(mask);
        if(idx & (1 << i)) occ |= 1ULL << sq;
    }
    return occ;
}

U64 getBishopAttacks(int sq, U64 occ) {
    return bishopAttackOTF(sq, occ);
}

U64 getRookAttacks(int sq, U64 occ) {
    return rookAttackOTF(sq, occ);
}

U64 getQueenAttacks(int sq, U64 occ) {
    return bishopAttackOTF(sq, occ) | rookAttackOTF(sq, occ);
}

void initAttacks() {
    // Pawn attacks
    for(int sq = 0; sq < 64; sq++) {
        U64 b = 1ULL << sq;
        pawnAttacks[WHITE][sq] = ((b << 9) & ~FILE_A) | ((b << 7) & ~FILE_H);
        pawnAttacks[BLACK][sq] = ((b >> 7) & ~FILE_A) | ((b >> 9) & ~FILE_H);
    }
    // Knight attacks
    for(int sq = 0; sq < 64; sq++) {
        U64 b = 1ULL << sq, atk = 0;
        atk |= ((b << 17) & ~FILE_A);
        atk |= ((b << 15) & ~FILE_H);
        atk |= ((b << 10) & ~(FILE_A | (FILE_A << 1)));
        atk |= ((b << 6) & ~(FILE_H | (FILE_H >> 1)));
        atk |= ((b >> 17) & ~FILE_H);
        atk |= ((b >> 15) & ~FILE_A);
        atk |= ((b >> 10) & ~(FILE_H | (FILE_H >> 1)));
        atk |= ((b >> 6) & ~(FILE_A | (FILE_A << 1)));
        knightAttacks[sq] = atk;
    }
    // King attacks
    for(int sq = 0; sq < 64; sq++) {
        U64 b = 1ULL << sq, atk = 0;
        atk |= (b << 8);
        atk |= (b >> 8);
        atk |= ((b << 1) & ~FILE_A);
        atk |= ((b >> 1) & ~FILE_H);
        atk |= ((b << 9) & ~FILE_A);
        atk |= ((b << 7) & ~FILE_H);
        atk |= ((b >> 7) & ~FILE_A);
        atk |= ((b >> 9) & ~FILE_H);
        kingAttacks[sq] = atk;
    }
    // Magic bitboards for bishops and rooks
    for(int sq = 0; sq < 64; sq++) {
        bishopMasks[sq] = bishopMaskGen(sq);
        rookMasks[sq] = rookMaskGen(sq);
        bishopBits[sq] = BISHOP_BITS_TABLE[sq];
        rookBits[sq] = ROOK_BITS_TABLE[sq];
        bishopMagics[sq] = BISHOP_MAGIC[sq];
        rookMagics[sq] = ROOK_MAGIC[sq];
        
        int bOcc = 1 << bishopBits[sq];
        for(int i = 0; i < bOcc; i++) {
            U64 occ = setOccupancy(i, bishopBits[sq], bishopMasks[sq]);
            int idx = (int)((occ * bishopMagics[sq]) >> (64 - bishopBits[sq]));
            bishopAttacks[sq][idx] = bishopAttackOTF(sq, occ);
        }
        int rOcc = 1 << rookBits[sq];
        for(int i = 0; i < rOcc; i++) {
            U64 occ = setOccupancy(i, rookBits[sq], rookMasks[sq]);
            int idx = (int)((occ * rookMagics[sq]) >> (64 - rookBits[sq]));
            rookAttacks[sq][idx] = rookAttackOTF(sq, occ);
        }
    }
}



// ============================================================
//  ZOBRIST HASHING
// ============================================================

static U64 ZP[2][6][64], ZSIDE, ZCASTLE[16], ZEP[8];
static bool zInited = false;

static U64 xrand() {
    static U64 s = 0x123456789ABCDEFULL;
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
}

static void initZobrist() {
    if(zInited) return;
    zInited = true;
    for(int c = 0; c < 2; c++)
        for(int p = 0; p < 6; p++)
            for(int s = 0; s < 64; s++)
                ZP[c][p][s] = xrand();
    ZSIDE = xrand();
    for(int i = 0; i < 16; i++) ZCASTLE[i] = xrand();
    for(int i = 0; i < 8; i++) ZEP[i] = xrand();
}

static U64 hashBoard(const Board &b) {
    U64 h = 0;
    for(int c = 0; c < 2; c++)
        for(int p = 0; p < 6; p++) {
            U64 bb = b.pieces[c][p];
            while(bb) {
                int sq = popLSBIdx(bb);
                h ^= ZP[c][p][sq];
            }
        }
    if(b.side == BLACK) h ^= ZSIDE;
    h ^= ZCASTLE[b.castling & 15];
    if(b.enPassant != NO_SQ) h ^= ZEP[b.enPassant % 8];
    return h;
}

// ============================================================
//  BOARD IMPLEMENTATION
// ============================================================

static const char PIECE_CHARS[2][7] = {
    {'P','N','B','R','Q','K','.'},
    {'p','n','b','r','q','k','.'}
};

static const int castleMask[64] = {
    13,15,15,15,12,15,15,14,15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15, 7,15,15,15, 3,15,15,11
};

void UndoInfo::save(const Board &b, Move m, int cap) {
    memcpy(pieces, b.pieces, sizeof(pieces));
    memcpy(occupancy, b.occupancy, sizeof(occupancy));
    memcpy(pieceOn, b.pieceOn, sizeof(pieceOn));
    memcpy(colorOn, b.colorOn, sizeof(colorOn));
    side = b.side;
    enPassant = b.enPassant;
    castling = b.castling;
    halfMove = b.halfMove;
    fullMove = b.fullMove;
    zobristHash = b.zobristHash;
    capturedPiece = cap;
}

void UndoInfo::restore(Board &b) {
    memcpy(b.pieces, pieces, sizeof(pieces));
    memcpy(b.occupancy, occupancy, sizeof(occupancy));
    memcpy(b.pieceOn, pieceOn, sizeof(pieceOn));
    memcpy(b.colorOn, colorOn, sizeof(colorOn));
    b.side = side;
    b.enPassant = enPassant;
    b.castling = castling;
    b.halfMove = halfMove;
    b.fullMove = fullMove;
    b.zobristHash = zobristHash;
}

void Board::reset() {
    memset(pieces, 0, sizeof(pieces));
    memset(occupancy, 0, sizeof(occupancy));
    for(int i = 0; i < 64; i++) {
        pieceOn[i] = NO_PIECE;
        colorOn[i] = BOTH;
    }
    side = WHITE;
    enPassant = NO_SQ;
    castling = 0;
    halfMove = 0;
    fullMove = 1;
    zobristHash = 0;
}

void Board::setFromFEN(const std::string &fen) {
    reset();
    std::istringstream ss(fen);
    std::string board, sideStr, castleStr, epStr;
    ss >> board >> sideStr >> castleStr >> epStr >> halfMove >> fullMove;
    int sq = 56;
    for(char c : board) {
        if(c == '/') sq -= 16;
        else if(c >= '1' && c <= '8') sq += c - '0';
        else {
            int col = (c >= 'a' && c <= 'z') ? BLACK : WHITE;
            char lc = tolower(c);
            int pt = (lc == 'p') ? PAWN : (lc == 'n') ? KNIGHT : (lc == 'b') ? BISHOP :
                     (lc == 'r') ? ROOK : (lc == 'q') ? QUEEN : KING;
            setBit(pieces[col][pt], sq);
            pieceOn[sq] = pt;
            colorOn[sq] = col;
            sq++;
        }
    }
    recalcOccupancy();
    side = (sideStr == "w") ? WHITE : BLACK;
    castling = 0;
    for(char c : castleStr) {
        if(c == 'K') castling |= WK;
        if(c == 'Q') castling |= WQ;
        if(c == 'k') castling |= BK;
        if(c == 'q') castling |= BQ;
    }
    if(epStr != "-" && epStr.length() >= 2) {
        enPassant = (epStr[1] - '1') * 8 + (epStr[0] - 'a');
    }
    zobristHash = hashBoard(*this);
}

std::string Board::toFEN() const {
    std::string fen;
    for(int r = 7; r >= 0; r--) {
        int empty = 0;
        for(int f = 0; f < 8; f++) {
            int sq = r * 8 + f;
            if(pieceOn[sq] == NO_PIECE) empty++;
            else {
                if(empty > 0) { fen += char('0' + empty); empty = 0; }
                fen += PIECE_CHARS[colorOn[sq] == BLACK ? 1 : 0][pieceOn[sq]];
            }
        }
        if(empty > 0) fen += char('0' + empty);
        if(r > 0) fen += '/';
    }
    fen += (side == WHITE) ? " w " : " b ";
    std::string c;
    if(castling & WK) c += 'K';
    if(castling & WQ) c += 'Q';
    if(castling & BK) c += 'k';
    if(castling & BQ) c += 'q';
    fen += c.empty() ? "-" : c;
    fen += ' ';
    if(enPassant == NO_SQ) fen += "-";
    else {
        fen += char('a' + (enPassant % 8));
        fen += char('1' + (enPassant / 8));
    }
    fen += ' ' + std::to_string(halfMove) + ' ' + std::to_string(fullMove);
    return fen;
}

void Board::print() const {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for(int r = 7; r >= 0; r--) {
        std::cout << (r + 1) << " |";
        for(int f = 0; f < 8; f++) {
            int sq = r * 8 + f;
            char c = (pieceOn[sq] == NO_PIECE) ? '.' : PIECE_CHARS[colorOn[sq] == BLACK ? 1 : 0][pieceOn[sq]];
            std::cout << " " << c << " |";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
}

void Board::recalcOccupancy() {
    occupancy[WHITE] = occupancy[BLACK] = 0;
    for(int c = 0; c < 2; c++)
        for(int p = 0; p < 6; p++)
            occupancy[c] |= pieces[c][p];
    occupancy[BOTH] = occupancy[WHITE] | occupancy[BLACK];
}

bool Board::isSquareAttacked(int sq, int by) const {
    if(pawnAttacks[by ^ 1][sq] & pieces[by][PAWN]) return true;
    if(knightAttacks[sq] & pieces[by][KNIGHT]) return true;
    if(kingAttacks[sq] & pieces[by][KING]) return true;
    if(getBishopAttacks(sq, occupancy[BOTH]) & (pieces[by][BISHOP] | pieces[by][QUEEN])) return true;
    if(getRookAttacks(sq, occupancy[BOTH]) & (pieces[by][ROOK] | pieces[by][QUEEN])) return true;
    return false;
}

bool Board::makeMove(Move mv, UndoInfo &undo) {
    undo.save(*this, mv, NO_PIECE);
    
    int from = moveFrom(mv), to = moveTo(mv), col = side, opp = col ^ 1;
    int pt = pieceOn[from];
    
    if(from < 0 || from > 63 || to < 0 || to > 63 || pt == NO_PIECE || colorOn[from] != col)
        return false;
    
    // Remove from source
    zobristHash ^= ZP[col][pt][from];
    
    if(moveCapture(mv) && !moveEP(mv)) {
        undo.capturedPiece = pieceOn[to];
        int capturedPt = pieceOn[to];
        if(capturedPt != NO_PIECE) {
            zobristHash ^= ZP[opp][capturedPt][to];
            clearBit(pieces[opp][capturedPt], to);
        }
        pieceOn[to] = NO_PIECE;
        colorOn[to] = BOTH;
        halfMove = 0;
    } else if(pt == PAWN) {
        halfMove = 0;
    } else {
        halfMove++;
    }
    
    clearBit(pieces[col][pt], from);
    setBit(pieces[col][pt], to);
    zobristHash ^= ZP[col][pt][to];
    
    pieceOn[to] = pt;
    colorOn[to] = col;
    pieceOn[from] = NO_PIECE;
    colorOn[from] = BOTH;
    
    // En passant
    if(enPassant != NO_SQ) {
        zobristHash ^= ZEP[enPassant % 8];
    }
    enPassant = NO_SQ;
    
    if(moveEP(mv)) {
        int c2 = to + (col == WHITE ? -8 : 8);
        undo.capturedPiece = PAWN;
        zobristHash ^= ZP[opp][PAWN][c2];
        clearBit(pieces[opp][PAWN], c2);
        pieceOn[c2] = NO_PIECE;
        colorOn[c2] = BOTH;
    }
    
    if(moveDP(mv)) {
        enPassant = to + (col == WHITE ? -8 : 8);
        zobristHash ^= ZEP[enPassant % 8];
    }
    
    if(movePromo(mv)) {
        int p = movePromo(mv);
        zobristHash ^= ZP[col][PAWN][to];
        zobristHash ^= ZP[col][p][to];
        clearBit(pieces[col][PAWN], to);
        setBit(pieces[col][p], to);
        pieceOn[to] = p;
    }
    
    if(moveCastle(mv)) {
        int rf, rt;
        if(to == G1) { rf = H1; rt = F1; }
        else if(to == C1) { rf = A1; rt = D1; }
        else if(to == G8) { rf = H8; rt = F8; }
        else { rf = A8; rt = D8; }
        zobristHash ^= ZP[col][ROOK][rf];
        zobristHash ^= ZP[col][ROOK][rt];
        clearBit(pieces[col][ROOK], rf);
        setBit(pieces[col][ROOK], rt);
        pieceOn[rt] = ROOK;
        colorOn[rt] = col;
        pieceOn[rf] = NO_PIECE;
        colorOn[rf] = BOTH;
    }
    
    U64 oldCastling = castling;
    castling &= castleMask[from] & castleMask[to];
    zobristHash ^= ZCASTLE[oldCastling & 15];
    zobristHash ^= ZCASTLE[castling & 15];
    
    side = opp;
    zobristHash ^= ZSIDE;
    if(side == WHITE) fullMove++;
    
    recalcOccupancy();
    
    U64 kb = pieces[col][KING];
    if(!kb) {
        undo.restore(*this);
        return false;
    }
    if(isSquareAttacked(lsb(kb), side)) {
        undo.restore(*this);
        return false;
    }
    return true;
}

void Board::unmakeMove(Move m, UndoInfo &undo) {
    undo.restore(*this);
}

// ============================================================
//  MOVE GENERATION
// ============================================================

static void addPawnMoves(const Board &b, MoveList &ml, int from, int to, bool cap) {
    int pr = (b.side == WHITE) ? 7 : 0;
    if(to / 8 == pr) {
        for(int p = 1; p <= 4; p++)
            ml.add(encodeMove(from, to, p, cap ? 1 : 0, 0, 0, 0));
    } else {
        ml.add(encodeMove(from, to, 0, cap ? 1 : 0, 0, 0, 0));
    }
}

void generateMoves(const Board &b, MoveList &ml) {
    ml.clear();
    int col = b.side, opp = col ^ 1;
    U64 enemies = b.occupancy[opp];
    U64 empty = ~b.occupancy[BOTH];
    
    // Pawns
    U64 pawns = b.pieces[col][PAWN];
    while(pawns) {
        int sq = popLSBIdx(pawns), rank = sq / 8, file = sq % 8;
        int dir = (col == WHITE) ? 8 : -8;
        int ps = sq + dir;
        
        if(ps >= 0 && ps < 64 && (empty & (1ULL << ps))) {
            addPawnMoves(b, ml, sq, ps, false);
            int sr = (col == WHITE) ? 1 : 6;
            if(rank == sr) {
                int dp = ps + dir;
                if(dp >= 0 && dp < 64 && (empty & (1ULL << dp)))
                    ml.add(encodeMove(sq, dp, 0, 0, 1, 0, 0));
            }
        }
        
        if(col == WHITE) {
            if(file > 0) {
                int c2 = sq + 7;
                if(b.pieceOn[c2] != NO_PIECE && b.colorOn[c2] == opp)
                    addPawnMoves(b, ml, sq, c2, true);
            }
            if(file < 7) {
                int c2 = sq + 9;
                if(b.pieceOn[c2] != NO_PIECE && b.colorOn[c2] == opp)
                    addPawnMoves(b, ml, sq, c2, true);
            }
        } else {
            if(file > 0) {
                int c2 = sq - 9;
                if(c2 >= 0 && b.pieceOn[c2] != NO_PIECE && b.colorOn[c2] == opp)
                    addPawnMoves(b, ml, sq, c2, true);
            }
            if(file < 7) {
                int c2 = sq - 7;
                if(c2 >= 0 && b.pieceOn[c2] != NO_PIECE && b.colorOn[c2] == opp)
                    addPawnMoves(b, ml, sq, c2, true);
            }
        }
        
        if(b.enPassant != NO_SQ) {
            int epf = b.enPassant % 8, epr = (col == WHITE) ? 4 : 3;
            if(rank == epr && abs(epf - file) == 1)
                ml.add(encodeMove(sq, b.enPassant, 0, 1, 0, 1, 0));
        }
    }
    
    // Knights
    U64 knights = b.pieces[col][KNIGHT];
    while(knights) {
        int sq = popLSBIdx(knights);
        U64 atk = knightAttacks[sq] & ~b.occupancy[col];
        while(atk) {
            int to = popLSBIdx(atk);
            ml.add(encodeMove(sq, to, 0, getBit(enemies, to) ? 1 : 0, 0, 0, 0));
        }
    }
    
    // Bishops
    U64 bishops = b.pieces[col][BISHOP];
    while(bishops) {
        int sq = popLSBIdx(bishops);
        U64 atk = getBishopAttacks(sq, b.occupancy[BOTH]) & ~b.occupancy[col];
        while(atk) {
            int to = popLSBIdx(atk);
            ml.add(encodeMove(sq, to, 0, getBit(enemies, to) ? 1 : 0, 0, 0, 0));
        }
    }
    
    // Rooks
    U64 rooks = b.pieces[col][ROOK];
    while(rooks) {
        int sq = popLSBIdx(rooks);
        U64 atk = getRookAttacks(sq, b.occupancy[BOTH]) & ~b.occupancy[col];
        while(atk) {
            int to = popLSBIdx(atk);
            ml.add(encodeMove(sq, to, 0, getBit(enemies, to) ? 1 : 0, 0, 0, 0));
        }
    }
    
    // Queens
    U64 queens = b.pieces[col][QUEEN];
    while(queens) {
        int sq = popLSBIdx(queens);
        U64 atk = getQueenAttacks(sq, b.occupancy[BOTH]) & ~b.occupancy[col];
        while(atk) {
            int to = popLSBIdx(atk);
            ml.add(encodeMove(sq, to, 0, getBit(enemies, to) ? 1 : 0, 0, 0, 0));
        }
    }
    
    // King
    if(b.pieces[col][KING]) {
        int sq = lsb(b.pieces[col][KING]);
        U64 atk = kingAttacks[sq] & ~b.occupancy[col];
        while(atk) {
            int to = popLSBIdx(atk);
            ml.add(encodeMove(sq, to, 0, getBit(enemies, to) ? 1 : 0, 0, 0, 0));
        }
        
        // Castling
        if(col == WHITE) {
            if((b.castling & WK) && !getBit(b.occupancy[BOTH], F1) && !getBit(b.occupancy[BOTH], G1) &&
               !b.isSquareAttacked(E1, BLACK) && !b.isSquareAttacked(F1, BLACK) && !b.isSquareAttacked(G1, BLACK))
                ml.add(encodeMove(E1, G1, 0, 0, 0, 0, 1));
            if((b.castling & WQ) && !getBit(b.occupancy[BOTH], D1) && !getBit(b.occupancy[BOTH], C1) && !getBit(b.occupancy[BOTH], B1) &&
               !b.isSquareAttacked(E1, BLACK) && !b.isSquareAttacked(D1, BLACK) && !b.isSquareAttacked(C1, BLACK))
                ml.add(encodeMove(E1, C1, 0, 0, 0, 0, 1));
        } else {
            if((b.castling & BK) && !getBit(b.occupancy[BOTH], F8) && !getBit(b.occupancy[BOTH], G8) &&
               !b.isSquareAttacked(E8, WHITE) && !b.isSquareAttacked(F8, WHITE) && !b.isSquareAttacked(G8, WHITE))
                ml.add(encodeMove(E8, G8, 0, 0, 0, 0, 1));
            if((b.castling & BQ) && !getBit(b.occupancy[BOTH], D8) && !getBit(b.occupancy[BOTH], C8) && !getBit(b.occupancy[BOTH], B8) &&
               !b.isSquareAttacked(E8, WHITE) && !b.isSquareAttacked(D8, WHITE) && !b.isSquareAttacked(C8, WHITE))
                ml.add(encodeMove(E8, C8, 0, 0, 0, 0, 1));
        }
    }
}

// ============================================================
//  EVALUATION CONSTANTS
// ============================================================

static const int MATERIAL[6] = {100, 320, 330, 500, 900, 20000};
static const int SEE_VAL[7] = {100, 320, 330, 500, 900, 20000, 0};

static const int PST_PAWN[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
};

static const int PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

static const int PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

static const int PST_ROOK[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
};

static const int PST_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

static const int PST_KING_MG[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

static const int PST_KING_EG[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
};

static const int* PST[6] = {PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING_MG};

static const int KNIGHT_MOB[9] = {-62,-40,-20, 0, 12, 22, 30, 36, 40};
static const int BISHOP_MOB[14] = {-48,-28,-10, 4, 15, 24, 32, 38, 43, 47, 50, 52, 54, 55};
static const int ROOK_MOB[15] = {-60,-40,-20, -4, 8, 18, 26, 33, 39, 44, 48, 51, 53, 54, 55};
static const int QUEEN_MOB[28] = {-30,-20,-10, -2, 4, 9, 14, 18, 22, 25, 28, 30, 32, 34,
                                    36, 37, 38, 39, 40, 41, 41, 42, 42, 42, 42, 42, 42, 42};

static const int PASSED_BONUS_MG[8] = {0, 5, 10, 20, 35, 55, 80, 0};
static const int PASSED_BONUS_EG[8] = {0, 10, 20, 35, 60, 95, 140, 0};

// ============================================================
//  TACTICAL EVALUATION — ELITE v14.0
// ============================================================

// Helper: squares between two aligned squares
static U64 betweenMask(int sq1, int sq2) {
    int f1 = sq1 % 8, r1 = sq1 / 8;
    int f2 = sq2 % 8, r2 = sq2 / 8;
    U64 mask = 0;
    
    if(f1 == f2) {
        int step = (r2 > r1) ? 1 : -1;
        for(int r = r1 + step; r != r2; r += step)
            setBit(mask, r * 8 + f1);
    }
    else if(r1 == r2) {
        int step = (f2 > f1) ? 1 : -1;
        for(int f = f1 + step; f != f2; f += step)
            setBit(mask, r1 * 8 + f);
    }
    else if(abs(f2 - f1) == abs(r2 - r1)) {
        int fStep = (f2 > f1) ? 1 : -1;
        int rStep = (r2 > r1) ? 1 : -1;
        int f = f1 + fStep, r = r1 + rStep;
        while(f != f2) {
            setBit(mask, r * 8 + f);
            f += fStep; r += rStep;
        }
    }
    return mask;
}

// All attackers on a square (both colors)
static U64 allAttackers(const Board &b, int sq, U64 occ) {
    return (pawnAttacks[WHITE][sq] & b.pieces[BLACK][PAWN]) |
           (pawnAttacks[BLACK][sq] & b.pieces[WHITE][PAWN]) |
           (knightAttacks[sq] & (b.pieces[WHITE][KNIGHT] | b.pieces[BLACK][KNIGHT])) |
           (getBishopAttacks(sq, occ) & (b.pieces[WHITE][BISHOP] | b.pieces[BLACK][BISHOP] |
                                         b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN])) |
           (getRookAttacks(sq, occ) & (b.pieces[WHITE][ROOK] | b.pieces[BLACK][ROOK] |
                                       b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN])) |
           (kingAttacks[sq] & (b.pieces[WHITE][KING] | b.pieces[BLACK][KING]));
}

// 1. FORK DETECTION
// Piece attacks 2+ non-pawn enemy pieces (or king + piece)
static int evalForks(const Board &b) {
    int score = 0;
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        
        // Knight forks
        U64 knights = b.pieces[col][KNIGHT];
        while(knights) {
            int sq = popLSBIdx(knights);
            U64 atk = knightAttacks[sq];
            int targets = 0;
            int targetValue = 0;
            
            for(int pt = ROOK; pt <= QUEEN; pt++) {
                U64 hits = atk & b.pieces[opp][pt];
                if(hits) {
                    targets += popcount(hits);
                    targetValue += MATERIAL[pt] * popcount(hits);
                }
            }
            // King + piece fork
            if(atk & b.pieces[opp][KING]) {
                targets += 2;
                targetValue += 400;
            }
            
            if(targets >= 2) {
                bool defended = b.isSquareAttacked(sq, opp);
                int bonus = (targetValue - MATERIAL[KNIGHT]) / 3;
                if(!defended) bonus = bonus * 3 / 2;
                score += sign * std::min(bonus, 200);
            }
        }
        
        // Bishop/Queen diagonal forks
        U64 bishops = b.pieces[col][BISHOP];
        while(bishops) {
            int sq = popLSBIdx(bishops);
            U64 atk = getBishopAttacks(sq, b.occupancy[BOTH]);
            int targets = 0;
            for(int pt = ROOK; pt <= KING; pt++) {
                if(atk & b.pieces[opp][pt]) targets++;
            }
            if(targets >= 2) score += sign * 60;
        }
    }
    return score;
}

// 2. PIN & SKEWER DETECTION
// Sliding piece attacks through enemy piece to king/queen/rook
static int evalPins(const Board &b) {
    int score = 0;
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        int oppKing = b.pieces[opp][KING] ? lsb(b.pieces[opp][KING]) : -1;
        if(oppKing < 0) continue;
        
        // Diagonal pins (Bishop/Queen)
        U64 diagAttackers = b.pieces[col][BISHOP] | b.pieces[col][QUEEN];
        while(diagAttackers) {
            int sq = popLSBIdx(diagAttackers);
            U64 ray = getBishopAttacks(sq, b.occupancy[BOTH]);
            if(ray & (1ULL << oppKing)) {
                U64 between = betweenMask(sq, oppKing) & b.occupancy[opp];
                if(popcount(between) == 1) {
                    int pinnedSq = lsb(between);
                    int pinnedPt = b.pieceOn[pinnedSq];
                    if(pinnedPt >= 0 && pinnedPt < 6) {
                        // More valuable pinned piece = bigger bonus
                        int bonus = MATERIAL[pinnedPt] / 4;
                        // If pinned to queen, even bigger
                        if(pinnedPt == QUEEN) bonus += 50;
                        score += sign * bonus;
                    }
                }
            }
        }
        
        // Rank/file pins (Rook/Queen)
        U64 lineAttackers = b.pieces[col][ROOK] | b.pieces[col][QUEEN];
        while(lineAttackers) {
            int sq = popLSBIdx(lineAttackers);
            U64 ray = getRookAttacks(sq, b.occupancy[BOTH]);
            if(ray & (1ULL << oppKing)) {
                U64 between = betweenMask(sq, oppKing) & b.occupancy[opp];
                if(popcount(between) == 1) {
                    int pinnedSq = lsb(between);
                    int pinnedPt = b.pieceOn[pinnedSq];
                    if(pinnedPt >= 0 && pinnedPt < 6) {
                        int bonus = MATERIAL[pinnedPt] / 3;
                        if(pinnedPt == QUEEN) bonus += 75;
                        score += sign * bonus;
                    }
                }
            }
        }
    }
    return score;
}

// 3. DISCOVERED ATTACK THREATS
// Moving a piece reveals attack on enemy king/queen
static int evalDiscovered(const Board &b) {
    int score = 0;
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        int oppKing = b.pieces[opp][KING] ? lsb(b.pieces[opp][KING]) : -1;
        int oppQueen = b.pieces[opp][QUEEN] ? lsb(b.pieces[opp][QUEEN]) : -1;
        
        U64 sliders = b.pieces[col][BISHOP] | b.pieces[col][ROOK] | b.pieces[col][QUEEN];
        
        // For each of our non-slider pieces, check if moving it reveals attack
        U64 blockers = b.occupancy[col] & ~(b.pieces[col][PAWN] | b.pieces[col][KING]);
        
        while(blockers) {
            int sq = popLSBIdx(blockers);
            int pt = b.pieceOn[sq];
            if(pt == BISHOP || pt == ROOK || pt == QUEEN) continue;
            
            U64 occWithout = b.occupancy[BOTH] & ~(1ULL << sq);
            
            // Check all our sliders
            U64 s = sliders;
            while(s) {
                int sliderSq = popLSBIdx(s);
                int sliderPt = b.pieceOn[sliderSq];
                
                U64 oldAtk = (sliderPt == BISHOP || sliderPt == QUEEN) ? 
                             getBishopAttacks(sliderSq, b.occupancy[BOTH]) :
                             getRookAttacks(sliderSq, b.occupancy[BOTH]);
                U64 newAtk = (sliderPt == BISHOP || sliderPt == QUEEN) ? 
                             getBishopAttacks(sliderSq, occWithout) :
                             getRookAttacks(sliderSq, occWithout);
                
                U64 revealed = newAtk & ~oldAtk;
                
                // Revealed attack on king = massive bonus
                if(oppKing >= 0 && (revealed & (1ULL << oppKing))) {
                    score += sign * 150;
                }
                // Revealed attack on queen
                if(oppQueen >= 0 && (revealed & (1ULL << oppQueen))) {
                    score += sign * 80;
                }
            }
        }
    }
    return score;
}

// 4. HANGING PIECES (Undefended attacked pieces)
static int evalHanging(const Board &b) {
    int score = 0;
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        
        for(int pt = KNIGHT; pt <= QUEEN; pt++) {
            U64 bb = b.pieces[col][pt];
            while(bb) {
                int sq = popLSBIdx(bb);
                U64 atk = allAttackers(b, sq, b.occupancy[BOTH]);
                U64 oppAtk = atk & b.occupancy[opp];
                U64 myDef = atk & b.occupancy[col];
                
                if(oppAtk && !myDef) {
                    // Completely hanging
                    int weakestAttacker = KING;
                    U64 a = oppAtk;
                    while(a) {
                        int asq = popLSBIdx(a);
                        int apt = b.pieceOn[asq];
                        if(apt < weakestAttacker) weakestAttacker = apt;
                    }
                    if(weakestAttacker < pt) {
                        score -= sign * MATERIAL[pt] / 2;
                    }
                }
            }
        }
    }
    return score;
}

// 5. TRAPPED PIECES (Limited mobility, especially minors)
static int evalTrapped(const Board &b) {
    int score = 0;
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        
        // Trapped knights (corners with low mobility)
        U64 knights = b.pieces[col][KNIGHT];
        while(knights) {
            int sq = popLSBIdx(knights);
            int mob = popcount(knightAttacks[sq] & ~b.occupancy[col]);
            int edgePenalty = 0;
            int f = sq % 8, r = sq / 8;
            if(f == 0 || f == 7) edgePenalty += 2;
            if(r == 0 || r == 7) edgePenalty += 2;
            
            if(mob <= 2 && edgePenalty >= 3) {
                score -= sign * (50 + (3 - mob) * 15);
            }
        }
        
        // Trapped bishops (blocked by own pawns on edges)
        U64 bishops = b.pieces[col][BISHOP];
        while(bishops) {
            int sq = popLSBIdx(bishops);
            int mob = popcount(getBishopAttacks(sq, b.occupancy[BOTH]) & ~b.occupancy[col]);
            int f = sq % 8, r = sq / 8;
            
            // Bishop on edge with 1-2 mobility = probably bad
            if((f == 0 || f == 7) && mob <= 2) {
                score -= sign * 30;
            }
            // Bishop trapped behind own pawns (TOMATO BISHOP)
            if(mob <= 3) {
                U64 ownPawns = b.pieces[col][PAWN];
                int blocked = 0;
                U64 atk = getBishopAttacks(sq, b.occupancy[BOTH]);
                while(atk) {
                    int asq = popLSBIdx(atk);
                    if(getBit(ownPawns, asq)) blocked++;
                }
                if(blocked >= 2) score -= sign * 25;
            }
        }
    }
    return score;
}

// 6. X-RAY ATTACKS (Attack through own piece)
static int evalXrays(const Board &b) {
    int score = 0;
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        
        // Queen x-rays through rook/bishop to enemy king
        U64 queens = b.pieces[col][QUEEN];
        while(queens) {
            int sq = popLSBIdx(queens);
            int oppKing = b.pieces[opp][KING] ? lsb(b.pieces[opp][KING]) : -1;
            if(oppKing < 0) continue;
            
            // Check if queen aligns with king with one piece between
            U64 occ = b.occupancy[BOTH];
            U64 diag = getBishopAttacks(sq, occ);
            U64 line = getRookAttacks(sq, occ);
            
            // Remove queen's own attacks, see what x-rays
            U64 occWithout = occ & ~b.pieces[col][QUEEN];
            U64 xrayDiag = getBishopAttacks(sq, occWithout) & ~diag;
            U64 xrayLine = getRookAttacks(sq, occWithout) & ~line;
            
            if((xrayDiag | xrayLine) & (1ULL << oppKing)) {
                score += sign * 40;  // X-ray on king
            }
        }
    }
    return score;
}

// 7. TACTICAL MOTIFS COMBINATION
// Piece attacked by lower value piece = tactical vulnerability
static int evalTacticalVulnerability(const Board &b) {
    int score = 0;
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        
        for(int pt = BISHOP; pt <= QUEEN; pt++) {
            U64 bb = b.pieces[col][pt];
            while(bb) {
                int sq = popLSBIdx(bb);
                U64 atk = allAttackers(b, sq, b.occupancy[BOTH]);
                U64 oppAtk = atk & b.occupancy[opp];
                
                if(!oppAtk) continue;
                
                // Find weakest attacker
                int weakest = KING;
                U64 a = oppAtk;
                while(a) {
                    int asq = popLSBIdx(a);
                    int apt = b.pieceOn[asq];
                    if(apt < weakest) weakest = apt;
                }
                
                // Attacked by lower value = bad
                if(weakest < pt) {
                    // Check if defended
                    U64 myDef = atk & b.occupancy[col];
                    if(!myDef) {
                        score -= sign * (MATERIAL[pt] - MATERIAL[weakest]) / 2;
                    }
                }
            }
        }
    }
    return score;
}

// MASTER TACTICAL EVALUATOR
static int evalTactical(const Board &b) {
    return evalForks(b) 
         + evalPins(b) 
         + evalDiscovered(b) 
         + evalHanging(b) 
         + evalTrapped(b) 
         + evalXrays(b)
         + evalTacticalVulnerability(b);
}

// ============================================================
//  MAIN EVALUATION
// ============================================================

static inline int tapered(int mg, int eg, int phase) {
    return (mg * phase + eg * (24 - phase)) / 24;
}

static inline int chebyshevDist(int a, int b) {
    return std::max(abs(a % 8 - b % 8), abs(a / 8 - b / 8));
}

int evaluate(const Board &b) {
    int score = 0;
    
    // Phase detection
    static const int PHASE_WEIGHT[6] = {0, 1, 1, 2, 4, 0};
    int phase = 0;
    for(int c = 0; c < 2; c++)
        for(int p = KNIGHT; p <= QUEEN; p++)
            phase += PHASE_WEIGHT[p] * popcount(b.pieces[c][p]);
    phase = std::min(phase, 24);
    
    bool isEndgame = (phase <= 10);
    bool isPawnEndgame = (phase <= 2);
    
    U64 allOcc = b.occupancy[BOTH];
    U64 center4 = (1ULL << D4) | (1ULL << E4) | (1ULL << D5) | (1ULL << E5);
    
    // Pawn structure
    int pawnMg = 0, pawnEg = 0;
    // Simple pawn eval (no hash for now - add later if needed)
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        U64 myPawns = b.pieces[col][PAWN];
        U64 oppPawns = b.pieces[opp][PAWN];
        
        U64 tmp = myPawns;
        while(tmp) {
            int sq = popLSBIdx(tmp);
            int rank = sq / 8, file = sq % 8;
            
            // Doubled pawns
            U64 fileMask = 0;
            for(int r = 0; r < 8; r++) setBit(fileMask, r * 8 + file);
            if(popcount(myPawns & fileMask) > 1) {
                pawnMg -= sign * 15;
                pawnEg -= sign * 10;
            }
            
            // Isolated pawns
            bool isolated = true;
            if(file > 0) {
                U64 leftFile = 0;
                for(int r = 0; r < 8; r++) setBit(leftFile, r * 8 + file - 1);
                if(myPawns & leftFile) isolated = false;
            }
            if(file < 7) {
                U64 rightFile = 0;
                for(int r = 0; r < 8; r++) setBit(rightFile, r * 8 + file + 1);
                if(myPawns & rightFile) isolated = false;
            }
            if(isolated) {
                pawnMg -= sign * 20;
                pawnEg -= sign * 15;
            }
            
            // Passed pawns
            bool passed = true;
            for(int af = std::max(0, file - 1); af <= std::min(7, file + 1); af++) {
                U64 ahead = 0;
                if(col == WHITE) {
                    for(int r = rank + 1; r < 8; r++) setBit(ahead, r * 8 + af);
                } else {
                    for(int r = 0; r < rank; r++) setBit(ahead, r * 8 + af);
                }
                if(oppPawns & ahead) { passed = false; break; }
            }
            if(passed) {
                int prank = (col == WHITE) ? rank : (7 - rank);
                pawnMg += sign * PASSED_BONUS_MG[prank];
                pawnEg += sign * PASSED_BONUS_EG[prank];
            }
        }
    }
    score += tapered(pawnMg, pawnEg, phase);
    
    // Main piece loop
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp = col ^ 1;
        
        int kingSq = b.pieces[col][KING] ? lsb(b.pieces[col][KING]) : 0;
        int oppKingSq = b.pieces[opp][KING] ? lsb(b.pieces[opp][KING]) : 0;
        
        // Material + PST
        for(int pt = 0; pt < 6; pt++) {
            U64 bb = b.pieces[col][pt];
            while(bb) {
                int sq = popLSBIdx(bb);
                int pstSq = (col == WHITE) ? sq : (sq ^ 56);
                int pstVal;
                if(pt == KING) {
                    int mg = PST_KING_MG[pstSq];
                    int eg = PST_KING_EG[pstSq];
                    pstVal = tapered(mg, eg, phase);
                } else {
                    pstVal = PST[pt][pstSq];
                }
                score += sign * (MATERIAL[pt] + pstVal);
            }
        }
        
        // Knight mobility + outposts
        U64 knights = b.pieces[col][KNIGHT];
        while(knights) {
            int sq = popLSBIdx(knights);
            U64 atk = knightAttacks[sq] & ~b.occupancy[col];
            int mob = popcount(atk);
            score += sign * tapered(KNIGHT_MOB[std::min(mob, 8)], KNIGHT_MOB[std::min(mob, 8)] * 7 / 10, phase);
            
            bool supported = (pawnAttacks[opp][sq] & b.pieces[col][PAWN]) != 0;
            bool attackedByPawn = (pawnAttacks[col][sq] & b.pieces[opp][PAWN]) != 0;
            bool inEnemyHalf = (col == WHITE) ? (sq / 8 >= 3) : (sq / 8 <= 4);
            if(inEnemyHalf && supported && !attackedByPawn)
                score += sign * tapered(18, 10, phase);
        }
        
        // Bishop mobility + pair
        U64 bishops = b.pieces[col][BISHOP];
        int bishopCount = popcount(bishops);
        if(bishopCount >= 2) score += sign * 40;
        
        while(bishops) {
            int sq = popLSBIdx(bishops);
            U64 atk = getBishopAttacks(sq, allOcc) & ~b.occupancy[col];
            int mob = popcount(atk);
            score += sign * tapered(BISHOP_MOB[std::min(mob, 13)], BISHOP_MOB[std::min(mob, 13)] * 9 / 10, phase);
            if(atk & center4) score += sign * 8;
        }
        
        // Rook mobility + open files
        U64 rooks = b.pieces[col][ROOK];
        while(rooks) {
            int sq = popLSBIdx(rooks);
            int file = sq % 8;
            U64 atk = getRookAttacks(sq, allOcc) & ~b.occupancy[col];
            int mob = popcount(atk);
            score += sign * tapered(ROOK_MOB[std::min(mob, 14)], ROOK_MOB[std::min(mob, 14)] * 12 / 10, phase);
            
            U64 fileMask = 0;
            for(int r = 0; r < 8; r++) setBit(fileMask, r * 8 + file);
            bool noOwnPawn = !(b.pieces[col][PAWN] & fileMask);
            bool noOppPawn = !(b.pieces[opp][PAWN] & fileMask);
            if(noOwnPawn && noOppPawn) score += sign * tapered(30, 20, phase);
            else if(noOwnPawn) score += sign * tapered(15, 8, phase);
            
            int rank7 = (col == WHITE) ? 6 : 1;
            if(sq / 8 == rank7) score += sign * tapered(20, 30, phase);
        }
        
        // Queen mobility
        U64 queens = b.pieces[col][QUEEN];
        while(queens) {
            int sq = popLSBIdx(queens);
            U64 atk = getQueenAttacks(sq, allOcc) & ~b.occupancy[col];
            int mob = popcount(atk);
            score += sign * tapered(QUEEN_MOB[std::min(mob, 27)], QUEEN_MOB[std::min(mob, 27)] * 7 / 10, phase);
            int qDist = chebyshevDist(sq, oppKingSq);
            score += sign * std::max(0, 6 - qDist) * 2;
        }
        
        // Basic king safety
        U64 kingZone = kingAttacks[oppKingSq] | (1ULL << oppKingSq);
        int attackScore = 0, attackers = 0;
        for(int pt = KNIGHT; pt <= QUEEN; pt++) {
            U64 bb = b.pieces[col][pt];
            while(bb) {
                int sq = popLSBIdx(bb);
                U64 atk = (pt == KNIGHT) ? knightAttacks[sq] :
                          (pt == BISHOP) ? getBishopAttacks(sq, allOcc) :
                          (pt == ROOK) ? getRookAttacks(sq, allOcc) :
                          getQueenAttacks(sq, allOcc);
                U64 hit = atk & kingZone;
                if(hit) {
                    attackers++;
                    attackScore += (pt == KNIGHT || pt == BISHOP) ? 20 : (pt == ROOK) ? 40 : 80;
                }
            }
        }
        if(attackers >= 2) {
            int danger = attackScore * attackers / 4;
            score += sign * danger;
        }
    }
    
    // Endgame mop-up
    if(isEndgame) {
        int materialScore = 0;
        for(int p = PAWN; p <= QUEEN; p++)
            materialScore += MATERIAL[p] * (popcount(b.pieces[WHITE][p]) - popcount(b.pieces[BLACK][p]));
        
        if(abs(materialScore) > 200) {
            int winningSide = (materialScore > 0) ? WHITE : BLACK;
            int winSign = (winningSide == WHITE) ? 1 : -1;
            int wKing = b.pieces[WHITE][KING] ? lsb(b.pieces[WHITE][KING]) : 0;
            int bKing = b.pieces[BLACK][KING] ? lsb(b.pieces[BLACK][KING]) : 0;
            int loserKing = (winningSide == WHITE) ? bKing : wKing;
            int winnerKing = (winningSide == WHITE) ? wKing : bKing;
            
            int lf = loserKing % 8, lr = loserKing / 8;
            int cornerPush = std::max(3 - lf, lf - 4) + std::max(3 - lr, lr - 4);
            score += winSign * cornerPush * 10;
            
            int dist = chebyshevDist(winnerKing, loserKing);
            score += winSign * (7 - dist) * 8;
        }
        
        if(isPawnEndgame) {
            score += (b.side == WHITE) ? 15 : -15;
        }
    }
    
    // 🔥 TACTICAL EVALUATION (THE MAIN UPGRADE)
    if(!isEndgame) {
        score += evalTactical(b);
    }
    
    return (b.side == WHITE) ? score : -score;
}

// ============================================================
//  SEARCH
// ============================================================

static const int INF = 1000000;
static const int MATE = 900000;
static const int MAX_PLY = 64;

static Move killers[MAX_PLY][2];
static int history[2][64][64];
static int lmrTable[64][64];
static bool lmrInited = false;

static void initLMR() {
    if(lmrInited) return;
    lmrInited = true;
    for(int d = 0; d < 64; d++) {
        for(int mc = 0; mc < 64; mc++) {
            if(d < 2 || mc < 2) { lmrTable[d][mc] = 0; continue; }
            int r = (int)(std::log((double)d) * std::log((double)mc) / 2.0);
            lmrTable[d][mc] = std::max(0, std::min(4, r));
        }
    }
}

static void clearTables() {
    memset(killers, 0, sizeof(killers));
    memset(history, 0, sizeof(history));
}

static inline void checkTime(SearchInfo &info) {
    if(info.timeLimit <= 0) return;

    if((info.nodes & 155) == 0) {
        auto e = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - info.startTime).count();

            if(e >= info.timeLimit)
            info.stop = true;
    }

    }

// SEE - Simplified but correct
static bool seeGe(const Board &b, Move m, int threshold) {
    int from = moveFrom(m), to = moveTo(m);
    int captured = moveEP(m) ? PAWN : b.pieceOn[to];
    if(captured == NO_PIECE) return threshold <= 0;
    
    int gain = SEE_VAL[captured] - threshold;
    if(gain < 0) return false;
    
    int attacker = b.pieceOn[from];
    gain -= SEE_VAL[attacker];
    if(gain >= 0) return true;
    
    U64 occ = b.occupancy[BOTH];
    clearBit(occ, from);
    
    U64 attackers = allAttackers(b, to, occ);
    int side = b.side ^ 1;
    
    while(true) {
        side ^= 1;
        attackers &= occ;
        U64 sideAtk = attackers & b.occupancy[side];
        if(!sideAtk) break;
        
        int pt;
        U64 cand;
        for(pt = PAWN; pt <= KING; pt++) {
            cand = sideAtk & b.pieces[side][pt];
            if(cand) break;
        }
        if(pt > KING) break;
        
        int sq = lsb(cand);
        clearBit(occ, sq);
        
        if(pt == PAWN || pt == BISHOP || pt == QUEEN)
            attackers |= getBishopAttacks(to, occ) & (b.pieces[WHITE][BISHOP] | b.pieces[BLACK][BISHOP] |
                                                      b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN]);
        if(pt == ROOK || pt == QUEEN)
            attackers |= getRookAttacks(to, occ) & (b.pieces[WHITE][ROOK] | b.pieces[BLACK][ROOK] |
                                                    b.pieces[WHITE][QUEEN] | b.pieces[BLACK][QUEEN]);
        
        gain = -gain - 1 - SEE_VAL[pt];
        if(gain >= 0) {
            if(pt == KING && (attackers & b.occupancy[side ^ 1]))
                side ^= 1;
            break;
        }
    }
    return b.side != side;
}

static int scoreMove(const Board &b, Move m, int ply) {
    int s = 0;
    if(moveCapture(m)) {
        int att = b.pieceOn[moveFrom(m)];
        int vic = moveEP(m) ? PAWN : b.pieceOn[moveTo(m)];
        if(att != NO_PIECE && vic != NO_PIECE) {
            if(seeGe(b, m, 1)) s = 10000 + MATERIAL[vic] - MATERIAL[att];
            else if(seeGe(b, m, 0)) s = 9000 + MATERIAL[vic] - MATERIAL[att];
            else s = 2000 + MATERIAL[vic] - MATERIAL[att];
        }
    }
    if(movePromo(m)) s += 9500 + (movePromo(m) == QUEEN ? 900 : 300);
    if(!moveCapture(m) && !movePromo(m)) {
        if(ply >= 0 && ply < MAX_PLY) {
            if(m == killers[ply][0]) s += 8500;
            else if(m == killers[ply][1]) s += 7500;
        }
        s += history[b.side][moveFrom(m)][moveTo(m)];
    }
    return s;
}

static void sortMoves(const Board &b, MoveList &ml, Move pvMove, int ply) {
    int scores[256];
    for(int i = 0; i < ml.count; i++)
        scores[i] = (ml.moves[i] == pvMove) ? (INF + 2) : scoreMove(b, ml.moves[i], ply);
    
    for(int i = 0; i < ml.count - 1; i++) {
        int best = i;
        for(int j = i + 1; j < ml.count; j++)
            if(scores[j] > scores[best]) best = j;
        if(best != i) {
            std::swap(scores[i], scores[best]);
            std::swap(ml.moves[i], ml.moves[best]);
        }
    }
}

// Transposition Table
enum TTFlag { TT_EXACT = 0, TT_ALPHA = 1, TT_BETA = 2 };
struct TTEntry {
    U64 key;
    int score;
    Move bestMove;
    int16_t depth;
    int8_t flag;
};

static const int TT_SIZE = 1 << 22;
static TTEntry TT[TT_SIZE];
static bool ttInited = false;

static void initTT() {
    if(ttInited) return;
    ttInited = true;
    memset(TT, 0, sizeof(TT));
}

static void ttStore(U64 key, int score, Move m, int depth, int flag) {
    int i = (int)(key & (TT_SIZE - 1));
    if(TT[i].key == 0 || depth >= TT[i].depth)
        TT[i] = {key, score, m, (int16_t)depth, (int8_t)flag};
}

static bool ttProbe(U64 key, int depth, int alpha, int beta, int &score, Move &bm) {
    int i = (int)(key & (TT_SIZE - 1));
    bm = 0;
    if(TT[i].key != key) return false;
    bm = TT[i].bestMove;
    if(TT[i].depth < depth) return false;
    score = TT[i].score;
    if(TT[i].flag == TT_EXACT) return true;
    if(TT[i].flag == TT_ALPHA && score <= alpha) { score = alpha; return true; }
    if(TT[i].flag == TT_BETA && score >= beta) { score = beta; return true; }
    return false;
}

// Eval cache
static const int EVAL_CACHE_SIZE = 1 << 20;
struct EvalCacheEntry { U64 key; int eval; };
static EvalCacheEntry evalCache[EVAL_CACHE_SIZE];

static inline int evaluateCached(const Board &b) {
    U64 key = hashBoard(b);
    EvalCacheEntry &e = evalCache[(int)(key & (EVAL_CACHE_SIZE - 1))];
    if(e.key == key) return e.eval;
    int v = evaluate(b);
    e = {key, v};
    return v;
}

int quiescence(Board &b, int alpha, int beta, int ply, SearchInfo &info) {
    info.nodes++;
    checkTime(info);
    if(info.stop) return alpha;
    
    int standPat = evaluateCached(b);
    if(standPat >= beta) return beta;
    if(standPat > alpha) alpha = standPat;
    if(ply >= MAX_PLY - 1) return alpha;
    
    MoveList ml;
    generateMoves(b, ml);
    bool inChk = b.inCheck();
    
    // Filter moves
    int scores[256], cnt = 0;
    for(int i = 0; i < ml.count; i++) {
        if(inChk || moveCapture(ml.moves[i])) {
            scores[cnt] = scoreMove(b, ml.moves[i], ply);
            ml.moves[cnt++] = ml.moves[i];
        }
    }
    ml.count = cnt;
    
    // Sort
    for(int i = 0; i < ml.count - 1; i++) {
        int best = i;
        for(int j = i + 1; j < ml.count; j++)
            if(scores[j] > scores[best]) best = j;
        if(best != i) {
            std::swap(scores[i], scores[best]);
            std::swap(ml.moves[i], ml.moves[best]);
        }
    }
    
    for(int i = 0; i < ml.count; i++) {
        // Delta pruning
        if(!inChk && moveCapture(ml.moves[i])) {
            int vic = b.pieceOn[moveTo(ml.moves[i])];
            if(vic != NO_PIECE && standPat + MATERIAL[vic] + 200 < alpha) continue;
        }
        
        UndoInfo u;
        if(!b.makeMove(ml.moves[i], u)) continue;
        int score = -quiescence(b, -beta, -alpha, ply + 1, info);
        b.unmakeMove(ml.moves[i], u);
        
        if(info.stop) return alpha;
        if(score >= beta) return beta;
        if(score > alpha) alpha = score;
    }
    return alpha;
}

static int staticEvalStack[MAX_PLY + 4];

int alphaBeta(Board &b, int alpha, int beta, int depth, int ply, SearchInfo &info, bool nullOk) {
    if(info.stop) return 0;
    info.nodes++;
    checkTime(info);
    if(info.stop) return 0;
    
    bool inChk = b.inCheck();
    if(inChk) depth++;
    if(depth <= 0) return quiescence(b, alpha, beta, ply, info);
    if(ply > 0 && b.halfMove >= 100) return 0;
    if(ply >= MAX_PLY - 1) return evaluateCached(b);
    
    // Mate distance pruning
    alpha = std::max(alpha, -(MATE - ply));
    beta = std::min(beta, MATE - ply - 1);
    if(alpha >= beta) return alpha;
    
    // TT probe
    U64 hash = hashBoard(b);
    Move ttMv = 0;
    int ttScore;
    if(ply > 0 && ttProbe(hash, depth, alpha, beta, ttScore, ttMv))
        return ttScore;
    
    // Static eval
    int rawEval = evaluateCached(b);
    staticEvalStack[ply] = rawEval;
    bool improving = (ply >= 2) && (rawEval > staticEvalStack[ply - 2]);
    
    // Razoring
    if(!inChk && depth <= 3 && rawEval < alpha - 300 - 200 * depth)
        return quiescence(b, alpha, beta, ply, info);
    
    // Futility pruning
    if(!inChk && depth < 8 && rawEval >= beta && rawEval - (80 - 20 * improving) * depth < MATE / 2 && beta > -MATE / 2)
        return rawEval;
    
    // Null move
    if(nullOk && !inChk && depth >= 3 && ply > 0 && rawEval >= beta) {
        int mat = 0;
        for(int p = 0; p < 5; p++) mat += MATERIAL[p] * popcount(b.pieces[b.side][p]);
        if(mat > MATERIAL[ROOK]) {
            UndoInfo nu;
            nu.save(b, 0, NO_PIECE);
            b.enPassant = NO_SQ;
            b.side ^= 1;
            if(b.side == WHITE) b.fullMove++;
            b.recalcOccupancy();
            
            int R = 3 + depth / 3 + std::min(3, (rawEval - beta) / 200);
            int ns = -alphaBeta(b, -beta, -beta + 1, depth - 1 - R, ply + 1, info, false);
            nu.restore(b);
            
            if(info.stop) return 0;
            if(ns >= beta && ns < MATE / 2) return beta;
        }
    }
    
    // IIR
    if(depth >= 4 && ttMv == 0) depth--;
    
    MoveList ml;
    generateMoves(b, ml);
    sortMoves(b, ml, ttMv, ply);
    
    bool hasLegal = false;
    int bestScore = -INF;
    Move bestMv = 0;
    int origAlpha = alpha;
    int mc = 0;
    
    for(int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        UndoInfo u;
        if(!b.makeMove(m, u)) continue;
        
        hasLegal = true;
        mc++;
        bool isCap = moveCapture(m) || moveEP(m);
        bool isPro = movePromo(m);
        bool givesCheck = b.inCheck();
        
        // SEE pruning
        if(!inChk && bestScore > -MATE / 2) {
            if(isCap && !isPro) {
                if(!seeGe(b, m, -90 * depth)) {
                    b.unmakeMove(m, u);
                    continue;
                }
            } else if(!isCap && !isPro && !givesCheck) {
                int lmrDepth = std::max(0, depth - 1 - (mc > 6 ? 2 : mc > 3 ? 1 : 0));
                if(rawEval + 60 + 120 * lmrDepth <= alpha && lmrDepth < 7) {
                    b.unmakeMove(m, u);
                    continue;
                }
                int maxMoves = 3 + depth * depth / (improving ? 1 : 2);
                if(mc > maxMoves && depth <= 6) {
                    b.unmakeMove(m, u);
                    continue;
                }
            }
        }
        
        int newDepth = depth - 1 + (givesCheck ? 1 : 0);
        int score;
        
        // LMR
        bool doLMR = newDepth >= 2 && mc > 2 && !isCap && !isPro && !inChk && ply > 0 && !givesCheck;
        
        if(mc == 1) {
            score = -alphaBeta(b, -beta, -alpha, newDepth, ply + 1, info, true);
        } else if(doLMR) {
            int R = lmrTable[std::min(63, std::max(0, newDepth))][std::min(63, std::max(0, mc))];
            if(!improving) R++;
            
            int reducedDepth = std::max(1, newDepth - R);
            score = -alphaBeta(b, -alpha - 1, -alpha, reducedDepth, ply + 1, info, true);
            
            if(score > alpha) {
                score = -alphaBeta(b, -beta, -alpha, newDepth, ply + 1, info, true);
            }
        } else {
            score = -alphaBeta(b, -alpha - 1, -alpha, newDepth, ply + 1, info, true);
            if(score > alpha && score < beta)
                score = -alphaBeta(b, -beta, -alpha, newDepth, ply + 1, info, true);
        }
        
        b.unmakeMove(m, u);
        if(info.stop) return 0;
        
        if(score > bestScore) { bestScore = score; bestMv = m; }
        if(score > alpha) {
            alpha = score;
            if(alpha >= beta) {
                if(!isCap && ply < MAX_PLY) {
                    killers[ply][1] = killers[ply][0];
                    killers[ply][0] = m;
                    int bonus = std::min(depth * depth, 400);
                    history[b.side ^ 1][moveFrom(m)][moveTo(m)] += bonus;
                    if(history[b.side ^ 1][moveFrom(m)][moveTo(m)] > 8000)
                        history[b.side ^ 1][moveFrom(m)][moveTo(m)] = 8000;
                    
                    for(int j = 0; j < i; j++) {
                        if(!moveCapture(ml.moves[j]) && !movePromo(ml.moves[j])) {
                            history[b.side ^ 1][moveFrom(ml.moves[j])][moveTo(ml.moves[j])] -= bonus / 4;
                        }
                    }
                }
                ttStore(hash, beta, m, depth, TT_BETA);
                return beta;
            }
        }
    }
    
    if(!hasLegal) return inChk ? -(MATE - ply) : 0;
    
    ttStore(hash, bestScore, bestMv, depth, (bestScore <= origAlpha) ? TT_ALPHA :
            (bestScore >= beta ? TT_BETA : TT_EXACT));
    return bestScore;
}

// ============================================================
//  OPENING BOOK
// ============================================================

struct BookLine {
    const char* moves[12];
};

static const BookLine BOOK[] = {
    {{"e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6",nullptr}},
    {{"e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6",nullptr}},
    {{"e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6",nullptr}},
    {{"e2e4","e7e6","d2d4","d7d5","b1c3","g8f6","c1g5",nullptr}},
    {{"d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","c1g5","f8e7",nullptr}},
    {{"d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","g1f3",nullptr}},
    {{"c2c4","e7e5","b1c3","g8f6","g1f3","b8c6","g2g3",nullptr}},
    {{"d2d4","d7d5","g1f3","g8f6","c1f4","e7e6","e2e3",nullptr}},
    {{"e2e4","c7c6","d2d4","d7d5","b1c3","g7g6",nullptr}},
    {{"d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","e2e3",nullptr}},
    {{"d2d4","d7d5","c2c4","c7c6","g1f3","g8f6","b1c3","e7e6",nullptr}},
    {{"e2e4","g7g6","d2d4","f8g7","b1c3","d7d6","g1f3",nullptr}},
};

static const int N_BOOK = 12;
static int g_bookLine = -1;

static Move getBookMove(const Board &b) {
    if(b.fullMove == 1 && b.side == WHITE) {
        g_bookLine = rand() % N_BOOK;
    }
    if(g_bookLine < 0) return 0;
    
    int idx = (b.fullMove - 1) * 2 + (b.side == BLACK ? 1 : 0);
    const char* mv = BOOK[g_bookLine].moves[idx];
    if(!mv) return 0;
    
    // Verify position
    Board tmp;
    tmp.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    for(int i = 0; i < idx; i++) {
        const char* bm = BOOK[g_bookLine].moves[i];
        if(!bm) return 0;
        Move m = 0;
        // Simple move parsing for book
        int from = (bm[0] - 'a') + (bm[1] - '1') * 8;
        int to = (bm[2] - 'a') + (bm[3] - '1') * 8;
        MoveList ml2; generateMoves(tmp, ml2);
        for(int j = 0; j < ml2.count; j++) {
            if(moveFrom(ml2.moves[j]) == from && moveTo(ml2.moves[j]) == to) {
                m = ml2.moves[j]; break;
            }
        }
        if(!m) { g_bookLine = -1; return 0; }
        UndoInfo u2; tmp.makeMove(m, u2);
    }
    
    for(int c = 0; c < 2; c++)
        for(int p = 0; p < 6; p++)
            if(tmp.pieces[c][p] != b.pieces[c][p]) { g_bookLine = -1; return 0; }
    if(tmp.side != b.side) { g_bookLine = -1; return 0; }
    
    int from = (mv[0] - 'a') + (mv[1] - '1') * 8;
    int to = (mv[2] - 'a') + (mv[3] - '1') * 8;
    MoveList ml2; generateMoves(b, ml2);
    for(int j = 0; j < ml2.count; j++) {
        if(moveFrom(ml2.moves[j]) == from && moveTo(ml2.moves[j]) == to) {
            UndoInfo u2; Board t = b;
            if(t.makeMove(ml2.moves[j], u2)) return ml2.moves[j];
        }
    }
    g_bookLine = -1;
    return 0;
}

// ============================================================
//  BEST MOVE / UCI
// ============================================================

Move bestMove(Board &b, int depth, int timeLimitMs) {
    // initZobrist();
    // initTT();
    // initLMR();
    // clearTables();
    // memset(evalCache, 0, sizeof(evalCache));

    SearchInfo info;
    info.depth = depth;
    info.nodes = 0;
    info.stop = false;
    info.timeLimit = timeLimitMs;
    info.startTime = std::chrono::steady_clock::now();

    // ───────────────── OPENING BOOK ─────────────────
    Move bookMv = getBookMove(b);
    if(bookMv) {
        std::string bm;
        bm += char('a' + moveFrom(bookMv) % 8);
        bm += char('1' + moveFrom(bookMv) / 8);
        bm += char('a' + moveTo(bookMv) % 8);
        bm += char('1' + moveTo(bookMv) / 8);

        if(movePromo(bookMv)) {
            const char pr[] = {'?', 'n', 'b', 'r', 'q'};
            bm += pr[movePromo(bookMv)];
        }

       std::cout << "info depth 0 score cp 20 nodes 1 time 0 pv "
                  << bm << "\n";
        std::cout << "bestmove " << bm << "\n";
        return bookMv;
    }

    // ───────────────── LEGAL MOVES ─────────────────
    MoveList ml;
    generateMoves(b, ml);

    MoveList legal;
    UndoInfo tu;

    for(int i = 0; i < ml.count; i++) {
        Board t = b;
        if(t.makeMove(ml.moves[i], tu))
            legal.add(ml.moves[i]);
    }

    if(legal.count == 0) {
        std::cout << "bestmove 0000\n";
        return 0;
    }

    Move best = legal.moves[0];
    int bestScore = -INF;

    // ───────────────── ITERATIVE DEEPENING ─────────────────
    for(int d = 1; d <= depth && !info.stop; d++) {
        sortMoves(b, legal, best, 0);

        int asp = INF;

        int alpha = (asp == INF || bestScore <= -MATE/2)
                    ? -INF : bestScore - asp;

        int beta  = (asp == INF || bestScore <= -MATE/2)
                    ? INF : bestScore + asp;

        int iterBest = -INF;
        Move iterMv = best;

        while(!info.stop) {

            int localAlpha = alpha;
            int localBeta  = beta;

            iterBest = -INF;

            for(int i = 0; i < legal.count && !info.stop; i++) {

                UndoInfo u;

                if(!b.makeMove(legal.moves[i], u))
                    continue;

                int score;

                if(i == 0) {
                    score = -alphaBeta(
                        b,
                        -localBeta,
                        -localAlpha,
                        d - 1,
                        1,
                        info,
                        true
                    );
                }
                else {
                    score = -alphaBeta(
                        b,
                        -localAlpha - 1,
                        -localAlpha,
                        d - 1,
                        1,
                        info,
                        true
                    );

                    if(score > localAlpha && score < localBeta) {
                        score = -alphaBeta(
                            b,
                            -localBeta,
                            -localAlpha,
                            d - 1,
                            1,
                            info,
                            true
                        );
                    }
                }

                b.unmakeMove(legal.moves[i], u);

                if(info.stop)
                    break;

                if(score > iterBest) {
                    iterBest = score;
                    iterMv = legal.moves[i];
                }

                if(score > localAlpha)
                    localAlpha = score;
            }

            if(asp == INF || d < 5 || info.stop)
                break;

            if(iterBest <= alpha) {
                asp = std::min(800, asp * 2);
                alpha = std::max(-INF, iterBest - asp);
                continue;
            }

            if(iterBest >= beta) {
                asp = std::min(800, asp * 2);
                beta = std::min(INF, iterBest + asp);
                continue;
            }

            break;
        }

        if(!info.stop) {

            best = iterMv;
            bestScore = iterBest;

            // Move ordering: best to front
            for(int i = 0; i < legal.count; i++) {
                if(legal.moves[i] == best) {
                    for(int j = i; j > 0; j--)
                        legal.moves[j] = legal.moves[j - 1];

                    legal.moves[0] = best;
                    break;
                }
            }

            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - info.startTime
                ).count();

            std::string pv;
            pv += char('a' + moveFrom(best) % 8);
            pv += char('1' + moveFrom(best) / 8);
            pv += char('a' + moveTo(best) % 8);
            pv += char('1' + moveTo(best) / 8);

            if(movePromo(best)) {
                const char pr[] = {'?', 'n', 'b', 'r', 'q'};
                pv += pr[movePromo(best)];
            }

            std::cout << "info depth " << d
                      << " score cp " << bestScore
                      << " nodes " << info.nodes
                      << " time " << elapsed
                      << " pv " << pv << "\n";

            // soft stop
            if(timeLimitMs > 0 &&
                depth >= 64 &&
               d >= 6 &&
               elapsed >= timeLimitMs * 60 / 100)
            {
                info.stop = true;
            }
        }
    }

    // ───────────────── FINAL BESTMOVE ─────────────────
    std::string bm;
    bm += char('a' + moveFrom(best) % 8);
    bm += char('1' + moveFrom(best) / 8);
    bm += char('a' + moveTo(best) % 8);
    bm += char('1' + moveTo(best) / 8);

    if(movePromo(best)) {
        const char pr[] = {'?', 'n', 'b', 'r', 'q'};
        bm += pr[movePromo(best)];
    }

    std::cout << "bestmove " << bm << "\n";
    return best;
}

// ============================================================
//  PERFT
// ============================================================

unsigned long long perft(Board &b, int depth) {
    if(depth == 0) return 1;
    MoveList ml;
    generateMoves(b, ml);
    unsigned long long n = 0;
    UndoInfo u;
    for(int i = 0; i < ml.count; i++) {
        if(!b.makeMove(ml.moves[i], u)) continue;
        n += perft(b, depth - 1);
        b.unmakeMove(ml.moves[i], u);
    }
    return n;
}

void divide(Board &b, int depth) {
    if(depth <= 0) { std::cout << "Depth must be > 0\n"; return; }
    MoveList ml;
    generateMoves(b, ml);
    std::cout << "Generated " << ml.count << " pseudo-legal moves\n";
    UndoInfo u;
    unsigned long long total = 0;
    int legal = 0;
    
    for(int i = 0; i < ml.count; i++) {
        if(!b.makeMove(ml.moves[i], u)) continue;
        legal++;
        unsigned long long n = perft(b, depth - 1);
        b.unmakeMove(ml.moves[i], u);
        
        std::string mvStr;
        mvStr += (char)('a' + moveFrom(ml.moves[i]) % 8);
        mvStr += (char)('1' + moveFrom(ml.moves[i]) / 8);
        mvStr += (char)('a' + moveTo(ml.moves[i]) % 8);
        mvStr += (char)('1' + moveTo(ml.moves[i]) / 8);
        
        std::cout << mvStr << ": " << n << "\n";
        total += n;
    }
    std::cout << "Legal moves: " << legal << "\nTotal: " << total << "\n";
}

// ============================================================
// moveToString
// ============================================================
std::string moveToStr(Move m) {
    std::string s = "0000";
    int from = moveFrom(m), to = moveTo(m);

    s[0] = 'a' + (from % 8);
    s[1] = '1' + (from / 8);
    s[2] = 'a' + (to % 8);
    s[3] = '1' + (to / 8);

    if(movePromo(m)) {
        char p='q';
        if(movePromo(m)==ROOK) p='r';
        else if(movePromo(m)==BISHOP) p='b';
        else if(movePromo(m)==KNIGHT) p='n';
        s += p;
    }
    return s;
}

// ============================================================
//  UCI LOOP
// ============================================================

void uciLoop() {
    initZobrist();
    initTT();
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);
    
    Board b;
    b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    std::string line, token;
    while(true) {
        if(!std::getline(std::cin, line)) break;
        if(!line.empty() && line.back() == '\r') line.pop_back();
        if(line.empty()) continue;
        
        std::istringstream ss(line);
        ss >> token;
        
        if(token == "uci") {
            std::cout << "id name Zero v14.0 TACTICAL\n"
                      << "id author Snoopy\n"
                      << "uciok\n";
        }
        else if(token == "isready") {
            std::cout << "readyok\n";
        }
        else if(token == "ucinewgame") {
            b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            clearTables();
            memset(TT, 0, sizeof(TT));
            memset(evalCache, 0, sizeof(evalCache));
            g_bookLine = -1;
        }
        else if(token == "position") {
            std::string pos;
            ss >> pos;
            if(pos == "startpos") {
                b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                ss >> token;
            }
            else if(pos == "fen") {
                std::string fen, p;
                ss >> p;
                fen = p;
                while(ss >> p && p != "moves") fen += " " + p;
                b.setFromFEN(fen);
                token = p;
            }
            std::string mv;
            while(ss >> mv) {
                int from = (mv[0] - 'a') + (mv[1] - '1') * 8;
                int to = (mv[2] - 'a') + (mv[3] - '1') * 8;
                MoveList ml2;
                generateMoves(b, ml2);
                for(int i = 0; i < ml2.count; i++) {
                    if(moveFrom(ml2.moves[i]) == from && moveTo(ml2.moves[i]) == to) {
                        UndoInfo u2;
                        if(b.makeMove(ml2.moves[i], u2)) break;
                    }
                }
            }
        }
        else if(token == "go") {
            int depth = 64, movetime = -1, wtime = -1, btime = -1, winc = 0, binc = 0;
            bool ds = false, ms = false;
            while(ss >> token) {
                if(token == "depth") { ss >> depth; ds = true; }
                else if(token == "movetime") { ss >> movetime; ms = true; }
                else if(token == "wtime") ss >> wtime;
                else if(token == "btime") ss >> btime;
                else if(token == "winc") ss >> winc;
                else if(token == "binc") ss >> binc;
                else if(token == "infinite") { depth = 64; ds = true; ms = false; movetime = -1; }
            }
            if(!ms && wtime > 0 && btime > 0) {
                int myTime = (b.side == WHITE) ? wtime : btime;
                int myInc = (b.side == WHITE) ? winc : binc;

                int usable = std::max(0, myTime - 30);
                movetime = usable / 15 + (myInc * 4) / 5;
                movetime = std::max(120, std::min(movetime, usable / 2));
            }
            // defaults
            if(!ds && !ms) depth = 64;

            //KEY FIX
            if(ds && !ms) movetime = 1000000000;
            else if (movetime < 0) movetime = 50000;
            bestMove(b, depth, movetime);
        }
        else if(token == "d") {
            b.print();
        }
        else if(token == "eval") {
            std::cout << "Evaluation: " << evaluate(b) << "\n";
        }
        else if(token == "perft") {
            int d;
            ss >> d;
            if(d > 0) {
                auto t1 = std::chrono::steady_clock::now();
                auto n = perft(b, d);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t1).count();
                std::cout << "perft " << d << " = " << n << " (" << ms << " ms)\n";
            }
        }
        else if(token == "divide") {
            int d;
            ss >> d;
            if(d > 0) divide(b, d);
        }
        else if(token == "quit" || token == "exit") {
            break;
        }
    }
}

