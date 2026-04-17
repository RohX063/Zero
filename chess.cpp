#include "chess.h"
#include <cstdlib>
#include <ctime>

// ============================================================
//  ATTACK TABLES
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

// Magic numbers
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
    0x0080001020400080ULL,0x0040001000200040ULL,0x0080081000200080ULL,0x0080040800100080ULL,
    0x0080020400080080ULL,0x0080010200040080ULL,0x0080008001000200ULL,0x0080002040800100ULL,
    0x0000800020400080ULL,0x0000400020005000ULL,0x0000801000200080ULL,0x0000800800100080ULL,
    0x0000800400080080ULL,0x0000800200040080ULL,0x0000800100020080ULL,0x0000800040800100ULL,
    0x0000208000400080ULL,0x0000404000201000ULL,0x0000808010002000ULL,0x0000808008001000ULL,
    0x0000808004000800ULL,0x0000808002000400ULL,0x0000010100020004ULL,0x0000020000408104ULL,
    0x0000208080004000ULL,0x0000200040005000ULL,0x0000100080200080ULL,0x0000080080100080ULL,
    0x0000040080080080ULL,0x0000020080040080ULL,0x0000010080800200ULL,0x0000800080004100ULL,
    0x0000204000800080ULL,0x0000200040401000ULL,0x0000100080802000ULL,0x0000080080801000ULL,
    0x0000040080800800ULL,0x0000020080800400ULL,0x0000020001010004ULL,0x0000800040800100ULL,
    0x0000204000808000ULL,0x0000200040008080ULL,0x0000100020008080ULL,0x0000080010008080ULL,
    0x0000040008008080ULL,0x0000020004008080ULL,0x0000010002008080ULL,0x0000004081020004ULL,
    0x0000204000800080ULL,0x0000200040008080ULL,0x0000100020008080ULL,0x0000080010008080ULL,
    0x0000040008008080ULL,0x0000020004008080ULL,0x0000800100020080ULL,0x0000800041000080ULL,
    0x00FFFCDDFCED714AULL,0x007FFCDDFCED714AULL,0x003FFFCDFFD88096ULL,0x0000040810002101ULL,
    0x0000007F37F01401ULL,0x0000007F37F01401ULL,0x00000036316C0003ULL,0x00000036316C0003ULL
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
    for(tr=r+1,tf=f+1; tr<8&&tf<8; tr++,tf++) { atk|=1ULL<<(tr*8+tf); if(occ&(1ULL<<(tr*8+tf))) break; }
    for(tr=r+1,tf=f-1; tr<8&&tf>=0; tr++,tf--){ atk|=1ULL<<(tr*8+tf); if(occ&(1ULL<<(tr*8+tf))) break; }
    for(tr=r-1,tf=f+1; tr>=0&&tf<8; tr--,tf++) { atk|=1ULL<<(tr*8+tf); if(occ&(1ULL<<(tr*8+tf))) break; }
    for(tr=r-1,tf=f-1; tr>=0&&tf>=0; tr--,tf--){ atk|=1ULL<<(tr*8+tf); if(occ&(1ULL<<(tr*8+tf))) break; }
    return atk;
}

static U64 rookAttackOTF(int sq, U64 occ) {
    U64 atk = 0;
    int r = sq/8, f = sq%8, tr, tf;
    for(tr=r+1; tr<8; tr++) { atk|=1ULL<<(tr*8+f); if(occ&(1ULL<<(tr*8+f))) break; }
    for(tr=r-1; tr>=0; tr--){ atk|=1ULL<<(tr*8+f); if(occ&(1ULL<<(tr*8+f))) break; }
    for(tf=f+1; tf<8; tf++) { atk|=1ULL<<(r*8+tf); if(occ&(1ULL<<(r*8+tf))) break; }
    for(tf=f-1; tf>=0; tf--){ atk|=1ULL<<(r*8+tf); if(occ&(1ULL<<(r*8+tf))) break; }
    return atk;
}

static U64 bishopMaskGen(int sq) {
    U64 atk = 0;
    int r=sq/8, f=sq%8, tr, tf;
    for(tr=r+1,tf=f+1; tr<7&&tf<7; tr++,tf++) atk|=1ULL<<(tr*8+tf);
    for(tr=r+1,tf=f-1; tr<7&&tf>0; tr++,tf--) atk|=1ULL<<(tr*8+tf);
    for(tr=r-1,tf=f+1; tr>0&&tf<7; tr--,tf++) atk|=1ULL<<(tr*8+tf);
    for(tr=r-1,tf=f-1; tr>0&&tf>0; tr--,tf--) atk|=1ULL<<(tr*8+tf);
    return atk;
}

static U64 rookMaskGen(int sq) {
    U64 atk = 0;
    int r=sq/8, f=sq%8, tr, tf;
    for(tr=r+1; tr<7; tr++) atk|=1ULL<<(tr*8+f);
    for(tr=r-1; tr>0; tr--) atk|=1ULL<<(tr*8+f);
    for(tf=f+1; tf<7; tf++) atk|=1ULL<<(r*8+tf);
    for(tf=f-1; tf>0; tf--) atk|=1ULL<<(r*8+tf);
    return atk;
}

static U64 setOccupancy(int idx, int bits, U64 mask) {
    U64 occ = 0;
    for(int i=0; i<bits; i++) {
        int sq = popLSBIdx(mask);
        if(idx & (1<<i)) occ |= 1ULL<<sq;
    }
    return occ;
}

void initAttacks() {
    // Pawn attacks
    for(int sq=0; sq<64; sq++) {
        U64 b = 1ULL<<sq;
        pawnAttacks[WHITE][sq] = ((b<<9)&~FILE_A) | ((b<<7)&~FILE_H);
        pawnAttacks[BLACK][sq] = ((b>>7)&~FILE_A) | ((b>>9)&~FILE_H);
    }
    // Knight attacks
    for(int sq=0; sq<64; sq++) {
        U64 b=1ULL<<sq, atk=0;
        atk|=((b<<17)&~FILE_A); atk|=((b<<15)&~FILE_H);
        atk|=((b<<10)&~(FILE_A|(FILE_A<<1))); atk|=((b<<6)&~(FILE_H|(FILE_H>>1)));
        atk|=((b>>17)&~FILE_H); atk|=((b>>15)&~FILE_A);
        atk|=((b>>10)&~(FILE_H|(FILE_H>>1))); atk|=((b>>6)&~(FILE_A|(FILE_A<<1)));
        knightAttacks[sq]=atk;
    }
    // King attacks
    for(int sq=0; sq<64; sq++) {
        U64 b=1ULL<<sq, atk=0;
        atk|=((b<<8)); atk|=((b>>8));
        atk|=((b<<1)&~FILE_A); atk|=((b>>1)&~FILE_H);
        atk|=((b<<9)&~FILE_A); atk|=((b<<7)&~FILE_H);
        atk|=((b>>7)&~FILE_A); atk|=((b>>9)&~FILE_H);
        kingAttacks[sq]=atk;
    }
    // Sliding pieces
    for(int sq=0; sq<64; sq++) {
        bishopMasks[sq] = bishopMaskGen(sq);
        rookMasks[sq]   = rookMaskGen(sq);
        bishopBits[sq]  = BISHOP_BITS_TABLE[sq];
        rookBits[sq]    = ROOK_BITS_TABLE[sq];
        bishopMagics[sq]= BISHOP_MAGIC[sq];
        rookMagics[sq]  = ROOK_MAGIC[sq];

        int bOcc = 1 << bishopBits[sq];
        for(int i=0; i<bOcc; i++) {
            U64 occ = setOccupancy(i, bishopBits[sq], bishopMasks[sq]);
            int idx = (int)((occ * bishopMagics[sq]) >> (64 - bishopBits[sq]));
            bishopAttacks[sq][idx] = bishopAttackOTF(sq, occ);
        }
        int rOcc = 1 << rookBits[sq];
        for(int i=0; i<rOcc; i++) {
            U64 occ = setOccupancy(i, rookBits[sq], rookMasks[sq]);
            int idx = (int)((occ * rookMagics[sq]) >> (64 - rookBits[sq]));
            rookAttacks[sq][idx] = rookAttackOTF(sq, occ);
        }
    }
}

inline U64 getBishopAttacks(int sq, U64 occ) {
    occ &= bishopMasks[sq];
    occ  = (occ * bishopMagics[sq]) >> (64 - bishopBits[sq]);
    return bishopAttacks[sq][occ];
}
inline U64 getRookAttacks(int sq, U64 occ) {
    occ &= rookMasks[sq];
    occ  = (occ * rookMagics[sq]) >> (64 - rookBits[sq]);
    return rookAttacks[sq][occ];
}
inline U64 getQueenAttacks(int sq, U64 occ) {
    return getBishopAttacks(sq,occ) | getRookAttacks(sq,occ);
}

// ============================================================
//  BOARD
// ============================================================

static const char PIECE_CHARS[2][7] = {
    {'P','N','B','R','Q','K','.'},
    {'p','n','b','r','q','k','.'}
};

void Board::reset() {
    memset(pieces,0,sizeof(pieces));
    memset(occupancy,0,sizeof(occupancy));
    for(int i=0;i<64;i++) { pieceOn[i]=NO_PIECE; colorOn[i]=BOTH; }
    side=WHITE; enPassant=NO_SQ; castling=0; halfMove=0; fullMove=1;
}

void Board::setFromFEN(const std::string &fen) {
    reset();
    std::istringstream ss(fen);
    std::string board, sideStr, castleStr, epStr;
    ss >> board >> sideStr >> castleStr >> epStr >> halfMove >> fullMove;

    int sq=56;
    for(char c : board) {
        if(c=='/') { sq-=16; }
        else if(c>='1'&&c<='8') { sq+=c-'0'; }
        else {
            int col = (c>='a'&&c<='z') ? BLACK : WHITE;
            char lc = tolower(c);
            int pt = (lc=='p')?PAWN:(lc=='n')?KNIGHT:(lc=='b')?BISHOP:
                     (lc=='r')?ROOK:(lc=='q')?QUEEN:KING;
            setBit(pieces[col][pt], sq);
            pieceOn[sq]=pt; colorOn[sq]=col;
            sq++;
        }
    }
    for(int c=0;c<2;c++) for(int p=0;p<6;p++) occupancy[c]|=pieces[c][p];
    occupancy[BOTH]=occupancy[WHITE]|occupancy[BLACK];

    side = (sideStr=="w") ? WHITE : BLACK;

    castling=0;
    for(char c:castleStr) {
        if(c=='K') castling|=WK;
        if(c=='Q') castling|=WQ;
        if(c=='k') castling|=BK;
        if(c=='q') castling|=BQ;
    }
    if(epStr!="-" && epStr.length()>=2) {
        int f=epStr[0]-'a', r=epStr[1]-'1';
        enPassant=r*8+f;
    }
}

std::string Board::toFEN() const {
    std::string fen;
    
    for(int r = 7; r >= 0; r--) {
        int empty = 0;
        for(int f = 0; f < 8; f++) {
            int sq = r * 8 + f;
            if(pieceOn[sq] == NO_PIECE) {
                empty++;
            } else {
                if(empty > 0) {
                    fen += char('0' + empty);
                    empty = 0;
                }
                int cidx = (colorOn[sq] == BLACK) ? 1 : 0;
                fen += PIECE_CHARS[cidx][pieceOn[sq]];
            }
        }
        if(empty > 0) fen += char('0' + empty);
        if(r > 0) fen += '/';
    }
    
    fen += (side == WHITE) ? " w " : " b ";
    
    std::string castle;
    if(castling & WK) castle += 'K';
    if(castling & WQ) castle += 'Q';
    if(castling & BK) castle += 'k';
    if(castling & BQ) castle += 'q';
    fen += castle.empty() ? "-" : castle;
    fen += ' ';
    
    if(enPassant == NO_SQ) {
        fen += "-";
    } else {
        fen += squareName(enPassant);
    }
    fen += ' ';
    
    fen += std::to_string(halfMove) + " " + std::to_string(fullMove);
    
    return fen;
}

void Board::print() const {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for(int r=7;r>=0;r--) {
        std::cout << r+1 << " |";
        for(int f=0;f<8;f++) {
            int sq=r*8+f;
            int cidx = (colorOn[sq]==BLACK) ? 1 : 0;
            char c = (pieceOn[sq]==NO_PIECE) ? '.' : PIECE_CHARS[cidx][pieceOn[sq]];
            std::cout << " " << c << " |";
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
    std::cout << "Side: " << (side==WHITE?"White":"Black")
              << "  EP: " << enPassant
              << "  Castle: " << castling << "\n\n";
}

bool Board::isSquareAttacked(int sq, int byColor) const {
    if(pawnAttacks[byColor^1][sq] & pieces[byColor][PAWN])   return true;
    if(knightAttacks[sq]          & pieces[byColor][KNIGHT]) return true;
    if(kingAttacks[sq]            & pieces[byColor][KING])   return true;
    if(getBishopAttacks(sq,occupancy[BOTH]) & (pieces[byColor][BISHOP]|pieces[byColor][QUEEN])) return true;
    if(getRookAttacks(sq,occupancy[BOTH])   & (pieces[byColor][ROOK]  |pieces[byColor][QUEEN])) return true;
    return false;
}

// ============================================================
//  MAKE / UNMAKE MOVE - FULLY FIXED
// ============================================================

bool Board::makeMove(Move mv, UndoInfo &undo) {
    undo.enPassant=enPassant; undo.castling=castling;
    undo.halfMove=halfMove; undo.move=mv;
    undo.capturedPiece=NO_PIECE; undo.capturedColor=BOTH;

    int from=moveFrom(mv), to=moveTo(mv);
    int pt=pieceOn[from], col=side, opp=col^1;

    // SAFETY CHECK: Valid piece and squares
    if(pt == NO_PIECE || from < 0 || from > 63 || to < 0 || to > 63) {
        return false;
    }
    if(colorOn[from] != col) return false;

    // SAFETY: Capture flags must match board reality
    if(!moveEP(mv) && moveCapture(mv)) {
        if(pieceOn[to] == NO_PIECE || colorOn[to] != opp) return false;
    }
    if(!moveCapture(mv) && !moveEP(mv) && pieceOn[to] != NO_PIECE) {
        return false;
    }
    if(moveEP(mv)) {
        int capSq = to + (col==WHITE ? -8 : 8);
        if(pt != PAWN || to != enPassant || capSq < 0 || capSq > 63) return false;
        if(pieceOn[capSq] != PAWN || colorOn[capSq] != opp) return false;
    }
    if(moveCastle(mv)) {
        if(pt != KING) return false;
        int rookFrom = -1;
        if(to==G1) rookFrom=H1;
        else if(to==C1) rookFrom=A1;
        else if(to==G8) rookFrom=H8;
        else if(to==C8) rookFrom=A8;
        else return false;
        if(pieceOn[rookFrom] != ROOK || colorOn[rookFrom] != col) return false;
    }

    // Capture
    if(moveCapture(mv) && !moveEP(mv)) {
        undo.capturedPiece=pieceOn[to];
        undo.capturedColor=opp;
        if(undo.capturedPiece != NO_PIECE) {
            clearBit(pieces[opp][pieceOn[to]], to);
            occupancy[opp]&=~(1ULL<<to);
        }
        halfMove=0;
    } else if(pt==PAWN) halfMove=0;
    else halfMove++;

    // Move piece
    clearBit(pieces[col][pt], from);
    setBit(pieces[col][pt], to);
    pieceOn[to]=pt; colorOn[to]=col;
    pieceOn[from]=NO_PIECE; colorOn[from]=BOTH;

    // En passant capture
    if(moveEP(mv)) {
        int capSq = to + (col==WHITE ? -8 : 8);
        undo.capturedPiece=PAWN; undo.capturedColor=opp;
        clearBit(pieces[opp][PAWN], capSq);
        occupancy[opp]&=~(1ULL<<capSq);
        pieceOn[capSq]=NO_PIECE; colorOn[capSq]=BOTH;
    }
    enPassant=NO_SQ;

    // Double pawn push
    if(moveDP(mv)) enPassant = to + (col==WHITE ? -8 : 8);

    // Promotion
    if(movePromo(mv)) {
        int promoPt = movePromo(mv);
        clearBit(pieces[col][PAWN], to);
        setBit(pieces[col][promoPt], to);
        pieceOn[to]=promoPt;
    }

    // Castling
    if(moveCastle(mv)) {
        int rookFrom, rookTo;
        if(to==G1){rookFrom=H1;rookTo=F1;}
        else if(to==C1){rookFrom=A1;rookTo=D1;}
        else if(to==G8){rookFrom=H8;rookTo=F8;}
        else {rookFrom=A8;rookTo=D8;}
        clearBit(pieces[col][ROOK],rookFrom);
        setBit(pieces[col][ROOK],rookTo);
        pieceOn[rookTo]=ROOK; colorOn[rookTo]=col;
        pieceOn[rookFrom]=NO_PIECE; colorOn[rookFrom]=BOTH;
        occupancy[col]=(occupancy[col]&~(1ULL<<rookFrom))|(1ULL<<rookTo);
    }

    // Update castling rights
    static const int castleMask[64] = {
        ~WQ,15,15,15,~(WK|WQ),15,15,~WK,
        15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,
        ~BQ,15,15,15,~(BK|BQ),15,15,~BK
    };
    castling &= castleMask[from] & castleMask[to];

    // Update occupancy
    occupancy[col]=(occupancy[col]&~(1ULL<<from))|(1ULL<<to);
    occupancy[BOTH]=occupancy[WHITE]|occupancy[BLACK];

    side^=1;
    if(side==WHITE) fullMove++;

    // CRITICAL: Check if our king is safe
    U64 kingBB = pieces[col][KING];
    if(!kingBB) {
        unmakeMove(mv, undo);
        return false;
    }
    
    int kingSq = lsb(kingBB);
    if(isSquareAttacked(kingSq, side)) {
        unmakeMove(mv, undo);
        return false;
    }

    return true;
}

void Board::unmakeMove(Move mv, const UndoInfo &undo) {
    side^=1;
    if(side==BLACK) fullMove--;

    int from=moveFrom(mv), to=moveTo(mv);
    int col=side, opp=col^1;
    
    // FIX: Get piece type BEFORE clearing
    int currentPiece = pieceOn[to];
    int pt = movePromo(mv) ? PAWN : currentPiece;

    // Move piece back
    if(pt != NO_PIECE && currentPiece != NO_PIECE) {
        clearBit(pieces[col][currentPiece], to);
        setBit(pieces[col][pt], from);
        pieceOn[from]=pt; colorOn[from]=col;
    }
    pieceOn[to]=NO_PIECE; colorOn[to]=BOTH;

    // Restore capture
    if(undo.capturedPiece!=NO_PIECE && !moveEP(mv)) {
        setBit(pieces[opp][undo.capturedPiece], to);
        pieceOn[to]=undo.capturedPiece; colorOn[to]=opp;
    }
    if(moveEP(mv)) {
        int capSq = to + (col==WHITE ? -8 : 8);
        setBit(pieces[opp][PAWN], capSq);
        pieceOn[capSq]=PAWN; colorOn[capSq]=opp;
    }
    // Castling
    if(moveCastle(mv)) {
        int rookFrom, rookTo;
        if(to==G1){rookFrom=H1;rookTo=F1;}
        else if(to==C1){rookFrom=A1;rookTo=D1;}
        else if(to==G8){rookFrom=H8;rookTo=F8;}
        else {rookFrom=A8;rookTo=D8;}
        clearBit(pieces[col][ROOK],rookTo);
        setBit(pieces[col][ROOK],rookFrom);
        pieceOn[rookFrom]=ROOK; colorOn[rookFrom]=col;
        pieceOn[rookTo]=NO_PIECE; colorOn[rookTo]=BOTH;
        occupancy[col]=(occupancy[col]&~(1ULL<<rookTo))|(1ULL<<rookFrom);
    }

    enPassant=undo.enPassant; castling=undo.castling; halfMove=undo.halfMove;
    occupancy[WHITE]=0; occupancy[BLACK]=0;
    for(int p=0;p<6;p++) { occupancy[WHITE]|=pieces[WHITE][p]; occupancy[BLACK]|=pieces[BLACK][p]; }
    occupancy[BOTH]=occupancy[WHITE]|occupancy[BLACK];
}

// ============================================================
//  MOVE GENERATION - COMPLETELY REWRITTEN (BUG FREE)
// ============================================================

static void addPawnMoves(const Board &b, MoveList &ml, int from, int to, bool cap) {
    int promoRank = (b.side==WHITE) ? 7 : 0;
    int toRank = to / 8;
    if(toRank == promoRank) {
        for(int p=1;p<=4;p++) ml.add(encodeMove(from,to,p,cap?1:0,0,0,0));
    } else {
        ml.add(encodeMove(from,to,0,cap?1:0,0,0,0));
    }
}

void generateMoves(const Board &b, MoveList &ml) {
    ml.clear();
    int col=b.side, opp=col^1;
    U64 enemies = b.occupancy[opp];

    // PAWNS - COMPLETELY SAFE VERSION
    U64 pawns = b.pieces[col][PAWN];
    while(pawns) {
        int sq = popLSBIdx(pawns);
        int rank = sq / 8;
        int file = sq % 8;
        int dir = (col == WHITE) ? 8 : -8;
        
        // 1. SINGLE PUSH (only if empty)
        int pushSq = sq + dir;
        if(pushSq >= 0 && pushSq < 64 && b.pieceOn[pushSq] == NO_PIECE) {
            addPawnMoves(b, ml, sq, pushSq, false);
            
            // 2. DOUBLE PUSH (only from start rank AND push was empty)
            int startRank = (col == WHITE) ? 1 : 6;
            if(rank == startRank) {
                int doublePushSq = pushSq + dir;
                if(doublePushSq >= 0 && doublePushSq < 64 && 
                   b.pieceOn[doublePushSq] == NO_PIECE) {
                    ml.add(encodeMove(sq, doublePushSq, 0, 0, 1, 0, 0));
                }
            }
        }
        
        // 3. CAPTURES (diagonal - must have enemy piece)
        int capOffsets[2] = {dir - 1, dir + 1};
        for(int i = 0; i < 2; i++) {
            int capSq = sq + capOffsets[i];
            if(capSq >= 0 && capSq < 64) {
                int capFile = capSq % 8;
                // Must be adjacent file (no wrap around)
                if(abs(capFile - file) == 1) {
                    if(b.pieceOn[capSq] != NO_PIECE && b.colorOn[capSq] == opp) {
                        addPawnMoves(b, ml, sq, capSq, true);
                    }
                }
            }
        }
        
        // 4. EN PASSANT
        if(b.enPassant != NO_SQ) {
            int epFile = b.enPassant % 8;
            int expectedRank = (col == WHITE) ? 4 : 3;
            
            if(rank == expectedRank && abs(epFile - file) == 1) {
                ml.add(encodeMove(sq, b.enPassant, 0, 1, 0, 1, 0));
            }
        }
    }

    // KNIGHTS
    U64 knights = b.pieces[col][KNIGHT];
    while(knights) {
        int sq=popLSBIdx(knights);
        U64 atk = knightAttacks[sq] & ~b.occupancy[col];
        while(atk) {
            int to=popLSBIdx(atk);
            ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));
        }
    }

    // BISHOPS
    U64 bishops = b.pieces[col][BISHOP];
    while(bishops) {
        int sq=popLSBIdx(bishops);
        U64 atk = getBishopAttacks(sq,b.occupancy[BOTH]) & ~b.occupancy[col];
        while(atk) {
            int to=popLSBIdx(atk);
            ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));
        }
    }

    // ROOKS
    U64 rooks = b.pieces[col][ROOK];
    while(rooks) {
        int sq=popLSBIdx(rooks);
        U64 atk = getRookAttacks(sq,b.occupancy[BOTH]) & ~b.occupancy[col];
        while(atk) {
            int to=popLSBIdx(atk);
            ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));
        }
    }

    // QUEENS
    U64 queens = b.pieces[col][QUEEN];
    while(queens) {
        int sq=popLSBIdx(queens);
        U64 atk = getQueenAttacks(sq,b.occupancy[BOTH]) & ~b.occupancy[col];
        while(atk) {
            int to=popLSBIdx(atk);
            ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));
        }
    }

    // KING
    if(b.pieces[col][KING]) {
        int sq=lsb(b.pieces[col][KING]);
        U64 atk=kingAttacks[sq] & ~b.occupancy[col];
        while(atk) {
            int to=popLSBIdx(atk);
            ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));
        }
        // Castling
        if(col==WHITE) {
            if((b.castling&WK) && !getBit(b.occupancy[BOTH],F1) && !getBit(b.occupancy[BOTH],G1)
               && !b.isSquareAttacked(E1,BLACK) && !b.isSquareAttacked(F1,BLACK))
                ml.add(encodeMove(E1,G1,0,0,0,0,1));
            if((b.castling&WQ) && !getBit(b.occupancy[BOTH],D1) && !getBit(b.occupancy[BOTH],C1) && !getBit(b.occupancy[BOTH],B1)
               && !b.isSquareAttacked(E1,BLACK) && !b.isSquareAttacked(D1,BLACK))
                ml.add(encodeMove(E1,C1,0,0,0,0,1));
        } else {
            if((b.castling&BK) && !getBit(b.occupancy[BOTH],F8) && !getBit(b.occupancy[BOTH],G8)
               && !b.isSquareAttacked(E8,WHITE) && !b.isSquareAttacked(F8,WHITE))
                ml.add(encodeMove(E8,G8,0,0,0,0,1));
            if((b.castling&BQ) && !getBit(b.occupancy[BOTH],D8) && !getBit(b.occupancy[BOTH],C8) && !getBit(b.occupancy[BOTH],B8)
               && !b.isSquareAttacked(E8,WHITE) && !b.isSquareAttacked(D8,WHITE))
                ml.add(encodeMove(E8,C8,0,0,0,0,1));
        }
    }
}

// ============================================================
//  MOVE ORDERING (MVV-LVA)
// ============================================================

// Improved piece values (more accurate)
static const int MATERIAL[6] = {100, 325, 335, 500, 975, 20000};

static const int MVV_LVA[6][6] = {
    {15, 14, 13, 12, 11, 10},
    {25, 24, 23, 22, 21, 20},
    {35, 34, 33, 32, 31, 30},
    {45, 44, 43, 42, 41, 40},
    {55, 54, 53, 52, 51, 50},
    {0,  0,  0,  0,  0,  0}
};

static int scoreMove(const Board &b, Move m) {
    int score = 0;
    
    if(moveCapture(m)) {
        int victim = b.pieceOn[moveTo(m)];
        int attacker = b.pieceOn[moveFrom(m)];
        
        if(victim != NO_PIECE && attacker != NO_PIECE) {
            score += MVV_LVA[victim][attacker] + 10000;
        } else if(moveEP(m)) {
            score += 10000 + 15;
        }
    }
    
    if(movePromo(m)) {
        score += 5000 + movePromo(m) * 100;
    }
    
    if(moveCastle(m)) {
        score += 100;
    }
    
    return score;
}

void scoreAndSortMoves(const Board &b, MoveList &ml, Move pvMove, int *scores) {
    int tempScores[256];
    if(!scores) scores = tempScores;
    
    for(int i = 0; i < ml.count; i++) {
        if(ml.moves[i] == pvMove) {
            scores[i] = 100000;
        } else {
            scores[i] = scoreMove(b, ml.moves[i]);
        }
    }
    
    for(int i = 0; i < ml.count - 1; i++) {
        int bestIdx = i;
        for(int j = i + 1; j < ml.count; j++) {
            if(scores[j] > scores[bestIdx]) {
                bestIdx = j;
            }
        }
        if(bestIdx != i) {
            std::swap(scores[i], scores[bestIdx]);
            std::swap(ml.moves[i], ml.moves[bestIdx]);
        }
    }
}

// ============================================================
//  EVALUATION
// ============================================================

static const int PST_PAWN[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     55,  55,  55,  55,  55,  55,  55,  55,
     12,  15,  25,  35,  35,  25,  15,  12,
      8,   8,  15,  30,  30,  15,   8,   8,
      3,   3,   5,  25,  25,   5,   3,   3,
      0,   0,   0,  15,  15,   0,   0,   0,
      5,   8,  10,-10, -10,  10,   8,   5,
      0,   0,   0,   0,   0,   0,   0,   0
};
static const int PST_KNIGHT[64] = {
    -60, -45, -35, -30, -30, -35, -45, -60,
    -45, -20, -10,  -5,  -5, -10, -20, -45,
    -35,  -5,   5,  15,  15,   5,  -5, -35,
    -30,   0,  15,  25,  25,  15,   0, -30,
    -30,   5,  15,  25,  25,  15,   5, -30,
    -35,  -5,  10,  15,  15,  10,  -5, -35,
    -45, -25,  -5,   0,   0,  -5, -25, -45,
    -60, -45, -35, -30, -30, -35, -45, -60
};
static const int PST_BISHOP[64] = {
    -25, -15, -10, -10, -10, -10, -15, -25,
    -15,  -5,   0,   5,   5,   0,  -5, -15,
    -10,   5,  10, 15,  15,  10,   5, -10,
    -10,  10,  15, 20,  20,  15,  10, -10,
    -10,   5,  15, 20,  20,  15,   5, -10,
    -10,   0,  10, 15,  15,  10,   0, -10,
    -15,  -5,   0,  5,   5,   0,  -5, -15,
    -25, -15, -10, -10, -10, -10, -15, -25
};
static const int PST_ROOK[64] = {
      5,  10,  15,  20,  20,  15,  10,   5,
     -5,   5,  10,  15,  15,  10,   5,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   0,   0,   5,   5,   0,   0,  -5,
      0,  0,   5,  10,  10,   5,   0,   0
};
static const int PST_QUEEN[64] = {
    -25, -15, -10,  -5,  -5, -10, -15, -25,
    -15,  -5,   5,  10,  10,   5,  -5, -15,
    -10,   5,  10,  15,  15,  10,   5, -10,
     -5,  5,  10,  15,  15,  10,   5,  -5,
      0,  5,  10,  15,  15,  10,   5,  -5,
    -10,  5,  10,  10,  10,  10,   5, -10,
    -15,  -5,   5,   5,   5,   5,  -5, -15,
    -25, -15, -10,  -5,  -5, -10, -15, -25
};
static const int PST_KING_MG[64] = {
    -40, -50, -50, -60, -60, -50, -50, -40,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -40, -50, -50, -60, -60, -50, -50, -40,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -10,  10,  10, -10, -20, -10,
     30,  20,  10,   0,   0,  10,  20,  30
};

static const int* PST[6] = {PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING_MG};

int evaluate(const Board &b) {
    int score = 0;
    for(int col=0; col<2; col++) {
        int sign = (col==WHITE) ? 1 : -1;
        for(int pt=0; pt<6; pt++) {
            U64 bb = b.pieces[col][pt];
            while(bb) {
                int sq = popLSBIdx(bb);
                int pstSq = (col==WHITE) ? sq : (sq^56);
                score += sign * (MATERIAL[pt] + PST[pt][pstSq]);
            }
        }
    }
    return (b.side==WHITE) ? score : -score;
}

// ============================================================
//  SEARCH - FIXED VERSION
// ============================================================

static const int INF = 1000000;
static const int MATE = 900000;
static const int MAX_DEPTH = 20;

int quiescence(Board &b, int alpha, int beta, SearchInfo &info, int qdepth) {
    if(qdepth >= 8) {
        return evaluate(b);
    }
    
    info.nodes++;
    
    if(info.timeLimit > 0 && (info.nodes & 2047) == 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - info.startTime).count();
        if(elapsed >= info.timeLimit) { 
            info.stop = true; 
            return 0; 
        }
    }
    
    int stand_pat = evaluate(b);
    if(stand_pat >= beta) return beta;
    if(stand_pat > alpha) alpha = stand_pat;

    MoveList ml;
    generateMoves(b, ml);
    
    int scores[256];
    int captureCount = 0;
    
    for(int i = 0; i < ml.count; i++) {
        if(moveCapture(ml.moves[i])) {
            scores[captureCount] = scoreMove(b, ml.moves[i]);
            ml.moves[captureCount] = ml.moves[i];
            captureCount++;
        }
    }
    ml.count = captureCount;
    
    for(int i = 0; i < ml.count - 1; i++) {
        for(int j = i + 1; j < ml.count; j++) {
            if(scores[j] > scores[i]) {
                std::swap(scores[i], scores[j]);
                std::swap(ml.moves[i], ml.moves[j]);
            }
        }
    }

    Board::UndoInfo undo;
    for(int i = 0; i < ml.count; i++) {
        int victim = b.pieceOn[moveTo(ml.moves[i])];
        int victimVal = (victim != NO_PIECE) ? MATERIAL[victim] : 0;
        if(stand_pat + victimVal + 200 < alpha) continue;

        if(!b.makeMove(ml.moves[i], undo)) continue;
        
        int score = -quiescence(b, -beta, -alpha, info, qdepth + 1);
        b.unmakeMove(ml.moves[i], undo);
        
        if(info.stop) return 0;
        if(score >= beta) return beta;
        if(score > alpha) alpha = score;
    }
    return alpha;
}

int alphaBeta(Board &b, int alpha, int beta, int depth, SearchInfo &info, bool nullMoveAllowed) {
    if(depth > MAX_DEPTH) {
        return evaluate(b);
    }
    
    info.nodes++;
    
    if(info.timeLimit > 0 && (info.nodes & 2047) == 0) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - info.startTime).count();
        if(elapsed >= info.timeLimit) { 
            info.stop = true; 
            return 0; 
        }
    }

    if(depth <= 0) return quiescence(b, alpha, beta, info, 0);

    if(nullMoveAllowed && depth >= 3 && !b.inCheck()) {
        b.side ^= 1;
        int nullScore = -alphaBeta(b, -beta, -beta + 1, depth - 1 - 2, info, false);
        b.side ^= 1;
        
        if(info.stop) return 0;
        if(nullScore >= beta) {
            return beta;
        }
    }

    MoveList ml;
    generateMoves(b, ml);
    
    if(ml.count == 0) {
        if(b.inCheck()) {
            return -(MATE - (info.depth - depth));
        } else {
            return 0;
        }
    }

    scoreAndSortMoves(b, ml, 0, nullptr);

    Board::UndoInfo undo;
    int legalMoves = 0;
    int bestScore = -INF;
    bool hasLegalMove = false;

    for(int i = 0; i < ml.count; i++) {
        if(!b.makeMove(ml.moves[i], undo)) continue;
        
        hasLegalMove = true;
        legalMoves++;
        
        int score;
        if(legalMoves == 1) {
            score = -alphaBeta(b, -beta, -alpha, depth - 1, info, true);
        } else {
            score = -alphaBeta(b, -alpha - 1, -alpha, depth - 1, info, true);
            if(score > alpha && score < beta) {
                score = -alphaBeta(b, -beta, -alpha, depth - 1, info, true);
            }
        }
        
        b.unmakeMove(ml.moves[i], undo);

        if(info.stop) return 0;
        
        if(score > bestScore) bestScore = score;
        if(score > alpha) {
            alpha = score;
            if(alpha >= beta) return beta;
        }
    }

    if(!hasLegalMove) {
        return b.inCheck() ? -(MATE - (info.depth - depth)) : 0;
    }
    
    return bestScore;
}

Move bestMove(Board &b, int depth, int timeLimitMs) {
    SearchInfo info;
    info.depth     = depth;
    info.nodes     = 0;
    info.stop      = false;
    info.timeLimit = timeLimitMs;
    info.startTime = std::chrono::steady_clock::now();

    MoveList ml;
    generateMoves(b, ml);
    
    MoveList legal;
    Board::UndoInfo testUndo;
    
    for(int i = 0; i < ml.count; i++) {
        Board tmp = b;
        if(tmp.makeMove(ml.moves[i], testUndo)) {
            legal.add(ml.moves[i]);
        }
    }

    if(legal.count == 0) {
        std::cout << "bestmove 0000\n";
        return 0;
    }

    // 🔴 OPENING RANDOMNESS: Early game shuffle for variety
    int totalPieces = popcount(b.occupancy[BOTH]);
    if(totalPieces > 28 && legal.count > 5) {
        // Shuffle first 3 moves
        for(int i = 0; i < 3 && i < legal.count - 1; i++) {
            int j = i + (rand() % (legal.count - i));
            std::swap(legal.moves[i], legal.moves[j]);
        }
    }

    Move best = legal.moves[0];
    int bestScore = -INF;
    bool openingPhase = popcount(b.occupancy[BOTH]) > 26;

    for(int d = 1; d <= depth && !info.stop; d++) {
        int currentBest = -INF;
        Move iterBest = legal.moves[0];
        int iterScores[256];
        for(int i = 0; i < legal.count; i++) iterScores[i] = -INF;
        int alpha = -INF, beta = INF;

        for(int i = 0; i < legal.count && !info.stop; i++) {
            Board::UndoInfo undo;
            
            if(!b.makeMove(legal.moves[i], undo)) continue;

            int score;
            if(i == 0) {
                score = -alphaBeta(b, -beta, -alpha, d - 1, info, true);
            } else {
                score = -alphaBeta(b, -alpha - 1, -alpha, d - 1, info, true);
                if(score > alpha && score < beta) {
                    score = -alphaBeta(b, -beta, -alpha, d - 1, info, true);
                }
            }
            
            b.unmakeMove(legal.moves[i], undo);

            if(info.stop) break;
            iterScores[i] = score;

            if(score > currentBest) {
                currentBest = score;
                iterBest = legal.moves[i];
            }
            
            if(score > alpha) alpha = score;
        }

        if(!info.stop) {
            // In the opening, add controlled variety among near-equal moves.
            if(openingPhase) {
                int candidates[256];
                int candidateCount = 0;
                const int blendWindow = 20; // centipawns
                for(int i = 0; i < legal.count; i++) {
                    if(iterScores[i] >= currentBest - blendWindow) {
                        candidates[candidateCount++] = i;
                    }
                }
                if(candidateCount > 0) {
                    int pick = candidates[rand() % candidateCount];
                    iterBest = legal.moves[pick];
                }
            }

            best = iterBest;
            bestScore = currentBest;
            
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - info.startTime).count();

            std::cout << "info depth " << d
                      << " score cp " << bestScore
                      << " nodes " << info.nodes
                      << " time " << elapsed
                      << " pv " << moveToStr(best) << "\n";
            std::cout.flush();
        }
    }

    if(best == 0) {
        std::cout << "bestmove 0000\n";
    } else {
        std::cout << "bestmove " << moveToStr(best) << "\n";
    }
    std::cout.flush();
    
    return best;
}

// ============================================================
//  UCI PROTOCOL - ZERO v1.2
// ============================================================

std::string squareName(int sq) {
    std::string s;
    s += (char)('a' + sq % 8);
    s += (char)('1' + sq / 8);
    return s;
}

std::string moveToStr(Move m) {
    if(m == 0) return "0000";
    std::string s = squareName(moveFrom(m)) + squareName(moveTo(m));
    if(movePromo(m)) {
        const char promo[] = {'?', 'n', 'b', 'r', 'q'};
        s += promo[movePromo(m)];
    }
    return s;
}

Move strToMove(const Board &b, const std::string &str) {
    if(str.length() < 4) return 0;
    
    int from = (str[0] - 'a') + (str[1] - '1') * 8;
    int to = (str[2] - 'a') + (str[3] - '1') * 8;
    
    if(from < 0 || from > 63 || to < 0 || to > 63) return 0;
    
    MoveList ml;
    generateMoves(b, ml);
    
    for(int i = 0; i < ml.count; i++) {
        if(moveFrom(ml.moves[i]) == from && moveTo(ml.moves[i]) == to) {
            if(str.length() > 4) {
                char promoChar = str[4];
                int promo = 0;
                if(promoChar == 'n') promo = 1;
                else if(promoChar == 'b') promo = 2;
                else if(promoChar == 'r') promo = 3;
                else if(promoChar == 'q') promo = 4;
                
                if(movePromo(ml.moves[i]) != promo) continue;
            } else if(movePromo(ml.moves[i]) != 0) {
                continue;
            }
            
            Board::UndoInfo undo;
            Board tmp = b;
            if(tmp.makeMove(ml.moves[i], undo)) {
                return ml.moves[i];
            }
        }
    }
    return 0;
}

void uciLoop() {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin,  NULL, _IONBF, 0);

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
            std::cout << "id name Zero v1.2\n"
                      << "id author Bhai+Claude\n"
                      << "uciok\n";
            std::cout.flush();
        }
        else if(token == "isready") {
            std::cout << "readyok\n";
            std::cout.flush();
        }
        else if(token == "ucinewgame") {
            b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }
        else if(token == "position") {
            std::string pos;
            ss >> pos;
            
            if(pos == "startpos") {
                b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                ss >> token;
            } else if(pos == "fen") {
                std::string fen, p;
                ss >> p;
                fen = p;
                while(ss >> p && p != "moves") {
                    fen += " " + p;
                }
                b.setFromFEN(fen);
            }
            
            std::string mv;
            while(ss >> mv) {
                Move m = strToMove(b, mv);
                if(m) {
                    Board::UndoInfo u;
                    b.makeMove(m, u);
                }
            }
        }
        else if(token == "go") {
            int depth = 6;
            int movetime = -1;
            int wtime = -1, btime = -1;
            int winc = 0, binc = 0;
            
            while(ss >> token) {
                if(token == "depth") ss >> depth;
                else if(token == "movetime") ss >> movetime;
                else if(token == "wtime") ss >> wtime;
                else if(token == "btime") ss >> btime;
                else if(token == "winc") ss >> winc;
                else if(token == "binc") ss >> binc;
                else if(token == "infinite") { depth = 20; movetime = -1; }
            }
            
            // 🔴 SMART TIME MANAGEMENT
            if(movetime < 0 && wtime > 0 && btime > 0) {
                int myTime = (b.side == WHITE) ? wtime : btime;
                int myInc = (b.side == WHITE) ? winc : binc;
                
                // Use ~8% of remaining time + full increment for stronger play.
                movetime = (myTime / 12) + myInc;
                
                // Minimum 1.5s, maximum 12s to avoid insta-moves.
                movetime = std::max(1500, std::min(movetime, 12000));
            }
            
            // 🔴 DEPTH BASED ON TIME
            if(movetime > 10000) depth = 9;
            else if(movetime > 7000) depth = 8;
            else if(movetime > 4000) depth = 7;
            else if(movetime > 2000) depth = 6;
            else depth = 5;
            
            if(movetime < 0) {
                movetime = 6000;
                depth = 7;
            }

            Move m = bestMove(b, depth, movetime);
        }
        else if(token == "d") {
            b.print();
            std::cout.flush();
        }
        else if(token == "eval") {
            int score = evaluate(b);
            std::cout << "Evaluation: " << score << "\n";
            std::cout.flush();
        }
        else if(token == "quit" || token == "exit") {
            break;
        }
    }
}
