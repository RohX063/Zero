#include "chess.h"

// ── Optional weight files (graceful fallback if missing) ─────
#if __has_include("tal_weights_header.h")
  #include "tal_weights_header.h"
  #define HAS_TAL_SOUL 1
#else
  #define HAS_TAL_SOUL 0
  // Dummy TalSoul namespace so code compiles without weights
  namespace TalSoul { inline float forward(const float*){return 0.f;} }
#endif

#if __has_include("nnue_weights.h")
  #include "nnue_weights.h"
  #define HAS_NNUE 1
#else
  #define HAS_NNUE 0
#endif

// ============================================================
//  TAL SOUL — Board to Neural Network Input
// ============================================================
static void boardToTalVec(const Board& b, float* vec) {
    // 64 floats: piece value at each square
    // White pieces = positive, Black = negative
    // Same encoding as Python training script:
    // P=1, N=2, B=3, R=4, Q=5, K=6
    static const float PIECE_ENCODE[7] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 0.f};

    for(int sq = 0; sq < 64; sq++) {
        int pt  = b.pieceOn[sq];
        int col = b.colorOn[sq];
        if(pt == NO_PIECE || col == BOTH) {
            vec[sq] = 0.f;
        } else {
            float val = PIECE_ENCODE[pt];
            vec[sq] = (col == WHITE) ? val : -val;
        }
    }
}

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
    U64 atk=0; int r=sq/8,f=sq%8,tr,tf;
    for(tr=r+1,tf=f+1;tr<8&&tf<8;tr++,tf++){atk|=1ULL<<(tr*8+tf);if(occ&(1ULL<<(tr*8+tf)))break;}
    for(tr=r+1,tf=f-1;tr<8&&tf>=0;tr++,tf--){atk|=1ULL<<(tr*8+tf);if(occ&(1ULL<<(tr*8+tf)))break;}
    for(tr=r-1,tf=f+1;tr>=0&&tf<8;tr--,tf++){atk|=1ULL<<(tr*8+tf);if(occ&(1ULL<<(tr*8+tf)))break;}
    for(tr=r-1,tf=f-1;tr>=0&&tf>=0;tr--,tf--){atk|=1ULL<<(tr*8+tf);if(occ&(1ULL<<(tr*8+tf)))break;}
    return atk;
}

static U64 rookAttackOTF(int sq, U64 occ) {
    U64 atk=0; int r=sq/8,f=sq%8,tr,tf;
    for(tr=r+1;tr<8;tr++){atk|=1ULL<<(tr*8+f);if(occ&(1ULL<<(tr*8+f)))break;}
    for(tr=r-1;tr>=0;tr--){atk|=1ULL<<(tr*8+f);if(occ&(1ULL<<(tr*8+f)))break;}
    for(tf=f+1;tf<8;tf++){atk|=1ULL<<(r*8+tf);if(occ&(1ULL<<(r*8+tf)))break;}
    for(tf=f-1;tf>=0;tf--){atk|=1ULL<<(r*8+tf);if(occ&(1ULL<<(r*8+tf)))break;}
    return atk;
}

static U64 bishopMaskGen(int sq) {
    U64 atk=0; int r=sq/8,f=sq%8,tr,tf;
    for(tr=r+1,tf=f+1;tr<7&&tf<7;tr++,tf++) atk|=1ULL<<(tr*8+tf);
    for(tr=r+1,tf=f-1;tr<7&&tf>0;tr++,tf--) atk|=1ULL<<(tr*8+tf);
    for(tr=r-1,tf=f+1;tr>0&&tf<7;tr--,tf++) atk|=1ULL<<(tr*8+tf);
    for(tr=r-1,tf=f-1;tr>0&&tf>0;tr--,tf--) atk|=1ULL<<(tr*8+tf);
    return atk;
}

static U64 rookMaskGen(int sq) {
    U64 atk=0; int r=sq/8,f=sq%8,tr,tf;
    for(tr=r+1;tr<=6;tr++) atk|=1ULL<<(tr*8+f);
    for(tr=r-1;tr>=1;tr--) atk|=1ULL<<(tr*8+f);
    for(tf=f+1;tf<=6;tf++) atk|=1ULL<<(r*8+tf);
    for(tf=f-1;tf>=1;tf--) atk|=1ULL<<(r*8+tf);
    return atk;
}

static U64 setOccupancy(int idx, int bits, U64 mask) {
    U64 occ=0;
    for(int i=0;i<bits;i++){int sq=popLSBIdx(mask);if(idx&(1<<i))occ|=1ULL<<sq;}
    return occ;
}

void initAttacks() {
    for(int sq=0;sq<64;sq++){
        U64 b=1ULL<<sq;
        pawnAttacks[WHITE][sq]=((b<<9)&~FILE_A)|((b<<7)&~FILE_H);
        pawnAttacks[BLACK][sq]=((b>>7)&~FILE_A)|((b>>9)&~FILE_H);
    }
    for(int sq=0;sq<64;sq++){
        U64 b=1ULL<<sq,atk=0;
        atk|=((b<<17)&~FILE_A);atk|=((b<<15)&~FILE_H);
        atk|=((b<<10)&~(FILE_A|(FILE_A<<1)));atk|=((b<<6)&~(FILE_H|(FILE_H>>1)));
        atk|=((b>>17)&~FILE_H);atk|=((b>>15)&~FILE_A);
        atk|=((b>>10)&~(FILE_H|(FILE_H>>1)));atk|=((b>>6)&~(FILE_A|(FILE_A<<1)));
        knightAttacks[sq]=atk;
    }
    for(int sq=0;sq<64;sq++){
        U64 b=1ULL<<sq,atk=0;
        atk|=(b<<8);atk|=(b>>8);
        atk|=((b<<1)&~FILE_A);atk|=((b>>1)&~FILE_H);
        atk|=((b<<9)&~FILE_A);atk|=((b<<7)&~FILE_H);
        atk|=((b>>7)&~FILE_A);atk|=((b>>9)&~FILE_H);
        kingAttacks[sq]=atk;
    }
    for(int sq=0;sq<64;sq++){
        bishopMasks[sq]=bishopMaskGen(sq); rookMasks[sq]=rookMaskGen(sq);
        bishopBits[sq]=BISHOP_BITS_TABLE[sq]; rookBits[sq]=ROOK_BITS_TABLE[sq];
        bishopMagics[sq]=BISHOP_MAGIC[sq];    rookMagics[sq]=ROOK_MAGIC[sq];
        int bOcc=1<<bishopBits[sq];
        for(int i=0;i<bOcc;i++){
            U64 occ=setOccupancy(i,bishopBits[sq],bishopMasks[sq]);
            int idx=(int)((occ*bishopMagics[sq])>>(64-bishopBits[sq]));
            bishopAttacks[sq][idx]=bishopAttackOTF(sq,occ);
        }
        int rOcc=1<<rookBits[sq];
        for(int i=0;i<rOcc;i++){
            U64 occ=setOccupancy(i,rookBits[sq],rookMasks[sq]);
            int idx=(int)((occ*rookMagics[sq])>>(64-rookBits[sq]));
            rookAttacks[sq][idx]=rookAttackOTF(sq,occ);
        }
    }
}

inline U64 getBishopAttacks(int sq,U64 occ){ return bishopAttackOTF(sq,occ); }
inline U64 getRookAttacks(int sq,U64 occ)  { return rookAttackOTF(sq,occ);   }
inline U64 getQueenAttacks(int sq,U64 occ) { return getBishopAttacks(sq,occ)|getRookAttacks(sq,occ); }

// ============================================================
//  BOARD
// ============================================================
static const char PIECE_CHARS[2][7]={{'P','N','B','R','Q','K','.'},{'p','n','b','r','q','k','.'}};

void Board::reset(){
    memset(pieces,0,sizeof(pieces)); memset(occupancy,0,sizeof(occupancy));
    for(int i=0;i<64;i++){pieceOn[i]=NO_PIECE;colorOn[i]=BOTH;}
    side=WHITE;enPassant=NO_SQ;castling=0;halfMove=0;fullMove=1;
}

void Board::setFromFEN(const std::string &fen){
    reset();
    std::istringstream ss(fen);
    std::string board,sideStr,castleStr,epStr;
    ss>>board>>sideStr>>castleStr>>epStr>>halfMove>>fullMove;
    int sq=56;
    for(char c:board){
        if(c=='/')sq-=16;
        else if(c>='1'&&c<='8')sq+=c-'0';
        else{
            int col=(c>='a'&&c<='z')?BLACK:WHITE;
            char lc=tolower(c);
            int pt=(lc=='p')?PAWN:(lc=='n')?KNIGHT:(lc=='b')?BISHOP:(lc=='r')?ROOK:(lc=='q')?QUEEN:KING;
            setBit(pieces[col][pt],sq); pieceOn[sq]=pt; colorOn[sq]=col; sq++;
        }
    }
    recalcOccupancy();
    side=(sideStr=="w")?WHITE:BLACK;
    castling=0;
    for(char c:castleStr){if(c=='K')castling|=WK;if(c=='Q')castling|=WQ;if(c=='k')castling|=BK;if(c=='q')castling|=BQ;}
    if(epStr!="-"&&epStr.length()>=2){enPassant=(epStr[1]-'1')*8+(epStr[0]-'a');}
}

std::string Board::toFEN() const {
    std::string fen;
    for(int r=7;r>=0;r--){
        int empty=0;
        for(int f=0;f<8;f++){
            int sq=r*8+f;
            if(pieceOn[sq]==NO_PIECE)empty++;
            else{
                if(empty>0){fen+=char('0'+empty);empty=0;}
                fen+=PIECE_CHARS[colorOn[sq]==BLACK?1:0][pieceOn[sq]];
            }
        }
        if(empty>0)fen+=char('0'+empty);
        if(r>0)fen+='/';
    }
    fen+=(side==WHITE)?" w ":" b ";
    std::string c;
    if(castling&WK)c+='K';if(castling&WQ)c+='Q';if(castling&BK)c+='k';if(castling&BQ)c+='q';
    fen+=c.empty()?"-":c; fen+=' ';
    if(enPassant==NO_SQ)fen+="-"; else fen+=squareName(enPassant);
    fen+=' '+std::to_string(halfMove)+' '+std::to_string(fullMove);
    return fen;
}

void Board::print() const {
    std::cout<<"\n  +---+---+---+---+---+---+---+---+\n";
    for(int r=7;r>=0;r--){
        std::cout<<r+1<<" |";
        for(int f=0;f<8;f++){int sq=r*8+f;char c=(pieceOn[sq]==NO_PIECE)?'.':PIECE_CHARS[colorOn[sq]==BLACK?1:0][pieceOn[sq]];std::cout<<" "<<c<<" |";}
        std::cout<<"\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout<<"    a   b   c   d   e   f   g   h\n\n";
    std::cout<<"Side: "<<(side==WHITE?"White":"Black")<<"  EP: "<<(enPassant==NO_SQ?std::string("-"):squareName(enPassant))<<"  Castle: "<<(int)castling<<"\n\n";
}

void Board::recalcOccupancy(){
    occupancy[WHITE]=occupancy[BLACK]=0;
    for(int c=0;c<2;c++) for(int p=0;p<6;p++) occupancy[c]|=pieces[c][p];
    occupancy[BOTH]=occupancy[WHITE]|occupancy[BLACK];
}

bool Board::isSquareAttacked(int sq,int by) const {
    if(pawnAttacks[by^1][sq]&pieces[by][PAWN]) return true;
    if(knightAttacks[sq]&pieces[by][KNIGHT])   return true;
    if(kingAttacks[sq]  &pieces[by][KING])     return true;
    if(getBishopAttacks(sq,occupancy[BOTH])&(pieces[by][BISHOP]|pieces[by][QUEEN])) return true;
    if(getRookAttacks(sq,occupancy[BOTH])  &(pieces[by][ROOK]  |pieces[by][QUEEN])) return true;
    return false;
}

static const int castleMask[64]={
    13,15,15,15,12,15,15,14,15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15,15,15,15,15,15,15,15, 7,15,15,15, 3,15,15,11
};

bool Board::makeMove(Move mv,UndoInfo &undo){
    undo.save(*this,mv,NO_PIECE);
    int from=moveFrom(mv),to=moveTo(mv),col=side,opp=col^1,pt=pieceOn[from];
    if(from<0||from>63||to<0||to>63||pt==NO_PIECE||colorOn[from]!=col){return false;}
    if(moveCapture(mv)&&!moveEP(mv)){
        undo.capturedPiece=pieceOn[to];
        clearBit(pieces[opp][pieceOn[to]],to);
        pieceOn[to]=NO_PIECE;colorOn[to]=BOTH;halfMove=0;
    } else if(pt==PAWN)halfMove=0; else halfMove++;
    clearBit(pieces[col][pt],from); setBit(pieces[col][pt],to);
    pieceOn[to]=pt;colorOn[to]=col;pieceOn[from]=NO_PIECE;colorOn[from]=BOTH;
    enPassant=NO_SQ;
    if(moveEP(mv)){int c2=to+(col==WHITE?-8:8);undo.capturedPiece=PAWN;clearBit(pieces[opp][PAWN],c2);pieceOn[c2]=NO_PIECE;colorOn[c2]=BOTH;}
    if(moveDP(mv)) enPassant=to+(col==WHITE?-8:8);
    if(movePromo(mv)){int p=movePromo(mv);clearBit(pieces[col][PAWN],to);setBit(pieces[col][p],to);pieceOn[to]=p;}
    if(moveCastle(mv)){
        int rf,rt;
        if(to==G1){rf=H1;rt=F1;}else if(to==C1){rf=A1;rt=D1;}else if(to==G8){rf=H8;rt=F8;}else{rf=A8;rt=D8;}
        clearBit(pieces[col][ROOK],rf);setBit(pieces[col][ROOK],rt);
        pieceOn[rt]=ROOK;colorOn[rt]=col;pieceOn[rf]=NO_PIECE;colorOn[rf]=BOTH;
    }
    castling&=castleMask[from]&castleMask[to];
    side=opp; if(side==WHITE)fullMove++;
    recalcOccupancy();
    U64 kb=pieces[col][KING];
    if(!kb){undo.restore(*this);return false;}
    if(isSquareAttacked(lsb(kb),side)){undo.restore(*this);return false;}
    return true;
}

void Board::unmakeMove(Move m,UndoInfo &undo){undo.restore(*this);}

// ============================================================
//  MOVE GENERATION
// ============================================================
static void addPawnMoves(const Board &b,MoveList &ml,int from,int to,bool cap){
    int pr=(b.side==WHITE)?7:0;
    if(to/8==pr){for(int p=1;p<=4;p++)ml.add(encodeMove(from,to,p,cap?1:0,0,0,0));}
    else ml.add(encodeMove(from,to,0,cap?1:0,0,0,0));
}

void generateMoves(const Board &b,MoveList &ml){
    ml.clear();
    int col=b.side,opp=col^1;
    U64 enemies=b.occupancy[opp],empty=~b.occupancy[BOTH];
    U64 pawns=b.pieces[col][PAWN];
    while(pawns){
        int sq=popLSBIdx(pawns),rank=sq/8,file=sq%8,dir=(col==WHITE)?8:-8;
        int ps=sq+dir;
        if(ps>=0&&ps<64&&(empty&(1ULL<<ps))){
            addPawnMoves(b,ml,sq,ps,false);
            int sr=(col==WHITE)?1:6;
            if(rank==sr){int dp=ps+dir;if(dp>=0&&dp<64&&(empty&(1ULL<<dp)))ml.add(encodeMove(sq,dp,0,0,1,0,0));}
        }
        if(col==WHITE){
            if(file>0){int c2=sq+7;if(b.pieceOn[c2]!=NO_PIECE&&b.colorOn[c2]==opp)addPawnMoves(b,ml,sq,c2,true);}
            if(file<7){int c2=sq+9;if(b.pieceOn[c2]!=NO_PIECE&&b.colorOn[c2]==opp)addPawnMoves(b,ml,sq,c2,true);}
        } else {
            if(file>0){int c2=sq-9;if(c2>=0&&b.pieceOn[c2]!=NO_PIECE&&b.colorOn[c2]==opp)addPawnMoves(b,ml,sq,c2,true);}
            if(file<7){int c2=sq-7;if(c2>=0&&b.pieceOn[c2]!=NO_PIECE&&b.colorOn[c2]==opp)addPawnMoves(b,ml,sq,c2,true);}
        }
        if(b.enPassant!=NO_SQ){
            int epf=b.enPassant%8,epr=(col==WHITE)?4:3;
            if(rank==epr&&abs(epf-file)==1) ml.add(encodeMove(sq,b.enPassant,0,1,0,1,0));
        }
    }
    U64 knights=b.pieces[col][KNIGHT];
    while(knights){int sq=popLSBIdx(knights);U64 atk=knightAttacks[sq]&~b.occupancy[col];while(atk){int to=popLSBIdx(atk);ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));}}
    U64 bishops=b.pieces[col][BISHOP];
    while(bishops){int sq=popLSBIdx(bishops);U64 atk=getBishopAttacks(sq,b.occupancy[BOTH])&~b.occupancy[col];while(atk){int to=popLSBIdx(atk);ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));}}
    U64 rooks=b.pieces[col][ROOK];
    while(rooks){int sq=popLSBIdx(rooks);U64 atk=getRookAttacks(sq,b.occupancy[BOTH])&~b.occupancy[col];while(atk){int to=popLSBIdx(atk);ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));}}
    U64 queens=b.pieces[col][QUEEN];
    while(queens){int sq=popLSBIdx(queens);U64 atk=getQueenAttacks(sq,b.occupancy[BOTH])&~b.occupancy[col];while(atk){int to=popLSBIdx(atk);ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));}}
    if(b.pieces[col][KING]){
        int sq=lsb(b.pieces[col][KING]);
        U64 atk=kingAttacks[sq]&~b.occupancy[col];
        while(atk){int to=popLSBIdx(atk);ml.add(encodeMove(sq,to,0,getBit(enemies,to)?1:0,0,0,0));}
        if(col==WHITE){
            if((b.castling&WK)&&!getBit(b.occupancy[BOTH],F1)&&!getBit(b.occupancy[BOTH],G1)&&!b.isSquareAttacked(E1,BLACK)&&!b.isSquareAttacked(F1,BLACK)&&!b.isSquareAttacked(G1,BLACK)) ml.add(encodeMove(E1,G1,0,0,0,0,1));
            if((b.castling&WQ)&&!getBit(b.occupancy[BOTH],D1)&&!getBit(b.occupancy[BOTH],C1)&&!getBit(b.occupancy[BOTH],B1)&&!b.isSquareAttacked(E1,BLACK)&&!b.isSquareAttacked(D1,BLACK)&&!b.isSquareAttacked(C1,BLACK)) ml.add(encodeMove(E1,C1,0,0,0,0,1));
        } else {
            if((b.castling&BK)&&!getBit(b.occupancy[BOTH],F8)&&!getBit(b.occupancy[BOTH],G8)&&!b.isSquareAttacked(E8,WHITE)&&!b.isSquareAttacked(F8,WHITE)&&!b.isSquareAttacked(G8,WHITE)) ml.add(encodeMove(E8,G8,0,0,0,0,1));
            if((b.castling&BQ)&&!getBit(b.occupancy[BOTH],D8)&&!getBit(b.occupancy[BOTH],C8)&&!getBit(b.occupancy[BOTH],B8)&&!b.isSquareAttacked(E8,WHITE)&&!b.isSquareAttacked(D8,WHITE)&&!b.isSquareAttacked(C8,WHITE)) ml.add(encodeMove(E8,C8,0,0,0,0,1));
        }
    }
}

// ============================================================
//  EVALUATION
// ============================================================
static const int MATERIAL[6]={100,320,330,500,900,20000};
static const int SEE_VAL[7] = {100,320,330,500,900,20000,0}; // for SEE
static const int MVV_LVA[6][6]={{15,14,13,12,11,10},{25,24,23,22,21,20},{35,34,33,32,31,30},{45,44,43,42,41,40},{55,54,53,52,51,50},{0,0,0,0,0,0}};

static const int PST_PAWN[64]={0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0};
static const int PST_KNIGHT[64]={-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50};
static const int PST_BISHOP[64]={-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20};
static const int PST_ROOK[64]={0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0};
static const int PST_QUEEN[64]={-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20};
static const int PST_KING_MG[64]={
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};
// King endgame PST: king should go to center and be ACTIVE
static const int PST_KING_EG[64]={
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
};
static const int* PST[6]={PST_PAWN,PST_KNIGHT,PST_BISHOP,PST_ROOK,PST_QUEEN,PST_KING_MG};

// ============================================================
//  NNUE EVALUATE — Self-contained, uses nnue_weights.h directly
//  Architecture: 768 → 256(CReLU) → 32(CReLU) → 32(CReLU) → 1
//  Returns score in centipawns from side-to-move perspective
// ============================================================
static inline float crelu(float x){ return x<0.f?0.f:(x>1.f?1.f:x); }

static int nnue_evaluate(const Board &b) {
#if !HAS_NNUE
    // NNUE weights not loaded — return 0 (HCE only)
    return 0;
#else
    // ── Build 768-float feature vector ───────────────────────
    float inp[768] = {};
    for(int sq=0;sq<64;sq++){
        int pt=b.pieceOn[sq], col=b.colorOn[sq];
        if(pt==NO_PIECE||col==BOTH) continue;
        int idx=col*384+pt*64+sq;
        if(idx>=0&&idx<768) inp[idx]=1.f;
    }

    // ── Layer 1: 768 → 256, two perspectives ─────────────────
    float l1w[256], l1b_arr[256]; // white and black perspectives
    for(int j=0;j<256;j++){
        float sw=NNUE::L1_B[j], sb=NNUE::L1_B[j];
        for(int i=0;i<768;i++){
            sw+=inp[i]*NNUE::L1_W[i][j];
            // Black perspective: swap color blocks (0..383 ↔ 384..767)
            int mi=(i<384)?i+384:i-384;
            sb+=inp[mi]*NNUE::L1_W[i][j];
        }
        l1w[j]=crelu(sw);
        l1b_arr[j]=crelu(sb);
    }

    // Concatenate: STM perspective first
    float l1[512];
    if(b.side==WHITE){
        memcpy(l1,     l1w,     256*sizeof(float));
        memcpy(l1+256, l1b_arr, 256*sizeof(float));
    } else {
        memcpy(l1,     l1b_arr, 256*sizeof(float));
        memcpy(l1+256, l1w,     256*sizeof(float));
    }

    // ── Layer 2: 512 → 32 ────────────────────────────────────
    float l2[32];
    for(int j=0;j<32;j++){
        float s=NNUE::L2_B[j];
        for(int i=0;i<512;i++) s+=l1[i]*NNUE::L2_W[i][j];
        l2[j]=crelu(s);
    }

    // ── Layer 3: 32 → 32 ─────────────────────────────────────
    float l3[32];
    for(int j=0;j<32;j++){
        float s=NNUE::L3_B[j];
        for(int i=0;i<32;i++) s+=l2[i]*NNUE::L3_W[i][j];
        l3[j]=crelu(s);
    }

    // ── Output: 32 → 1 ───────────────────────────────────────
    float out=NNUE::OUT_B;
    for(int i=0;i<32;i++) out+=l3[i]*NNUE::OUT_W[i];

    // Scale: trained with SCORE_SCALE=600
    return (int)(out*600.f);
#endif // HAS_NNUE
}


// ============================================================
//  EVALUATE — Mobility + Pawn Structure + King Safety
// ============================================================

// Mobility bonus per extra reachable square (tuned values)
static const int KNIGHT_MOB[9] = {-62,-40,-20,  0, 12, 22, 30, 36, 40};
static const int BISHOP_MOB[14]= {-48,-28,-10,  4, 15, 24, 32, 38, 43, 47, 50, 52, 54, 55};
static const int ROOK_MOB[15]  = {-60,-40,-20, -4,  8, 18, 26, 33, 39, 44, 48, 51, 53, 54, 55};
static const int QUEEN_MOB[28] = {-30,-20,-10, -2,  4,  9, 14, 18, 22, 25, 28, 30, 32, 34,
                                    36, 37, 38, 39, 40, 41, 41, 42, 42, 42, 42, 42, 42, 42};

// Passed pawn bonus by rank (rank 1=0 not reachable, rank 7=promotion)
static const int PASSED_BONUS[8] = {0, 5, 10, 20, 35, 55, 80, 0};

// King safety: attacker weights per piece type
static const int ATTACK_WEIGHT[6] = {0, 20, 20, 40, 80, 0};

// Manhattan distance between two squares
static inline int manhattanDist(int a, int b2){
    return abs(a%8 - b2%8) + abs(a/8 - b2/8);
}
// Chebyshev distance (king moves)
static inline int chebyshevDist(int a, int b2){
    return std::max(abs(a%8 - b2%8), abs(a/8 - b2/8));
}

int evaluate(const Board &b) {
    int score = 0;

    // ── Phase detection ────────────────────────────────────────
    // Count non-pawn material to determine game phase
    // Middlegame = 256, Endgame = 0 (Stockfish-style tapering)
    static const int PHASE_WEIGHT[6]={0,1,1,2,4,0};
    int phase=0;
    for(int c=0;c<2;c++)
        for(int p=KNIGHT;p<=QUEEN;p++)
            phase+=PHASE_WEIGHT[p]*popcount(b.pieces[c][p]);
    phase=std::min(phase,24); // max 24 (both sides full pieces)
    // phase=24: middlegame, phase=0: endgame
    bool isEndgame = (phase <= 10);  // endgame threshold
    bool isPawnEndgame = (phase <= 2); // only kings + pawns

    // Precompute pawn bitboards for pawn structure
    U64 wPawns = b.pieces[WHITE][PAWN];
    U64 bPawns = b.pieces[BLACK][PAWN];
    U64 allOcc = b.occupancy[BOTH];

    // File masks (precomputed inline)
    // For each file f: all squares on file f
    // We build pawn file sets on the fly

    // ── Pawn structure (both sides together for efficiency) ──
    for(int col = 0; col < 2; col++) {
        int sign   = (col == WHITE) ? 1 : -1;
        int opp    = col ^ 1;
        U64 myPawns  = b.pieces[col][PAWN];
        U64 oppPawns = b.pieces[opp][PAWN];

        U64 tmp = myPawns;
        while(tmp) {
            int sq   = popLSBIdx(tmp);
            int rank = sq / 8;
            int file = sq % 8;

            // ── Doubled pawn penalty ──────────────────────────
            // Another own pawn on same file?
            U64 fileMask = 0;
            for(int r=0;r<8;r++) setBit(fileMask, r*8+file);
            int pawnsOnFile = popcount(myPawns & fileMask);
            if(pawnsOnFile > 1)
                score -= sign * 15;  // doubled pawn

            // ── Isolated pawn penalty ─────────────────────────
            // No own pawn on adjacent files?
            bool isolated = true;
            if(file > 0) {
                U64 leftFile = 0;
                for(int r=0;r<8;r++) setBit(leftFile, r*8+file-1);
                if(myPawns & leftFile) isolated = false;
            }
            if(file < 7) {
                U64 rightFile = 0;
                for(int r=0;r<8;r++) setBit(rightFile, r*8+file+1);
                if(myPawns & rightFile) isolated = false;
            }
            if(isolated)
                score -= sign * 20;  // isolated pawn

            // ── Passed pawn bonus ─────────────────────────────
            // No enemy pawn on same or adjacent files ahead
            bool passed = true;
            for(int adjF = std::max(0,file-1); adjF <= std::min(7,file+1); adjF++) {
                U64 aheadMask = 0;
                if(col == WHITE) {
                    for(int r=rank+1;r<8;r++) setBit(aheadMask, r*8+adjF);
                } else {
                    for(int r=0;r<rank;r++) setBit(aheadMask, r*8+adjF);
                }
                if(oppPawns & aheadMask) { passed = false; break; }
            }
            if(passed) {
                int prank = (col == WHITE) ? rank : (7 - rank);
                // Endgame: passed pawns are MUCH more valuable
                int passedVal = PASSED_BONUS[prank];
                if(isEndgame) passedVal = passedVal * 2;
                score += sign * passedVal;

                // ── King proximity to passed pawn ─────────────
                if(isEndgame) {
                    int promSq = (col==WHITE) ? (7*8+file) : file;
                    // Our king close to pawn = bonus
                    int myKSq = b.pieces[col][KING] ? lsb(b.pieces[col][KING]) : 0;
                    int oppKSq = b.pieces[col^1][KING] ? lsb(b.pieces[col^1][KING]) : 0;
                    int myDist  = chebyshevDist(myKSq, promSq);
                    int oppDist = chebyshevDist(oppKSq, promSq);
                    // Bonus if we're closer to promotion square
                    score += sign * (oppDist - myDist) * 5;

                    // ── Rule of the Square ─────────────────────
                    // Can opponent king catch the pawn?
                    // If not, pawn promotes for free!
                    if(b.pieces[col^1][QUEEN]==0 && b.pieces[col^1][ROOK]==0) {
                        int pawnDist = (col==WHITE) ? (7-rank) : rank;
                        // If side to move, pawn gets one extra step
                        if(b.side == col) pawnDist--;
                        if(oppDist > pawnDist) {
                            // Pawn promotes! Give huge bonus
                            score += sign * 400;
                        }
                    }
                }
            }

            // ── Connected pawn bonus ──────────────────────────
            // Pawn defended by another own pawn
            if(col == WHITE) {
                if(file>0 && rank>0 && getBit(myPawns,(rank-1)*8+file-1)) score += sign * 8;
                if(file<7 && rank>0 && getBit(myPawns,(rank-1)*8+file+1)) score += sign * 8;
            } else {
                if(file>0 && rank<7 && getBit(myPawns,(rank+1)*8+file-1)) score += sign * 8;
                if(file<7 && rank<7 && getBit(myPawns,(rank+1)*8+file+1)) score += sign * 8;
            }
        }
    }

    // ── Main piece loop ────────────────────────────────────────
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp  = col ^ 1;

        // King square for king safety
        int kingSq = b.pieces[col][KING] ? lsb(b.pieces[col][KING]) : 0;
        int oppKingSq = b.pieces[opp][KING] ? lsb(b.pieces[opp][KING]) : 0;

        // King safety: count attackers near opponent king
        int kingAttackScore = 0;
        int kingAttackers   = 0;

        // King zone: squares around opponent king
        U64 kingZone = kingAttacks[oppKingSq] | (1ULL << oppKingSq);

        // ── Material + PST (phase-tapered) ──────────────────
        for(int pt = 0; pt < 6; pt++) {
            U64 bb = b.pieces[col][pt];
            while(bb) {
                int sq = popLSBIdx(bb);
                int pstSq = (col == WHITE) ? sq : (sq ^ 56);
                int pstVal;
                if(pt == KING) {
                    // Taper king PST: MG → EG smoothly
                    int mg = PST_KING_MG[pstSq];
                    int eg = PST_KING_EG[pstSq];
                    pstVal = (mg * phase + eg * (24 - phase)) / 24;
                } else {
                    pstVal = PST[pt][pstSq];
                }
                score += sign * (MATERIAL[pt] + pstVal);
            }
        }

        // ── Knight mobility ───────────────────────────────────
        U64 knights = b.pieces[col][KNIGHT];
        while(knights) {
            int sq = popLSBIdx(knights);
            // Mobility: squares attacked, excluding own pieces
            U64 atk = knightAttacks[sq] & ~b.occupancy[col];
            int mob = popcount(atk);
            score += sign * KNIGHT_MOB[std::min(mob, 8)];
            // King safety contribution
            U64 kAtk = atk & kingZone;
            if(kAtk) { kingAttackScore += ATTACK_WEIGHT[KNIGHT]; kingAttackers++; }
        }

        // ── Bishop mobility + pair ────────────────────────────
        U64 bishops = b.pieces[col][BISHOP];
        int bishopCount = popcount(bishops);
        if(bishopCount >= 2) score += sign * 40;  // bishop pair

        while(bishops) {
            int sq = popLSBIdx(bishops);
            U64 atk = getBishopAttacks(sq, allOcc) & ~b.occupancy[col];
            int mob = popcount(atk);
            score += sign * BISHOP_MOB[std::min(mob, 13)];
            U64 kAtk = atk & kingZone;
            if(kAtk) { kingAttackScore += ATTACK_WEIGHT[BISHOP]; kingAttackers++; }
        }

        // ── Rook mobility + open file ─────────────────────────
        U64 rooks = b.pieces[col][ROOK];
        while(rooks) {
            int sq   = popLSBIdx(rooks);
            int file = sq % 8;
            U64 atk  = getRookAttacks(sq, allOcc) & ~b.occupancy[col];
            int mob  = popcount(atk);
            score += sign * ROOK_MOB[std::min(mob, 14)];

            // Open / semi-open file
            U64 fileMask = 0;
            for(int r=0;r<8;r++) setBit(fileMask, r*8+file);
            bool noOwnPawn  = !(b.pieces[col][PAWN] & fileMask);
            bool noOppPawn  = !(b.pieces[opp][PAWN] & fileMask);
            if(noOwnPawn && noOppPawn) score += sign * 30;  // open file
            else if(noOwnPawn)         score += sign * 15;  // semi-open

            // Rook on 7th rank (enemy pawns on 7th = big bonus)
            int rank7 = (col == WHITE) ? 6 : 1;
            if(sq/8 == rank7) score += sign * 20;

            U64 kAtk = atk & kingZone;
            if(kAtk) { kingAttackScore += ATTACK_WEIGHT[ROOK]; kingAttackers++; }
        }

        // ── Queen mobility ────────────────────────────────────
        U64 queens = b.pieces[col][QUEEN];
        while(queens) {
            int sq  = popLSBIdx(queens);
            U64 atk = getQueenAttacks(sq, allOcc) & ~b.occupancy[col];
            int mob = popcount(atk);
            score += sign * QUEEN_MOB[std::min(mob, 27)];
            U64 kAtk = atk & kingZone;
            if(kAtk) { kingAttackScore += ATTACK_WEIGHT[QUEEN]; kingAttackers++; }
        }

        // ── King safety ───────────────────────────────────────
        // Subtract from score: opponent attacking our king
        // More attackers = exponentially more dangerous
        if(kingAttackers >= 2) {
            int danger = kingAttackScore * kingAttackers / 4;
            score -= sign * danger;
        }

        // Pawn shield for our own king
        int kfile = kingSq % 8;
        int krank = kingSq / 8;
        int shield = 0;
        U64 myPawns = b.pieces[col][PAWN];
        U64 tmp2 = myPawns;
        while(tmp2) {
            int psq = popLSBIdx(tmp2);
            int pf  = psq % 8;
            int pr  = psq / 8;
            if(abs(pf - kfile) <= 1) {
                if(col == WHITE && pr > krank && pr <= krank+2) shield++;
                if(col == BLACK && pr < krank && pr >= krank-2) shield++;
            }
        }
        score += sign * shield * 12;
    }

    // ── Endgame specific evaluations ─────────────────────────
    if(isEndgame) {
        int wKing = b.pieces[WHITE][KING] ? lsb(b.pieces[WHITE][KING]) : 0;
        int bKing = b.pieces[BLACK][KING] ? lsb(b.pieces[BLACK][KING]) : 0;

        // ── Mop-up evaluation ─────────────────────────────────
        // Winning side: push enemy king to corner + bring own king close
        // This is how engines win KRvK, KQvK, etc.
        int materialScore = 0;
        for(int p=PAWN;p<=QUEEN;p++)
            materialScore += MATERIAL[p]*(popcount(b.pieces[WHITE][p])
                                        - popcount(b.pieces[BLACK][p]));

        if(std::abs(materialScore) > 200) {
            int winningSide = (materialScore > 0) ? WHITE : BLACK;
            int losingSide  = winningSide ^ 1;
            int winSign     = (winningSide == WHITE) ? 1 : -1;
            int loserKing   = (losingSide == WHITE) ? wKing : bKing;
            int winnerKing  = (winningSide == WHITE) ? wKing : bKing;

            // Push loser king to corner
            // Corner squares: a1,a8,h1,h8
            // Manhattan distance from center d4/d5/e4/e5
            int loserFile = loserKing % 8;
            int loserRank = loserKing / 8;
            int cornerPush = (std::max(3-loserFile, loserFile-4) +
                              std::max(3-loserRank, loserRank-4));
            score += winSign * cornerPush * 10;

            // Winner king close to loser king
            int kingDist = chebyshevDist(winnerKing, loserKing);
            score += winSign * (7 - kingDist) * 8;

            // Scale mop-up by winning margin
            int mopScale = std::min(std::abs(materialScore), 900) / 100;
            score = score * mopScale / 5;
        }

        // ── Pawn endgame: tempo + opposition ─────────────────
        if(isPawnEndgame) {
            // Side to move bonus (tempo)
            score += (b.side == WHITE) ? 15 : -15;

            // King opposition: kings directly facing each other
            int fileDiff = abs(wKing%8 - bKing%8);
            int rankDiff = abs(wKing/8 - bKing/8);
            if(fileDiff==0 && rankDiff==2) {
                // Direct opposition - side NOT to move has it
                score += (b.side==WHITE) ? -30 : 30;
            } else if(fileDiff==0 && rankDiff==1) {
                score += (b.side==WHITE) ? -20 : 20;
            }
        }

        // ── Rook endgame: 7th rank bonus ─────────────────────
        if(b.pieces[WHITE][ROOK] || b.pieces[BLACK][ROOK]) {
            // Extra bonus for rook cutting off king
            for(int col=0;col<2;col++){
                int sign2=(col==WHITE)?1:-1;
                int oppKingSq2=(col==WHITE)?bKing:wKing;
                int oppKingRank=oppKingSq2/8;
                U64 rooks2=b.pieces[col][ROOK];
                while(rooks2){
                    int rsq=popLSBIdx(rooks2);
                    int rRank=rsq/8;
                    // Rook cuts off king on rank
                    if(col==WHITE && rRank>oppKingRank && rRank>=5)
                        score+=sign2*15;
                    if(col==BLACK && rRank<oppKingRank && rRank<=2)
                        score+=sign2*15;
                }
            }
        }

        // ── Minor piece endgame: bishop better than knight ───
        // (in open endgames with pawns on both wings)
        int wPawnFiles=0,bPawnFiles=0;
        for(int f=0;f<8;f++){
            U64 fm=0;for(int r=0;r<8;r++)setBit(fm,r*8+f);
            if(wPawns&fm)wPawnFiles++;
            if(bPawns&fm)bPawnFiles++;
        }
        if(wPawnFiles>=4 || bPawnFiles>=4){
            score += popcount(b.pieces[WHITE][BISHOP])*10;
            score -= popcount(b.pieces[BLACK][BISHOP])*10;
        }
    }

    // ── Hanging Piece Penalty ─────────────────────────────────
    // A piece is "hanging" if it is attacked by the opponent
    // and either undefended OR defended by a less valuable piece
    // Stockfish calls this "threat" detection
    for(int col = 0; col < 2; col++) {
        int sign = (col == WHITE) ? 1 : -1;
        int opp  = col ^ 1;

        // For each non-pawn, non-king piece
        for(int pt = KNIGHT; pt <= QUEEN; pt++) {
            U64 bb = b.pieces[col][pt];
            while(bb) {
                int sq = popLSBIdx(bb);

                // Is this square attacked by the opponent?
                if(!b.isSquareAttacked(sq, opp)) continue;

                // Is it defended by us?
                bool defended = b.isSquareAttacked(sq, col);

                if(!defended) {
                    // Completely undefended piece under attack = HANGING
                    // Penalty: full piece value (it will be captured for free)
                    score -= sign * SEE_VAL[pt] / 2;
                } else {
                    // Defended, but check if cheapest attacker < piece value
                    // (e.g. pawn attacks our bishop = losing exchange)
                    // Quick check: is there a pawn attacking it?
                    if(pt >= BISHOP) {
                        U64 pawnatk = pawnAttacks[col][sq] & b.pieces[opp][PAWN];
                        if(pawnatk) {
                            // Pawn attacks a bishop/rook/queen = bad trade
                            score -= sign * (SEE_VAL[pt] - SEE_VAL[PAWN]) / 4;
                        }
                    }
                    if(pt >= ROOK) {
                        U64 knatk = knightAttacks[sq] &
                                    (b.pieces[opp][KNIGHT]|b.pieces[opp][BISHOP]);
                        if(knatk) {
                            // Knight/bishop attacks rook/queen
                            score -= sign * (SEE_VAL[pt] - SEE_VAL[KNIGHT]) / 4;
                        }
                    }
                }
            }
        }
    }

    // ── Space advantage ──────────────────────────────────────
    // Count squares behind pawns that we control
    // More space = better piece mobility long term
    if(phase >= 12) {
        for(int col=0;col<2;col++){
            int sg=(col==WHITE)?1:-1;
            U64 spaceMask=(col==WHITE)?
                0x3C3C3C000000ULL:   // ranks 3-6, files c-f for white
                0x3C3C3C0000ULL;     // ranks 3-6, files c-f for black
            U64 blocked=b.pieces[col][PAWN]&(col==WHITE?
                (b.occupancy[BOTH]>>8):(b.occupancy[BOTH]<<8));
            U64 space=spaceMask&~b.occupancy[col]&~pawnAttacks[col^1][0]; // rough
            // Count controlled space
            U64 pawns2=b.pieces[col][PAWN];
            int spaceScore=0;
            while(pawns2){
                int sq=popLSBIdx(pawns2);
                // Squares behind this pawn that are safe
                if(col==WHITE){for(int r=1;r<sq/8;r++) if(!getBit(b.occupancy[BOTH],r*8+sq%8))spaceScore++;}
                else{for(int r=sq/8+1;r<8;r++) if(!getBit(b.occupancy[BOTH],r*8+sq%8))spaceScore++;}
            }
            score+=sg*spaceScore;
        }
    }

    // ── Outpost bonus ─────────────────────────────────────────
    // Knight/Bishop on square that can't be attacked by enemy pawns
    for(int col=0;col<2;col++){
        int sg=(col==WHITE)?1:-1, opp=col^1;
        // Knight outposts
        U64 kn=b.pieces[col][KNIGHT];
        while(kn){
            int sq=popLSBIdx(kn);
            int rank=sq/8, file=sq%8;
            // Outpost: ranks 4-6 for white, 3-5 for black
            int minRank=(col==WHITE)?3:2, maxRank=(col==WHITE)?5:4;
            if(rank<minRank||rank>maxRank) continue;
            // No enemy pawn can attack this square
            bool safe=true;
            U64 opPawns=b.pieces[opp][PAWN];
            while(opPawns){
                int psq=popLSBIdx(opPawns);
                int pr=psq/8, pf=psq%8;
                if(abs(pf-file)==1){
                    if(opp==WHITE&&pr<rank) {safe=false;break;}
                    if(opp==BLACK&&pr>rank) {safe=false;break;}
                }
            }
            if(safe){
                // Bonus: defended by own pawn = even better
                bool defended=(pawnAttacks[opp][sq]&b.pieces[col][PAWN])!=0;
                score+=sg*(defended?30:15);
            }
        }
    }

    // ── Weak squares penalty ──────────────────────────────────
    // Squares that can never be defended by own pawns
    for(int col=0;col<2;col++){
        int sg=(col==WHITE)?1:-1;
        U64 myPawns2=b.pieces[col][PAWN];
        // Check color complex weakness (simplified)
        int lightWeakness=0, darkWeakness=0;
        U64 tmp3=myPawns2;
        while(tmp3){
            int sq=popLSBIdx(tmp3);
            if((sq/8+sq%8)%2==0) lightWeakness++;
            else darkWeakness++;
        }
        int totalPawns=popcount(myPawns2);
        if(totalPawns>0){
            // More pawns on one color complex = weakness on other color
            int diff=abs(lightWeakness-darkWeakness);
            score-=sg*diff*3;
        }
    }

    // ── Tempo + initiative ────────────────────────────────────
    score += (b.side == WHITE) ? 10 : -10;

    // Pure HCE — NNUE disconnected for tuning
    return (b.side == WHITE) ? score : -score;
}

// ============================================================
//  PERFT
// ============================================================
unsigned long long perft(Board &b,int depth){
    if(depth==0)return 1;
    MoveList ml;generateMoves(b,ml);
    unsigned long long n=0;UndoInfo u;
    for(int i=0;i<ml.count;i++){if(!b.makeMove(ml.moves[i],u))continue;n+=perft(b,depth-1);b.unmakeMove(ml.moves[i],u);}
    return n;
}

void divide(Board &b,int depth){
    if(depth<=0){std::cout<<"Depth must be > 0\n";return;}
    MoveList ml;generateMoves(b,ml);
    std::cout<<"Generated "<<ml.count<<" pseudo-legal moves\n";
    UndoInfo u;unsigned long long total=0;int legal=0;
    for(int i=0;i<ml.count;i++){
        if(!b.makeMove(ml.moves[i],u))continue;
        legal++;unsigned long long n=perft(b,depth-1);b.unmakeMove(ml.moves[i],u);
        std::cout<<moveToStr(ml.moves[i])<<": "<<n<<"\n";total+=n;
    }
    std::cout<<"Legal moves: "<<legal<<"\nTotal: "<<total<<"\n";
}

// ============================================================
//  TRANSPOSITION TABLE
// ============================================================
static U64 ZP[2][6][64], ZSIDE, ZCASTLE[16], ZEP[8];
static bool zInited=false;
static U64 xrand(){static U64 s=0x123456789ABCDEFULL;s^=s>>12;s^=s<<25;s^=s>>27;return s*2685821657736338717ULL;}
static void initZobrist(){
    if(zInited)return;zInited=true;
    for(int c=0;c<2;c++)for(int p=0;p<6;p++)for(int s=0;s<64;s++)ZP[c][p][s]=xrand();
    ZSIDE=xrand();
    for(int i=0;i<16;i++)ZCASTLE[i]=xrand();
    for(int i=0;i<8;i++)ZEP[i]=xrand();
}
static U64 hashBoard(const Board &b){
    U64 h=0;
    for(int c=0;c<2;c++)for(int p=0;p<6;p++){U64 bb=b.pieces[c][p];while(bb){int sq=popLSBIdx(bb);h^=ZP[c][p][sq];}}
    if(b.side==BLACK)h^=ZSIDE;
    h^=ZCASTLE[b.castling&15];
    if(b.enPassant!=NO_SQ)h^=ZEP[b.enPassant%8];
    return h;
}

enum TTFlag{TT_EXACT=0,TT_ALPHA=1,TT_BETA=2};
struct TTEntry{U64 key;int score;Move bestMove;int16_t depth;int8_t flag;};
static const int TT_SIZE=1<<22; // ~128MB
static TTEntry TT[TT_SIZE];
static bool ttInited=false;
static void initTT(){if(ttInited)return;ttInited=true;memset(TT,0,sizeof(TT));}

static void ttStore(U64 key,int score,Move m,int depth,int flag){
    int i=(int)(key&(TT_SIZE-1));
    if(TT[i].key==0||depth>=TT[i].depth){TT[i]={key,score,m,(int16_t)depth,(int8_t)flag};}
}
static bool ttProbe(U64 key,int depth,int alpha,int beta,int &score,Move &bm){
    int i=(int)(key&(TT_SIZE-1));bm=0;
    if(TT[i].key!=key)return false;
    bm=TT[i].bestMove;
    if(TT[i].depth<depth)return false;
    score=TT[i].score;
    if(TT[i].flag==TT_EXACT)return true;
    if(TT[i].flag==TT_ALPHA&&score<=alpha){score=alpha;return true;}
    if(TT[i].flag==TT_BETA &&score>=beta) {score=beta; return true;}
    return false;
}

// ============================================================
//  SEE — Static Exchange Evaluation
//  Returns the material gain/loss of a capture sequence on sq
//  Stockfish-style: considers all attackers, uses least valuable
//  attacker first
// ============================================================

// Get all pieces attacking a square (both sides)
static U64 allAttackers(const Board &b, int sq, U64 occ) {
    return (pawnAttacks[WHITE][sq]  & b.pieces[BLACK][PAWN])
         | (pawnAttacks[BLACK][sq]  & b.pieces[WHITE][PAWN])
         | (knightAttacks[sq]       & (b.pieces[WHITE][KNIGHT]|b.pieces[BLACK][KNIGHT]))
         | (getBishopAttacks(sq,occ)& (b.pieces[WHITE][BISHOP]|b.pieces[BLACK][BISHOP]
                                      |b.pieces[WHITE][QUEEN] |b.pieces[BLACK][QUEEN]))
         | (getRookAttacks(sq,occ)  & (b.pieces[WHITE][ROOK]  |b.pieces[BLACK][ROOK]
                                      |b.pieces[WHITE][QUEEN] |b.pieces[BLACK][QUEEN]))
         | (kingAttacks[sq]         & (b.pieces[WHITE][KING]  |b.pieces[BLACK][KING]));
}

// Returns SEE value of moving to 'to' (positive = winning capture)
// threshold: minimum gain required (0 = any winning capture)
static bool seeGe(const Board &b, Move m, int threshold) {
    int from = moveFrom(m), to = moveTo(m);

    // Initial gain = value of captured piece
    int gain = SEE_VAL[moveEP(m) ? PAWN : b.pieceOn[to]] - threshold;
    if(gain < 0) return false;  // Even taking for free doesn't meet threshold

    // Attacker value
    int attackerPt = b.pieceOn[from];
    gain -= SEE_VAL[attackerPt];
    if(gain >= 0) return true;  // We're winning even if recaptured

    // Build occupancy after this move
    U64 occ = b.occupancy[BOTH];
    clearBit(occ, from);
    setBit(occ, to);

    // Get all attackers
    U64 attackers = allAttackers(b, to, occ);

    // Side that just moved
    int side = b.side ^ 1;  // opponent now attacks

    // SEE loop: each side picks least valuable attacker
    while(true) {
        side ^= 1;  // switch side
        attackers &= occ;  // remove captured pieces

        // Find least valuable attacker for current side
        U64 sideAttackers = attackers & b.occupancy[side];
        if(!sideAttackers) break;

        int pt;
        U64 candidates;
        for(pt = PAWN; pt <= KING; pt++) {
            candidates = sideAttackers & b.pieces[side][pt];
            if(candidates) break;
        }
        if(pt > KING) break;

        // "Make" the capture
        int sq = lsb(candidates);
        clearBit(occ, sq);

        // Add x-ray attackers (sliders behind captured piece)
        if(pt == PAWN || pt == BISHOP || pt == QUEEN)
            attackers |= (getBishopAttacks(to, occ) & 
                         (b.pieces[WHITE][BISHOP]|b.pieces[BLACK][BISHOP]|
                          b.pieces[WHITE][QUEEN] |b.pieces[BLACK][QUEEN]));
        if(pt == ROOK || pt == QUEEN)
            attackers |= (getRookAttacks(to, occ) &
                         (b.pieces[WHITE][ROOK]|b.pieces[BLACK][ROOK]|
                          b.pieces[WHITE][QUEEN]|b.pieces[BLACK][QUEEN]));

        gain = -gain - 1 - SEE_VAL[pt];
        if(gain >= 0) {
            // If we're about to use king and opponent still has attackers, we lose
            if(pt == KING && (attackers & b.occupancy[side^1]))
                side ^= 1;
            break;
        }
    }
    return b.side != side;  // we win if current side is not the one that "lost"
}

// Simple SEE score (not just threshold): returns material swing value
static int seeScore(const Board &b, int to, int fromPt, int capturedPt) {
    // Quick approximation: victim - attacker (if positive, winning capture)
    return SEE_VAL[capturedPt] - SEE_VAL[fromPt];
}


static const int INF=1000000,MATE=900000,MAX_PLY=64;
static Move killers[MAX_PLY][2];
static int  history[2][64][64];

// ── Continuation History (Stockfish: contHist) ───────────────
// contHist[piece][to][piece][to] = how good is (piece→to) after (prev_piece→prev_to)
// Indexed: [prev_pt*64+prev_to][pt*64+to], pieces 0-5, squares 0-63
// Simplified: contHist[prev_pt][prev_to][pt][to]
static int contHist[6][64][6][64];

// ── Counter Move Heuristic ────────────────────────────────────
// counterMove[prev_pt][prev_to] = best response to that move
static Move counterMove[6][64];

// ── Per-ply stack for continuation history ────────────────────
struct PlyInfo {
    Move   currentMove;
    int    movedPiece;   // piece type that moved
    int    movedTo;      // destination square
    Move   excludedMove; // for singular extensions
};
static PlyInfo plyStack[MAX_PLY+4];

static void clearTables(){
    memset(killers,     0, sizeof(killers));
    memset(history,     0, sizeof(history));
    memset(contHist,    0, sizeof(contHist));
    memset(counterMove, 0, sizeof(counterMove));
    memset(plyStack,    0, sizeof(plyStack));
}

static inline void checkTime(SearchInfo &info){
    if(info.timeLimit>0&&(info.nodes&2047)==0){
        auto e=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-info.startTime).count();
        if(e>=info.timeLimit)info.stop=true;
    }
}

static int scoreMove(const Board &b,Move m,int ply){
    int s=0;
    if(moveCapture(m)){
        int att=b.pieceOn[moveFrom(m)];
        int vic=moveEP(m)?PAWN:b.pieceOn[moveTo(m)];
        if(att!=NO_PIECE&&vic!=NO_PIECE){
            if(seeGe(b,m,1))       s = 10000 + MVV_LVA[vic][att]; // winning
            else if(seeGe(b,m,0))  s =  9000 + MVV_LVA[vic][att]; // equal
            else {
                // TAL SOUL: "Losing" captures near enemy king get a big boost!
                // Tal deliberately sacrificed near the enemy king to create chaos.
                // Normal engines bury these at 2000 — Tal would play them FIRST.
                int to = moveTo(m);
                int oppKingSq = b.pieces[b.side^1][KING]
                                ? lsb(b.pieces[b.side^1][KING]) : 0;
                U64 kingZoneTal = kingAttacks[oppKingSq] | (1ULL << oppKingSq);
                bool sacrificeNearKing = (kingZoneTal & (1ULL << to)) != 0;
                // Near enemy king: bump 2000 → 5500 (just below killer moves!)
                // Elsewhere: keep at 2000 (still explore but deprioritise)
                s = sacrificeNearKing
                    ? 5500 + MVV_LVA[vic][att]  // TAL SACRIFICE! 😈
                    : 2000 + MVV_LVA[vic][att]; // normal losing capture
            }
        }
    }
    if(movePromo(m))s+=9500+(movePromo(m)==4?900:movePromo(m)==3?500:movePromo(m)==2?330:320);
    if(!moveCapture(m)&&!movePromo(m)){
        int pt  = b.pieceOn[moveFrom(m)];
        int to  = moveTo(m);
        if(ply>=0&&ply<MAX_PLY){
            if(m==killers[ply][0])      s+=8500;
            else if(m==killers[ply][1]) s+=7500;
            // Counter move bonus (Stockfish counter-move heuristic)
            else if(ply>0 && plyStack[ply-1].currentMove &&
                    m==counterMove[plyStack[ply-1].movedPiece][plyStack[ply-1].movedTo])
                s+=6000;
        }
        // History heuristic
        s += history[b.side][moveFrom(m)][to];
        // Continuation history (1-ply and 2-ply back)
        if(pt>=0&&pt<6&&to>=0&&to<64){
            if(ply>0 && plyStack[ply-1].movedPiece>=0)
                s += contHist[plyStack[ply-1].movedPiece][plyStack[ply-1].movedTo][pt][to] / 2;
            if(ply>1 && plyStack[ply-2].movedPiece>=0)
                s += contHist[plyStack[ply-2].movedPiece][plyStack[ply-2].movedTo][pt][to] / 4;
        }
    }
    return s;
}

// External signature required by header
void scoreAndSortMoves(const Board &b,MoveList &ml,Move pvMove,int *scores){
    int tmp[256];if(!scores)scores=tmp;
    for(int i=0;i<ml.count;i++) scores[i]=(ml.moves[i]==pvMove)?(INF+1):scoreMove(b,ml.moves[i],0);
    for(int i=0;i<ml.count-1;i++){int best=i;for(int j=i+1;j<ml.count;j++)if(scores[j]>scores[best])best=j;if(best!=i){std::swap(scores[i],scores[best]);std::swap(ml.moves[i],ml.moves[best]);}}
}

static void sortMoves(const Board &b,MoveList &ml,Move pvMove,int ply){
    int s[256];
    for(int i=0;i<ml.count;i++) s[i]=(ml.moves[i]==pvMove)?(INF+2):scoreMove(b,ml.moves[i],ply);
    for(int i=0;i<ml.count-1;i++){int best=i;for(int j=i+1;j<ml.count;j++)if(s[j]>s[best])best=j;if(best!=i){std::swap(s[i],s[best]);std::swap(ml.moves[i],ml.moves[best]);}}
}

int quiescence(Board &b,int alpha,int beta,int depth,int ply,SearchInfo &info){
    info.nodes++;checkTime(info);if(info.stop)return alpha;
    int sp=evaluate(b);  // use full evaluate, not raw nnue_evaluate
    if(sp>=beta)return beta;
    if(sp>alpha)alpha=sp;
    if(ply>=MAX_PLY-1)return alpha;
    MoveList ml;generateMoves(b,ml);
    bool inChk=b.inCheck();
    int sc[256],cnt=0;
    for(int i=0;i<ml.count;i++){
        if(inChk||(moveCapture(ml.moves[i]))){
            sc[cnt]=scoreMove(b,ml.moves[i],ply);ml.moves[cnt++]=ml.moves[i];
        }
    }
    ml.count=cnt;
    for(int i=0;i<ml.count-1;i++){int best=i;for(int j=i+1;j<ml.count;j++)if(sc[j]>sc[best])best=j;if(best!=i){std::swap(sc[i],sc[best]);std::swap(ml.moves[i],ml.moves[best]);}}
    for(int i=0;i<ml.count;i++){
        if(!inChk&&moveCapture(ml.moves[i])){int v=b.pieceOn[moveTo(ml.moves[i])];if(v!=NO_PIECE&&sp+MATERIAL[v]+200<alpha)continue;}
        UndoInfo u;if(!b.makeMove(ml.moves[i],u))continue;
        int score=-quiescence(b,-beta,-alpha,depth-1,ply+1,info);
        b.unmakeMove(ml.moves[i],u);
        if(info.stop)return alpha;
        if(score>=beta)return beta;
        if(score>alpha)alpha=score;
    }
    return alpha;
}

// Static eval history per ply — for improving flag (Stockfish idea)
static int staticEvalStack[MAX_PLY+4];

int alphaBeta(Board &b,int alpha,int beta,int depth,int ply,SearchInfo &info,bool nullOk,Move excludedMove=0){
    if(info.stop)return 0;
    info.nodes++;checkTime(info);if(info.stop)return 0;

    bool inChk=b.inCheck();
    if(inChk)depth++;                            // Check extension
    if(depth<=0)return quiescence(b,alpha,beta,0,ply,info);
    if(ply>0&&b.halfMove>=100)return 0;
    if(ply>=MAX_PLY-1)return evaluate(b);

    // ── Mate distance pruning ──────────────────────────────────
    int matedScore = -(MATE-ply);
    int mateScore  =  (MATE-ply-1);
    alpha = std::max(alpha, matedScore);
    beta  = std::min(beta,  mateScore);
    if(alpha>=beta) return alpha;

    // ── Transposition table ───────────────────────────────────
    U64  hash = hashBoard(b);
    Move ttMv=0; int ttSc=0, ttDepth=0, ttFlag=0, ttValue=0;
    if(!excludedMove && ply>0){
        int idx=(int)(hash&(TT_SIZE-1));
        if(TT[idx].key==hash){
            ttMv    = TT[idx].bestMove;
            ttDepth = TT[idx].depth;
            ttFlag  = TT[idx].flag;
            ttValue = TT[idx].score;
            if(ttDepth>=depth && ttValue!=0){  // never trust 0 scores
                if(ttFlag==TT_EXACT)             return ttValue;
                if(ttFlag==TT_ALPHA&&ttValue<=alpha) return alpha;
                if(ttFlag==TT_BETA &&ttValue>=beta)  return beta;
            }
        }
    }

    // ── Static evaluation ─────────────────────────────────────
    int staticEval = evaluate(b);
    staticEvalStack[ply] = staticEval;

    // Improving: is our position better than 2 plies ago?
    bool improving = (ply>=2) && (staticEval > staticEvalStack[ply-2]);

    // ── Razoring (from Stockfish step 7) ─────────────────────
    // If static eval is way below alpha even with a big margin,
    // just drop into qsearch — no point doing a full search
    if(!inChk && depth<=3 && staticEval < alpha - 300 - 200*depth)
        return quiescence(b,alpha,beta,0,ply,info);

    // ── Futility pruning (Stockfish step 8) ──────────────────
    // If static eval - margin is still >= beta, return early
    // Only at shallow depths, not in check, not near mate
    if(!inChk && depth<8 && staticEval>=beta
       && staticEval - (80 - 20*improving)*depth < MATE/2
       && beta > -MATE/2)
        return staticEval;

    // ── Null move pruning (Stockfish step 9) ─────────────────
    if(nullOk&&!inChk&&depth>=3&&ply>0&&staticEval>=beta){
        int mat=0;for(int p=0;p<5;p++)mat+=MATERIAL[p]*popcount(b.pieces[b.side][p]);
        if(mat>MATERIAL[ROOK]){
            UndoInfo nu;nu.save(b,0,NO_PIECE);
            b.enPassant=NO_SQ;b.side^=1;
            if(b.side==WHITE)b.fullMove++;b.recalcOccupancy();
            // Stockfish: R = 4 + depth/3 + min(3, (staticEval-beta)/200)
            int R=3+depth/3+std::min(3,(staticEval-beta)/200);
            int ns=-alphaBeta(b,-beta,-beta+1,depth-1-R,ply+1,info,false);
            nu.restore(b);
            if(info.stop)return 0;
            if(ns>=beta&&ns<MATE/2) return beta;
        }
    }

    // ── IIR: Internal Iterative Reduction (Stockfish step 10) ─
    // No TT move at high depth? Reduce by 1 to save time
    if(depth>=4 && ttMv==0)
        depth--;

    // Singular extensions removed for stability
    int singularExtension = 0;

    // ProbCut removed - was causing TT score pollution

    // ── Generate and sort moves ───────────────────────────────
    MoveList ml;generateMoves(b,ml);
    sortMoves(b,ml,ttMv,ply);

    bool hasLegal=false;int bestScore=-INF;Move bestMv=0;int origAlpha=alpha;int mc=0;

    for(int i=0;i<ml.count;i++){
        Move m=ml.moves[i];
        if(m==excludedMove) continue;  // skip excluded move (for singular search)
        UndoInfo u;
        if(!b.makeMove(m,u))continue;
        hasLegal=true;mc++;
        bool isCap=moveCapture(m)||moveEP(m),isPro=movePromo(m),gvChk=b.inCheck();

        // Update ply stack for continuation history
        if(ply<MAX_PLY){
            plyStack[ply].currentMove  = m;
            plyStack[ply].movedPiece   = b.pieceOn[moveTo(m)] != NO_PIECE ?
                                         b.pieceOn[moveTo(m)] : // after capture
                                         u.pieceOn[moveFrom(m)]; // moved piece type
            // Actually use the piece that moved (from undo info)
            plyStack[ply].movedPiece   = u.pieceOn[moveFrom(m)];
            plyStack[ply].movedTo      = moveTo(m);
        }

        // Apply singular extension to TT move
        int extension = (m==ttMv) ? singularExtension : 0;
        // Also extend checks (but not double with singular)
        if(gvChk && extension==0) extension=1;

        // ── SEE + Futility pruning (Stockfish step 14) ──────────
        if(!inChk && bestScore>-MATE/2){
            if(isCap && !isPro){
                // TAL SOUL: Never prune sacrifices near the enemy king!
                // Tal's whole style was built on "unsound" sacrifices that
                // turned out to be VERY sound because of the chaos they created.
                int toSq = moveTo(m);
                int oppKingSqP = b.pieces[b.side^1][KING]
                                 ? lsb(b.pieces[b.side^1][KING]) : 0;
                U64 kingZoneP = kingAttacks[oppKingSqP] | (1ULL << oppKingSqP);
                bool isTalSacrifice = (kingZoneP & (1ULL << toSq)) != 0;
                // SEE pruning: skip losing captures at depth
                // Stockfish: !see_ge(move, -margin*depth)
                // TAL: if near enemy king, be more lenient (margin halved)
                int seePruneMargin = isTalSacrifice ? -45*depth : -90*depth;
                if(!seeGe(b,m,seePruneMargin)){
                    b.unmakeMove(m,u);
                    continue;
                }
            } else if(!isCap && !isPro && !gvChk){
                int lmrDepth=std::max(0,depth-1-(mc>6?2:mc>3?1:0));
                int futilityVal=staticEval+60+120*lmrDepth;
                if(futilityVal<=alpha && lmrDepth<7){
                    b.unmakeMove(m,u);
                    continue;
                }
                int maxMoves=3+depth*depth/(improving?1:2);
                if(mc>maxMoves && depth<=6){
                    b.unmakeMove(m,u);
                    continue;
                }
                // SEE: prune quiet moves with very negative exchange
                if(depth<=8 && !seeGe(b,m,-30*depth*depth)){
                    b.unmakeMove(m,u);
                    continue;
                }
            }
        }

        int newDepth = depth - 1 + extension;
        int score;

        // ── LMR: Late Move Reduction (Stockfish step 17) ──────
        // TAL SOUL: Extend search for captures near enemy king by 1 ply
        // (on top of existing extensions) — Tal searched these very deeply!
        if(isCap && extension == 0 && ply < 6) {
            int toSqL = moveTo(m);
            int oppKingSqL = b.pieces[b.side^1][KING]
                             ? lsb(b.pieces[b.side^1][KING]) : 0;
            U64 kingZoneL = kingAttacks[oppKingSqL] | (1ULL << oppKingSqL);
            if(kingZoneL & (1ULL << toSqL))
                extension = 1;  // Tal Extension: search king attacks 1 ply deeper
        }

        bool doLMR=newDepth>=2&&mc>2&&!isCap&&!isPro&&!inChk&&ply>0&&extension==0;
        if(mc==1){
            score=-alphaBeta(b,-beta,-alpha,newDepth,ply+1,info,true);
        } else if(doLMR){
            int R=1;
            if(newDepth>=6)  R++;
            if(newDepth>=10) R++;
            if(mc>6)         R++;
            if(mc>12)        R++;
            if(!improving)   R++;
            // Reduce more for moves with bad continuation history
            if(ply>0 && plyStack[ply].movedPiece>=0 && plyStack[ply].movedPiece<6){
                int pt2=plyStack[ply].movedPiece, to2=plyStack[ply].movedTo;
                int prevPt=ply>0?plyStack[ply-1].movedPiece:0;
                int prevTo=ply>0?plyStack[ply-1].movedTo:0;
                if(prevPt>=0&&prevPt<6&&prevTo>=0&&prevTo<64)
                    if(contHist[prevPt][prevTo][pt2][to2] < -500) R++;
            }

            int reducedDepth=std::max(1,newDepth-R);
            score=-alphaBeta(b,-alpha-1,-alpha,reducedDepth,ply+1,info,true);

            if(score>alpha){
                int fullDepth=newDepth+(score>bestScore+50?1:0);
                score=-alphaBeta(b,-beta,-alpha,fullDepth,ply+1,info,true);
            }
        } else {
            score=-alphaBeta(b,-alpha-1,-alpha,newDepth,ply+1,info,true);
            if(score>alpha&&score<beta)
                score=-alphaBeta(b,-beta,-alpha,newDepth,ply+1,info,true);
        }

        b.unmakeMove(m,u);
        if(info.stop)return 0;

        if(score>bestScore){bestScore=score;bestMv=m;}
        if(score>alpha){
            alpha=score;
            if(alpha>=beta){
                // Update killers and history on quiet cutoff
                if(!isCap&&ply<MAX_PLY){
                    killers[ply][1]=killers[ply][0];
                    killers[ply][0]=m;
                    int bonus=std::min(depth*depth, 400);
                    int mover=b.side^1; // side that just moved
                    int pt  =u.pieceOn[moveFrom(m)]; // piece that moved
                    int to  =moveTo(m);

                    // History bonus
                    history[mover][moveFrom(m)][to] += bonus;
                    // Clamp to avoid overflow
                    if(history[mover][moveFrom(m)][to] > 8000)
                        history[mover][moveFrom(m)][to] = 8000;

                    // Counter move: remember this move as response to previous
                    if(ply>0 && plyStack[ply-1].movedPiece>=0){
                        counterMove[plyStack[ply-1].movedPiece][plyStack[ply-1].movedTo] = m;
                    }

                    // Continuation history bonus
                    if(pt>=0&&pt<6&&to>=0&&to<64){
                        if(ply>0 && plyStack[ply-1].movedPiece>=0)
                            contHist[plyStack[ply-1].movedPiece][plyStack[ply-1].movedTo][pt][to]+=bonus;
                        if(ply>1 && plyStack[ply-2].movedPiece>=0)
                            contHist[plyStack[ply-2].movedPiece][plyStack[ply-2].movedTo][pt][to]+=bonus/2;
                    }

                    // Penalize quiet moves that didn't cause cutoff
                    for(int j=0;j<i;j++){
                        if(ml.moves[j]==excludedMove) continue;
                        if(!moveCapture(ml.moves[j])&&!movePromo(ml.moves[j])){
                            int ppt=u.pieceOn[moveFrom(ml.moves[j])];
                            int pto=moveTo(ml.moves[j]);
                            history[mover][moveFrom(ml.moves[j])][pto] -= bonus/4;
                            if(ppt>=0&&ppt<6&&pto>=0&&pto<64){
                                if(ply>0&&plyStack[ply-1].movedPiece>=0)
                                    contHist[plyStack[ply-1].movedPiece][plyStack[ply-1].movedTo][ppt][pto]-=bonus/4;
                            }
                        }
                    }
                }
                ttStore(hash,beta,m,depth,TT_BETA);
                return beta;
            }
        }
    }

    if(!hasLegal) return inChk?-(MATE-ply):0;
    ttStore(hash,bestScore,bestMv,depth,(bestScore<=origAlpha)?TT_ALPHA:
            (bestScore>=beta?TT_BETA:TT_EXACT));
    return bestScore;
}

// ============================================================
//  OPENING BOOK — random se ek strong opening choose hoti hai
// ============================================================
struct BookLine { const char* moves[12]; };
static const BookLine BOOK[] = {
    {{"e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6",nullptr}},   // Ruy Lopez
    {{"e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6",nullptr}},   // Italian
    {{"e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6","b1c3","a7a6",nullptr}}, // Sicilian Najdorf
    {{"e2e4","e7e6","d2d4","d7d5","b1c3","g8f6","c1g5",nullptr}},           // French Classical
    {{"d2d4","d7d5","c2c4","e7e6","b1c3","g8f6","c1g5","f8e7",nullptr}},   // QGD
    {{"d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6","g1f3",nullptr}}, // King's Indian
    {{"c2c4","e7e5","b1c3","g8f6","g1f3","b8c6","g2g3",nullptr}},           // English
    {{"d2d4","d7d5","g1f3","g8f6","c1f4","e7e6","e2e3",nullptr}},           // London System
    {{"e2e4","c7c6","d2d4","d7d5","b1c3","g7g6",nullptr}},                  // Caro-Kann
    {{"d2d4","g8f6","c2c4","e7e6","b1c3","f8b4","e2e3",nullptr}},           // Nimzo-Indian
    {{"d2d4","d7d5","c2c4","c7c6","g1f3","g8f6","b1c3","e7e6",nullptr}},   // Slav
    {{"e2e4","g7g6","d2d4","f8g7","b1c3","d7d6","g1f3",nullptr}},           // Pirc
};
static const int N_BOOK = 12;
static int g_bookLine = -1; // -1 = not yet chosen

static Move getBookMove(const Board &b) {
    // Pick a line on move 1
    if(b.fullMove == 1 && b.side == WHITE) {
        g_bookLine = rand() % N_BOOK;
    }
    if(g_bookLine < 0) return 0;

    // Which move index are we at?
    int idx = (b.fullMove - 1) * 2 + (b.side == BLACK ? 1 : 0);
    const char* mv = BOOK[g_bookLine].moves[idx];
    if(!mv) return 0;

    // Verify position matches by replaying
    Board tmp;
    tmp.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    for(int i = 0; i < idx; i++) {
        const char* bm = BOOK[g_bookLine].moves[i];
        if(!bm) return 0;
        Move m = strToMove(tmp, bm);
        if(!m) { g_bookLine=-1; return 0; }
        UndoInfo u; tmp.makeMove(m, u);
    }
    // Quick match check
    for(int c=0;c<2;c++) for(int p=0;p<6;p++)
        if(tmp.pieces[c][p]!=b.pieces[c][p]){ g_bookLine=-1; return 0; }
    if(tmp.side!=b.side){ g_bookLine=-1; return 0; }

    Move bookMv = strToMove(b, mv);
    if(!bookMv) g_bookLine=-1;
    return bookMv;
}

Move bestMove(Board &b,int depth,int timeLimitMs){
    initZobrist();initTT();clearTables();
    SearchInfo info;
    info.depth=depth;info.nodes=0;info.stop=false;
    info.timeLimit=timeLimitMs;info.startTime=std::chrono::steady_clock::now();

    // ── Opening Book ─────────────────────────────────────────
    Move bookMv = getBookMove(b);
    if(bookMv){
        std::cout<<"info depth 0 score cp 20 nodes 1 time 0 pv "<<moveToStr(bookMv)<<"\n";
        std::cout<<"bestmove "<<moveToStr(bookMv)<<"\n";
        std::cout.flush();
        return bookMv;
    }

    MoveList ml;generateMoves(b,ml);
    MoveList legal;UndoInfo tu;
    for(int i=0;i<ml.count;i++){Board t=b;if(t.makeMove(ml.moves[i],tu))legal.add(ml.moves[i]);}
    if(legal.count==0){std::cout<<"bestmove 0000\n";return 0;}

    Move best=legal.moves[0];int bestScore=-INF;

    for(int d=1;d<=depth&&!info.stop;d++){
        int alpha=-INF,beta=INF;
        // Aspiration windows from depth 5+
        if(d>=5&&bestScore>-MATE/2){alpha=bestScore-50;beta=bestScore+50;}

        int iterBest=-INF;Move iterMv=legal.moves[0];
        bool retry=false;
        do {
            retry=false;iterBest=-INF;iterMv=legal.moves[0];
            for(int i=0;i<legal.count&&!info.stop;i++){
                UndoInfo u;if(!b.makeMove(legal.moves[i],u))continue;
                int score;
                if(i==0){score=-alphaBeta(b,-beta,-alpha,d-1,1,info,true);}
                else{
                    score=-alphaBeta(b,-alpha-1,-alpha,d-1,1,info,true);
                    if(score>alpha&&score<beta)score=-alphaBeta(b,-beta,-alpha,d-1,1,info,true);
                }
                b.unmakeMove(legal.moves[i],u);
                if(info.stop)break;
                if(score>iterBest){iterBest=score;iterMv=legal.moves[i];}
                if(score>alpha)alpha=score;
            }
            if(!info.stop&&d>=5){
                if(iterBest<=alpha-50){alpha=-INF;retry=true;}
                else if(iterBest>=beta+50){beta=INF;retry=true;}
            }
        } while(retry&&!info.stop);

        if(!info.stop){
            best=iterMv;bestScore=iterBest;
            // Move best to front
            for(int i=0;i<legal.count;i++){if(legal.moves[i]==best){for(int j=i;j>0;j--)legal.moves[j]=legal.moves[j-1];legal.moves[0]=best;break;}}
            auto e=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-info.startTime).count();
            std::cout<<"info depth "<<d<<" score cp "<<bestScore<<" nodes "<<info.nodes<<" time "<<e<<" pv "<<moveToStr(best)<<"\n";
            std::cout.flush();
        }
    }
    std::cout<<"bestmove "<<moveToStr(best)<<"\n";std::cout.flush();
    return best;
}

// ============================================================
//  UTILITIES
// ============================================================
std::string squareName(int sq){std::string s;s+=(char)('a'+sq%8);s+=(char)('1'+sq/8);return s;}

std::string moveToStr(Move m){
    if(!m)return "0000";
    std::string s=squareName(moveFrom(m))+squareName(moveTo(m));
    if(movePromo(m)){const char pr[]={'?','n','b','r','q'};s+=pr[movePromo(m)];}
    return s;
}

Move strToMove(const Board &b,const std::string &str){
    if(str.length()<4)return 0;
    int from=(str[0]-'a')+(str[1]-'1')*8,to=(str[2]-'a')+(str[3]-'1')*8;
    if(from<0||from>63||to<0||to>63)return 0;
    MoveList ml;generateMoves(b,ml);
    for(int i=0;i<ml.count;i++){
        if(moveFrom(ml.moves[i])==from&&moveTo(ml.moves[i])==to){
            if(str.length()>4){char pc=str[4];int pr=(pc=='n')?1:(pc=='b')?2:(pc=='r')?3:(pc=='q')?4:0;if(movePromo(ml.moves[i])!=pr)continue;}
            else if(movePromo(ml.moves[i])!=0)continue;
            UndoInfo u;Board t=b;if(t.makeMove(ml.moves[i],u))return ml.moves[i];
        }
    }
    return 0;
}

void uciLoop(){
    initZobrist();initTT();
    setvbuf(stdout,NULL,_IONBF,0);setvbuf(stdin,NULL,_IONBF,0);
    Board b;b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::string line,token;
    while(true){
        if(!std::getline(std::cin,line))break;
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        if(line.empty())continue;
        std::istringstream ss(line);ss>>token;
        if(token=="uci"){
            std::cout<<"id name Zero v12.0\n"
                     <<"id author Rohan+Claude\n"
                     #if HAS_NNUE
                     <<"option name NNUE type check default true\n"
                     <<"info string NNUE: ACTIVE (Lc0 188M, 0.3cp)\n"
                     #else
                     <<"info string NNUE: NOT LOADED (HCE only)\n"
                     #endif
                     #if HAS_TAL_SOUL
                     <<"info string TalSoul: ACTIVE\n"
                     #endif
                     <<"uciok\n";
            std::cout.flush();
        }
        else if(token=="isready"){std::cout<<"readyok\n";std::cout.flush();}
        else if(token=="ucinewgame"){b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");clearTables();memset(TT,0,sizeof(TT));}
        else if(token=="position"){
            std::string pos;ss>>pos;
            if(pos=="startpos"){b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");ss>>token;}
            else if(pos=="fen"){std::string fen,p;ss>>p;fen=p;while(ss>>p&&p!="moves")fen+=" "+p;b.setFromFEN(fen);token=p;}
            std::string mv;while(ss>>mv){Move m=strToMove(b,mv);if(m){UndoInfo u;b.makeMove(m,u);}}
        }
        else if(token=="go"){
            int depth=64,movetime=-1,wtime=-1,btime=-1,winc=0,binc=0;
            bool ds=false,ms=false;
            while(ss>>token){
                if(token=="depth"){ss>>depth;ds=true;}
                else if(token=="movetime"){ss>>movetime;ms=true;}
                else if(token=="wtime")ss>>wtime;
                else if(token=="btime")ss>>btime;
                else if(token=="winc")ss>>winc;
                else if(token=="binc")ss>>binc;
                else if(token=="infinite"){depth=64;ds=true;ms=false;movetime=-1;}
            }
            if(!ms&&wtime>0&&btime>0){
                int myTime=(b.side==WHITE)?wtime:btime;
                int myInc =(b.side==WHITE)?winc:binc;
                movetime=myTime/15+(myInc*4)/5;
                movetime=std::max(200,std::min(movetime,myTime/3));
            }
            if(!ds&&!ms)depth=64;
            if(movetime<0)movetime=5000;
            bestMove(b,depth,movetime);
        }
        else if(token=="d"){b.print();std::cout.flush();}
        else if(token=="eval"){std::cout<<"Evaluation: "<<evaluate(b)<<"\n";std::cout.flush();}
        else if(token=="perft"){int d;ss>>d;if(d>0){auto t1=std::chrono::steady_clock::now();auto n=perft(b,d);auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t1).count();std::cout<<"perft "<<d<<" = "<<n<<" ("<<ms<<" ms)\n";}}
        else if(token=="divide"){int d;ss>>d;if(d>0)divide(b,d);}
        else if(token=="quit"||token=="exit")break;
    }
}
