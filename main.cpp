#include "chess.h"
#include <cstdlib>
#include <ctime>

// ============================================================
//  ZERO v1.2 - The Void Consumes All
//  Compile: g++ -O3 -o Zero.exe main.cpp chess.cpp
// ============================================================

int main(int argc, char* argv[]) {
    // Initialize random seed for opening variety
    srand(static_cast<unsigned>(time(nullptr)));
    
    // Initialize attack tables
    initAttacks();

    // Self-test mode
    if(argc > 1 && std::string(argv[1]) == "test") {
        Board b;
        b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        b.print();

        MoveList ml;
        generateMoves(b, ml);
        std::cout << "Generated moves: " << ml.count << " (expected: 20)\n";

        // Count legal moves
        int legalCount = 0;
        for(int i = 0; i < ml.count; i++) {
            Board::UndoInfo u;
            Board tmp = b;
            if(tmp.makeMove(ml.moves[i], u)) legalCount++;
        }
        std::cout << "Legal moves: " << legalCount << " (expected: 20)\n\n";

        std::cout << "Testing search with time management...\n";
        Move m = bestMove(b, 5, 3000); // Depth 5, 3 seconds
        if(m) {
            std::cout << "\nBest move found: " << moveToStr(m) << "\n";
        } else {
            std::cout << "\nNo move found!\n";
        }
        
        std::cout << "\n=== ZERO v1.2 READY ===\n";
        std::cout << "From nothing, checkmate.\n";
        return 0;
    }

    // Normal UCI mode
    uciLoop();
    return 0;
}