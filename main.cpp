#include "chess.h"
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {
    srand(static_cast<unsigned>(time(nullptr)));
    initAttacks();

    // Handle command line arguments
    if(argc > 1) {
        std::string cmd = argv[1];
        
        if(cmd == "test") {
            Board b;
            b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            b.print();

            std::cout << "=== PERFT TESTS ===\n";
            auto t1 = std::chrono::steady_clock::now();
            unsigned long long n = perft(b, 5);
            auto t2 = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
            std::cout << "Perft(5) = " << n << " (expected: 4865609)\n";
            std::cout << "Time: " << ms << "ms\n";
            if(n == 4865609ULL) std::cout << "PASSED\n"; 
            else std::cout << "FAILED\n";

            std::cout << "\n=== SEARCH TEST ===\n";
            Move m = bestMove(b, 5, 3000);
            if(m) std::cout << "\nBest move: " << moveToStr(m) << "\n";
            else std::cout << "\nNo move!\n";

            std::cout << "\n=== ZERO READY ===\n";
            return 0;
        }
        else if(cmd == "divide" && argc > 2) {
            // DIVIDE COMMAND FIX
            int depth = std::stoi(argv[2]);
            Board b;
            b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            
            std::cout << "=== DIVIDE DEPTH " << depth << " ===\n";
            divide(b, depth);
            return 0;
        }
        else if(cmd == "perft" && argc > 2) {
            // PERFT COMMAND FIX
            int depth = std::stoi(argv[2]);
            Board b;
            b.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            
            std::cout << "=== PERFT DEPTH " << depth << " ===\n";
            auto t1 = std::chrono::steady_clock::now();
            unsigned long long n = perft(b, depth);
            auto t2 = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
            std::cout << "perft " << depth << " = " << n << " (" << ms << " ms)\n";
            return 0;
        }
    }

    // Default: UCI mode
    uciLoop();
    return 0;
}
