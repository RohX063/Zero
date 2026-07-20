#include "uci.h"

#include "board.h"
#include "search.h"
#include "movegen.h"
#include "tt.h"
#include "history.h"
#include "killer.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void uciLoop()
{
    std::string command;
    Board board;

    while(std::getline(std::cin, command))
    {
        std::cerr << "RECV: [" << command << "]" << std::endl;

        if(command == "uci")
        {
            std::cout << "id name Zero" << std::endl;
            std::cout << "id author Rohan Singh" << std::endl;
            std::cout << "uciok" << std::endl;
        }
        else if(command == "isready")
        {
            std::cout << "readyok\n";
        }
        else if(command == "quit")
        {
            break;
        }
        else if(command.rfind("go", 0) == 0)
        {
            std::cerr << "GO RECEIVED\n";

            std::cerr << "Side to move = "
                      << (board.isWhiteToMove() ? "White" : "Black")
                      << '\n';

            std::cerr << "Generating moves...\n";

            auto legalMoves = generateLegalMoves(board, board.isWhiteToMove());

            std::cerr << "Legal move count = "
                      << legalMoves.size()
                      << '\n';

            Move bestMove = iterativeDeepening(board, 5);

            std::cerr << "Search finished\n";

            std::cout << "bestmove "
                      << char('a' + bestMove.fromCol)
                      << (8 - bestMove.fromRow)
                      << char('a' + bestMove.toCol)
                      << (8 - bestMove.toRow)
                      << std::endl;
        }
        else if(command.rfind("position startpos", 0) == 0)
        {
            if(!board.loadFEN(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"
            ))
            {
                std::cerr << "Failed to load start position\n";
            }

            const size_t movesPos = command.find("moves");

            if(movesPos != std::string::npos)
            {
                std::string movesString = command.substr(movesPos + 6);
                std::stringstream ss(movesString);
                std::string moveString;

                while(ss >> moveString)
                {
                    std::cerr << "Move = " << moveString << '\n';

                    std::vector<Move> legalMoves =
                        generateLegalMoves(board, board.isWhiteToMove());

                    bool found = false;

                    for(Move &move : legalMoves)
                    {
                        std::string uciMove;

                        uciMove += char('a' + move.fromCol);
                        uciMove += char('8' - move.fromRow);
                        uciMove += char('a' + move.toCol);
                        uciMove += char('8' - move.toRow);

                        // Promotion
                        if(move.isPromotion)
                        {
                            switch(move.promotionPiece)
                            {
                                case WHITE_QUEEN:
                                case BLACK_QUEEN:
                                    uciMove += 'q';
                                    break;

                                case WHITE_ROOK:
                                case BLACK_ROOK:
                                    uciMove += 'r';
                                    break;

                                case WHITE_BISHOP:
                                case BLACK_BISHOP:
                                    uciMove += 'b';
                                    break;

                                case WHITE_KNIGHT:
                                case BLACK_KNIGHT:
                                    uciMove += 'n';
                                    break;
                            }
                        }

                        if(uciMove == moveString)
                        {
                            board.makeMove(move);
                            found = true;

                            std::cerr
                                << "Applied: "
                                << uciMove
                                << " | Side to move = "
                                << (board.isWhiteToMove() ? "White" : "Black")
                                << '\n';

                            break;
                        }
                    }

                    if(!found)
                    {
                        std::cerr << "Invalid move: " << moveString << '\n';

                        std::cerr << "Legal moves are:\n";

                        for(const Move &m : legalMoves)
                        {
                            std::string u;

                            u += char('a' + m.fromCol);
                            u += char('8' - m.fromRow);
                            u += char('a' + m.toCol);
                            u += char('8' - m.toRow);

                            if(m.isPromotion)
                            {
                                switch(m.promotionPiece)
                                {
                                    case WHITE_QUEEN:
                                    case BLACK_QUEEN:   u += 'q'; break;

                                    case WHITE_ROOK:
                                    case BLACK_ROOK:    u += 'r'; break;

                                    case WHITE_BISHOP:
                                    case BLACK_BISHOP:  u += 'b'; break;

                                    case WHITE_KNIGHT:
                                    case BLACK_KNIGHT:  u += 'n'; break;
                                }
                            }

                            std::cerr << u << '\n';
                        }

                        break;
                    }
                }
            }
        }
    }
}
