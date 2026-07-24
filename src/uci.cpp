#include "uci.h"

#include "board.h"
#include "movegen.h"
#include "search.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
std::string toUciMove(const Move& move)
{
    std::string uciMove;
    uciMove += char('a' + move.fromCol);
    uciMove += char('8' - move.fromRow);
    uciMove += char('a' + move.toCol);
    uciMove += char('8' - move.toRow);

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
            default:
                break;
        }
    }

    return uciMove;
}

bool applyMove(Board& board, const std::string& moveText)
{

    const std::vector<Move> legalMoves =
        generateLegalMoves(board, board.isWhiteToMove());

    for (const Move& legalMove : legalMoves)
    {
        if (toUciMove(legalMove) == moveText)
        {

            Move move = legalMove;
            board.makeMove(move);

            return true;
        }
    }
    return false;
}

void setPosition(Board& board, const std::string& command)
{

    const std::string prefix = "position ";
    if(command.rfind(prefix, 0) != 0)
        return;

    std::string body = command.substr(prefix.size());

    if(body.rfind("startpos", 0) == 0)
    {
        board = Board();
        body = body.substr(8);
    }
    else if(body.rfind("fen ", 0) == 0)
    {
        std::string fenAndMoves = body.substr(4);
        const size_t movesPos = fenAndMoves.find(" moves ");

        std::string fen = fenAndMoves;
        std::string movesText;

        if(movesPos != std::string::npos)
        {
            fen = fenAndMoves.substr(0, movesPos);
            movesText = fenAndMoves.substr(movesPos + 7);
        }

        if(!board.loadFEN(fen))
        {
            board = Board();
            return;
        }

        if(!movesText.empty())
        {
            std::stringstream ss(movesText);
            std::string moveText;
            while(ss >> moveText)
            {
                if(!applyMove(board, moveText))
                    break;
            }
        }


        return;
    }
    else
    {
        return;
    }

    if(!body.empty() && body[0] == ' ')
        body.erase(0, 1);

    const size_t movesPos = body.find("moves ");
    if(movesPos != std::string::npos)
    {
        std::stringstream ss(body.substr(movesPos + 6));
        std::string moveText;
        while(ss >> moveText)
        {
            if(!applyMove(board, moveText))
                break;
        }
    }
}

int parseGoDepth(const std::string& command)
{
    std::stringstream ss(command);
    std::string token;
    int depth = 0;

    while(ss >> token)
    {
        if(token == "depth")
        {
            ss >> token;
            depth = std::stoi(token);
            break;
        }
    }

    return depth > 0 ? depth : 5;
}
} // namespace

void uciLoop()
{
    std::string command;
    Board board;

    while(std::getline(std::cin, command))
    {
        if(command.empty())
            continue;

        if(command == "uci")
        {
            std::cout << "id name Zero" << std::endl;
            std::cout << "id author Rohan Singh" << std::endl;
            std::cout << "option name Hash type spin default 64 min 1 max 1024" << std::endl;
            std::cout << "uciok" << std::endl;
        }
        else if(command == "isready")
        {
            std::cout << "readyok" << std::endl;
        }
        else if(command == "quit")
        {
            break;
        }
        else if(command == "ucinewgame")
        {
            board = Board();
        }
        else if(command.rfind("position ", 0) == 0)
        {
            setPosition(board, command);
        }
        else if(command.rfind("go", 0) == 0)
        {
            board.printBoard();
            std::cout
            << "GO COMMAND: side = "
            << (board.isWhiteToMove() ? "WHITE" : "BLACK")
            << std::endl;
            const int depth = parseGoDepth(command);
            const Move bestMove = iterativeDeepening(board, depth);
            std::cout << "bestmove " << toUciMove(bestMove) << std::endl;
        }
        else if(command.rfind("setoption ", 0) == 0)
        {
            // Ignore for now; cutechess only needs basic UCI compliance.
        }
        else if(command.rfind("stop", 0) == 0)
        {
            // Search is single-shot in this implementation.
        }
    }
}
