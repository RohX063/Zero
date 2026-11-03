#include "position.h"
#include "search.h"
#include "search_helpers.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
int main(int argc,char**argv) {
 if(argc!=3) return 1;
 Position::init(); std::ifstream f(argv[1]); std::string fen; int n=0; int depth=std::stoi(argv[2]);
 std::cout<<"index,nodes,move,score,eligible,up,down,punished,to_zero,lmr,research,prior0,prior1,prior2,prior3plus\n";
 while(std::getline(f,fen)) {
  if(fen.empty()||fen[0]=='#') continue;
  Position pos; if(!pos.loadFEN(fen+" 0 1")) return 2;
  Position original=pos;
  Zero::Search::Worker w(pos);
  std::ostringstream sink; auto *old=std::cout.rdbuf(sink.rdbuf());
  auto result=w.iterativeDeepening(depth);
  std::cout.rdbuf(old);
  if(!pos.equals(original)||!result.completed||!result.bestMove.isOk()) return 3;
  const auto& s=w.stats();
  std::cout<<++n<<','<<w.nodes()<<','<<Zero::Search::uciMoveString(result.bestMove)<<','<<result.score<<','<<s.hindsightEligible<<','<<s.hindsightUp<<','<<s.hindsightDown<<','<<s.punishApplied<<','<<s.punishToZero<<','<<s.lmrReductions<<','<<s.lmrResearches<<','<<s.prior0<<','<<s.prior1<<','<<s.prior2<<','<<s.prior3plus<<'\n';
 }
}
