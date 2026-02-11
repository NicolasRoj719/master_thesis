#include <memory>
#include <vector>
#include <type_traits>
#include <cassert>
#include <limits>
#include "jacobian.hpp"
#include "table_cell.hpp"
#include "generator.hpp"
#include "table.hpp"
#include "behaviour_build_chain.hpp"
#include "behaviour_fill_table.hpp"

#ifndef SOLVER_HPP
#define SOLVER_HPP

template<class Cell_type, class Jacobian_type>
class Solver{
 public:
  virtual void solve(std::vector<std::vector<std::size_t>> problem) = 0;
};

template<class Cell_type, class Jacobian_type>
class DJCPB: public Solver<Cell_type, Jacobian_type>{ 
 public:
  void solve(std::vector<std::vector<std::size_t>> problem) override{
    build_chain_dense<Jacobian_type> chain_beh;
    chain_beh.build_chain(problem);
    chain_beh.print();
    std::vector<Jacobian_type> jacobian_chain = chain_beh.get_chain(); 
    std::size_t chain_len = problem.size();
    std::size_t table_len = (chain_len + 1) * chain_len / 2;
    Table<Cell_type> table{problem.size(), table_len};
    fill_DJCPB<Cell_type, Jacobian_type> fill_beh;
    fill_beh.fill(table, jacobian_chain);
    table.print();
  }
};

template<class Cell_type, class Jacobian_type>
class MFDJCPB: public Solver<Cell_type, Jacobian_type>{
 public:
  MFDJCPB(std::size_t mem_limit = std::numeric_limits<size_t>::max()):
  memory_limit(mem_limit){
    static_assert(std::is_same_v<Cell_type, cell_MFDJCPB>);
  }
  void solve(std::vector<std::vector<std::size_t>> problem) override{
    build_chain_dense<Jacobian_type> chain_beh;
    chain_beh.build_chain(problem);
    chain_beh.print();
    std::vector<Jacobian_type> jacobian_chain = chain_beh.get_chain(); 
    std::size_t chain_len = problem.size();
    std::size_t table_len = (chain_len + 1) * chain_len / 2;
    Table<Cell_type> table{problem.size(), table_len};
    std::cout<<"PRINTING MEMORY LIMIT BEFORE CALLING FILL BEHAVIOUR.\n";
    std::cout<<memory_limit<<'\n';
    fill_MFDJCPB<Cell_type, Jacobian_type> fill_beh{memory_limit};
    fill_beh.fill(table, jacobian_chain);
    table.print();
  }

 private:
  std::size_t memory_limit;
};

#endif
