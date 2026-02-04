#include <memory>
#include <vector>
#include <type_traits>
#include <cassert>
#include "jacobian.hpp"
#include "table_cell.hpp"
#include "generator.hpp"
#include "dptable.hpp"
#include "fill_dptable.hpp"

#ifndef SOLVER_HPP
#define SOLVER_HPP

template<class Cell_type, class Jacobian_type>
class Solver{
 public:
  Solver(){}

  Solver(std::size_t len_elemental_jacs, std::size_t dim_lb,
    std::size_t dim_ub, std::size_t n_E_lb, std::size_t n_E_ub):
    elemental_jacobian_chain(len_elemental_jacs),
    elemental_jacobian_chain_size(len_elemental_jacs),
    dim_lower_bound(dim_lb), dim_upper_bound(dim_ub),
    n_E_lower_bound(n_E_lb), n_E_upper_bound(n_E_ub){};

  virtual ~Solver() = default;

  void build_problem(){
    gen_type -> build_problem(); 
  }

  void print_problem(){
    gen_type -> print();
  }

  virtual void build_elemental_jacs(std::size_t len_data){
    //Assert if the problem was already build.
    assert(gen_type -> get_problem_size()>0);

    auto& len = elemental_jacobian_chain_size;

    std::vector<size_t> F_i_data;

    for(std::size_t i=0; i<len; i++){
      F_i_data = std::move(gen_type->read_Fi_description(i));
      Jacobian_type jacobian_i;

      for(std::size_t j=0; j<len_data; j++){
        jacobian_i[j] = F_i_data[j];
      }

      elemental_jacobian_chain[i]= jacobian_i;
    }
  }

  void print_format_elemental_jacs() {
    std::cout<<"F'_i: [n m n_E]\n";
  }
  
  void print_elemental_jacs(){

    print_format_elemental_jacs();

    for(std::size_t i; i<elemental_jacobian_chain.size(); i++){
      elemental_jacobian_chain[i].print();
    }

  }

  virtual void build_table() = 0;

  void print_table(){table_type->print();}

  virtual void build_sequence() = 0;
  virtual void print_sequence() = 0;

 protected:
  std::unique_ptr<Generator> gen_type;
  std::unique_ptr<Table<Cell_type>> table_type;
  std::unique_ptr<behaviour_fill_dptable<Cell_type, Jacobian_type>> fill_behaviour;
  std::vector<Jacobian_type> elemental_jacobian_chain;
  std::size_t elemental_jacobian_chain_size;
  // Information required to call the generator.
  std::size_t dim_lower_bound= 1, dim_upper_bound;
  std::size_t n_E_lower_bound=1, n_E_upper_bound;
};

template<class Cell_type, class Jacobian_type>
class DJCPB: public Solver<Cell_type, Jacobian_type>{ 
 public:
  DJCPB(std::size_t chain_len, std::size_t dim_lb, std::size_t dim_ub,
      std::size_t n_E_lb, std::size_t n_E_ub):
    Solver<Cell_type, Jacobian_type>(chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub){
    assert((std::is_same_v<Cell_type, cell_DJCPB>));
    assert(!(std::is_same_v<Jacobian_type, Base_Jacobian>));

    this->gen_type = std::make_unique<n_m_n_E_Generator>(chain_len, dim_lb, dim_ub,
        n_E_lb, n_E_ub);

    this->table_type = std::make_unique<Table<Cell_type>>(chain_len,
        (chain_len+1) * chain_len / 2);

    this->fill_behaviour = std::make_unique<fill_DJCPB<Table<Cell_type>, Jacobian_type>>();
  }

  void build_elemental_jacs(std::size_t len_data=3) override{
    //n (input dimension), m (output dimensioni), n_E (number of edges in the DAG)
    Solver<Cell_type, Jacobian_type>::build_elemental_jacs(len_data);
  }

  void build_table() override{
    this->fill_behaviour->fill(this->table_type->get_table, this->elemental_jacobian_chain);
  }

};

template<class Cell_type, class Jacobian_type>
class MFDJCPB: public Solver<Cell_type, Jacobian_type>{
 public:
  MFDJCPB(std::size_t chain_len, std::size_t dim_lb, std::size_t dim_ub,
      std::size_t n_E_lb, std::size_t n_E_ub, std::size_t mem_limit):
    Solver<Cell_type, Jacobian_type>(chain_len, dim_lb, dim_ub, n_E_lb, n_E_ub),
    memory_limit(mem_limit){
    assert((std::is_same_v<Cell_type, cell_MFDJCPB>));
    assert(!(std::is_same_v<Jacobian_type, Base_Jacobian>));

    this -> gen_type = std::make_unique<n_m_n_E_Generator>(chain_len, dim_lb, dim_ub,
        n_E_lb, n_E_ub);

    this -> table_type = std::make_unique<Table<Cell_type>>(chain_len,
        (chain_len+1) * chain_len / 2);

    this -> fill_behaviour = std::make_unique<fill_MFDJCP<Cell_type, Jacobian_type>>();
  }

  void build_elemental_jacs(std::size_t len_data=3) override{
    //n (input dimension), m (output dimensioni), n_E (number of edges in the DAG)
    Solver<Cell_type, Jacobian_type>::build_elemental_jacs(len_data);
  }

  void build_table() override{
    this->fill_behaviour->fill(this->table_type->get_table, this->elemental_jacobian_chain, memory_limit);
  }

 private:
  std::size_t memory_limit;
};

#endif
