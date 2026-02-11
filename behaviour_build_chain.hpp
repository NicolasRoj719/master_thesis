#include <cstdint>
#include <vector>
#include "jacobian.hpp"

template <class Jacobian_type>
class behaviour_build_chain{
 public:
  virtual ~behaviour_build_chain() = default;
  
  virtual void build_chain(const std::vector<std::vector<std::size_t>>& problem_data) = 0;

  std::vector<Jacobian_type> get_chain(){
    return jacobian_chain;
  }

  void print(){
    for(std::size_t i=0; i<jacobian_chain.size(); i++){
      std::cout<<"F'_"<<i<<' ';
      jacobian_chain[i].print();
    }
  }

  std::size_t size(){
    return(jacobian_chain.size());
  }

 protected:
    std::vector<Jacobian_type> jacobian_chain;
};

template <class Jacobian_type>
class build_chain_dense: public behaviour_build_chain<Jacobian_type>{
 public:
  void build_chain(const std::vector<std::vector<std::size_t>>& problem_data) override{
    for(std::size_t i=0; i<problem_data.size(); i++){
      this->jacobian_chain.emplace_back(problem_data[i]);
    }
   }
};
