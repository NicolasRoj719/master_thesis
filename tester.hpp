#include <memory>
#include "generator.hpp"
#include "jacobian.hpp"
#include "dptable.hpp"

template<Class Cell_type, Class Jacobian_type>
class Solver{
 public:
  virtual ~Solver() = default;
  virtual void build_elemental_jac_chain() = 0;
  virtual void build_table() = 0;

  std::unique_ptr<behaviour_fill_dptable<Class Cell_type, Class Jacobian_type>> ptr_fill_dptable;

  void fill(std::vector<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs){
    ptr_fill_dptable->fill(std::vector<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs);
  }

  void fill(std::vector<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs, std::size_t memory_limit){
    ptr_fill_dptable->fill(std::vector<Cell_type>& table, std::vector<Jacobian_type>& elemental_jacs, std::size_t memory_limit);
  }
  virtual void fill_table() = 0;
}
