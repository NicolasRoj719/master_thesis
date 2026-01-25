#include <cstdint>
class Jacobian{
 public:
  Jacobian(std::size_t n_, std::size_t m_):n(n_), m(m_){}
  //Input dimension R^n
  std::size_t n;
  //Output dimension R^m
  std::size_t m;
};

class Jacobian_matrix_free: public Jacobian{
 public:
  Jacobian_matrix_free(std::size_t n_, std::size_t m_, std::size_t n_E_):
    Jacobian(n_, m_), n_E(n_E_){}
   // Number of edges in the DAG representation
  std::size_t n_E;
};
