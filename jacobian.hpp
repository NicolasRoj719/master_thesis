#include <cstdint>
#include <iostream>
#include <memory>

class Jacobian{
 public:
  Jacobian(std::size_t n_, std::size_t m_):
    n(n_), m(m_){};
  //Input dimension R^n
  std::size_t n;
  //Output dimension R^m
  std::size_t m;
  //Print function
  virtual void print()=0;
};

class Base_Jacobian:public Jacobian{
 public:
  Base_Jacobian(std::size_t n_, std::size_t m_):Jacobian(n_, m_){}

  void print() override{
    std::cout<<'['<< n <<' '<< m <<"]\n";
  }
};

class Dense_Jacobian: public Jacobian{
 public:
  Dense_Jacobian(std::size_t n_, std::size_t m_,std::size_t n_E_):
    Jacobian(n_, m_), n_E(n_E_){}

  void print() override{
    std::cout<<'['<< n <<' '<< m <<' '<<
      n_E <<' '<<"]\n";
  }

  //Number of edges in the DAG representation
  std::size_t n_E;
};

class Sparse_Jacobian: public Dense_Jacobian{
 public:
  /* Sparse_Jacobian(){}; */
};
