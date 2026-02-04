#include <cstdint>
#include <iostream>

#ifndef JACOBIAN_HPP
#define JACOBIAN_HPP

class Jacobian{
 public:
  Jacobian(){}
  Jacobian(std::size_t n_, std::size_t m_):
    n(n_), m(m_){}

  virtual ~Jacobian() = default;
  //Print function
  virtual void print()=0;

  // Operator overloading to access data_members.
  // Derived classes will override this function to extend the
  // funtionality of the operator []
  virtual std::size_t get_extra(std::size_t index) const{
    return 0;
  }

  virtual std::size_t& get_extra_ref(std::size_t index) {
    throw std::out_of_range("Invalid index");
  }

  std::size_t operator[](std::size_t index) const{
    switch(index){
      case 0: return n;
      case 1: return m;
      default: return get_extra(index);
    }
  }

  std::size_t& operator[](std::size_t index) {
    switch(index){
      case 0: return n;
      case 1: return m;
      default: return get_extra_ref(index);
    }
  }

 protected:
  //Input dimension R^n
  std::size_t n=0;
  //Output dimension R^m
  std::size_t m=0;
};

class Base_Jacobian:public Jacobian{
 public:
  Base_Jacobian(){}
  Base_Jacobian(std::size_t n_, std::size_t m_):Jacobian(n_, m_){}

  void print() override{
    std::cout<<"[ "<< n <<' '<< m <<" ]\n";
  }
};

class Dense_Jacobian: public Base_Jacobian{
 public:
  Dense_Jacobian(){}
  Dense_Jacobian(std::size_t n_, std::size_t m_,std::size_t n_E_):
    Base_Jacobian(n_, m_), n_E(n_E_){}

  void print() override{
    std::cout<<"[ "<< n <<' '<< m <<' '<<
      n_E <<" ]\n";
  }

  std::size_t get_extra(std::size_t index) const override{
    if(index == 2){
      return n_E;
    }
    return 0;
  }

  std::size_t& get_extra_ref(std::size_t index) override{
    if(index == 2){
      return n_E;
    }
    throw std::out_of_range("Invalid index");
  }

  //Number of edges in the DAG representation
  std::size_t n_E=0;
};

class Sparse_Jacobian: public Dense_Jacobian{
 public:
  /* Sparse_Jacobian(){}; */
};

#endif
