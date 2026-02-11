#include <cstdint>
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

#ifndef JACOBIAN_HPP
#define JACOBIAN_HPP

class Jacobian{
 public:
  Jacobian(std::vector<size_t> jac_info){
    /* assert(1<jac_info.size()); */
    if(jac_info.size()<2){
      throw std::invalid_argument("Jacobian information must have at least two elements");
    }
    n_ = jac_info[0]; m_ = jac_info[1];
  }

  //Print function
  void print(){
    std::cout<<"[ "<< n_ <<' '<< m_ <<" ]\n";
  }

  std::size_t n() const noexcept {return n_;}
  std::size_t m() const noexcept {return m_;}

 protected:
  //Input dimension R^n
  std::size_t n_;
  //Output dimension R^m
  std::size_t m_;
};

class Dense_Jacobian: public Jacobian{
 public:
  Dense_Jacobian(std::vector<size_t> jac_info): Jacobian(jac_info){
    /* assert(2<jac_info.size()); */
    if(jac_info.size()<3){
      throw std::invalid_argument("Jacobian information must have at least three elements");
    }
    n_E_ = jac_info[2];
  }

  void print(){
    std::cout<<"[ "<< n_ <<' '<< m_ <<' '<<
      n_E_ <<" ]\n";
  }
 
  std::size_t n_E() const noexcept {return n_E_;}
  
 protected:
  //Number of edges in the DAG representation
  std::size_t n_E_;
};

#endif
