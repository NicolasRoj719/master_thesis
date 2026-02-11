//#include <iostream> included in jacobian.hpp
#include <vector>
#include <cstdint>
#include "./../jacobian.hpp"

void print_vector(std::vector<size_t> v){
  std::cout<<"jac information: [ ";
  for(std::size_t i=0; i<v.size(); i++){
    std::cout<<v[i]<<' ';
  }
  std::cout<<"]\n";
}

int main(){
  //Testing Jacobian constructor
  std::vector<std::size_t> jac {3,4};
  print_vector(jac);
  Jacobian j_0{jac};
  std::cout<<"j_0.n() = "<<j_0.n()<<", j_0.m() = "<<j_0.m()<<'\n';

  //Testing Dense_Jacobian constructor
  jac.push_back(20);
  print_vector(jac);
  Dense_Jacobian j_1{jac};
  std::cout<<"j_1.n() = "<<j_1.n()<<", j_1.m() = "<<j_1.m()<< ", j_1.n_E() = "<<j_1.n_E()<<'\n';

  //Testing with a jacobian with more information than required.
  std::vector<std::size_t> jac_ {20,35,50,70,80};
  print_vector(jac_);
  Jacobian j_2{jac_};
  std::cout<<"j_2.n() = "<<j_2.n()<<", j_2.m() = "<<j_2.m()<<'\n';

  Dense_Jacobian j_3{jac_};
  std::cout<<"j_3.n() = "<<j_3.n()<<", j_3.m() = "<<j_3.m()<< ", j_3.n_E() = "<<j_3.n_E()<<'\n';

  //Testing throw error
  /* std::vector<size_t> error; */
  /* Jacobian j_error{error}; */
  /* Dense_Jacobian j_error_0{error}; */

  //Testing Dense_Jacobian error
  /* jac.pop_back(); */
  /* Dense_Jacobian j_error{jac}; */

  return 0;
}
