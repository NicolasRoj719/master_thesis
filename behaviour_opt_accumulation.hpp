#include <vector>
#include <iostream>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "table_cell.hpp"
#include "table.hpp"
#include "node.hpp"

#ifndef BEH_OPT_ACCUMULATION
#define BEH_OPT_ACCUMULATION

template <class Cell_type, class Jacobian_type>
void get_accumulation_seq(std::size_t j, std::size_t i,
    std::vector<Node<Cell_type>>& sequence, Table<Cell_type>& table, 
    std::vector<Jacobian_type>& jac_chain, Cell_type* parent_ptr = nullptr){
    /* sequence.push_back(&table(j,i)); */
    std::size_t cost = 0;
    if(j>i){
        std::size_t k = table(j,i).k;
        switch(table(j,i).operation){
            case Operation::MULTIPLICATION:
                cost = jac_chain(j).m() * jac_chain(k).m() * jac_chain(i).n();
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(parent_ptr, &table(j,k+1), &table(k,i), &table(j,i), j, i, cost, 0);
                }
                else{
                    sequence.emplace_back(parent_ptr, &table(j,k+1), &table(k,i), &table(j,i), j, i, cost);
                }

                get_accumulation_seq(j, k+1, sequence, table, jac_chain, &table(j,i));
                get_accumulation_seq(k, i, sequence, table, jac_chain, &table(j,i));

                break;
            case Operation::TANGENT:
                for(std::size_t idx = k+1; idx<=j; idx++){
                    cost += jac_chain(idx).n_E();    
                }
                cost = cost*jac_chain(i).n();
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(parent_ptr, &table(k,i), &table(j,i), j, i, cost, 0);
                }
                else{
                    sequence.emplace_back(parent_ptr, &table(k,i), &table(j,i), j, i, cost);
                }

                get_accumulation_seq(k, i, sequence, table, jac_chain, &table(j,i));

                break;
            case Operation::ADJOINT:
                for(std::size_t idx = i; idx<=k; idx++){
                    cost += jac_chain(idx).n_E();
                }
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(parent_ptr, &table(j,k+1), &table(j,i), j, i, cost*jac_chain(j).m(), cost);
                }
                else{
                    cost = cost*jac_chain(j).m();
                    sequence.emplace_back(parent_ptr, &table(j,k+1), &table(j,i), j, i, cost);
                }

                get_accumulation_seq(j, k+1, sequence, table, jac_chain, &table(j,i));

                break;
            default:
                throw std::logic_error("Invalid Operation in table entry ("+std::to_string(j)+
                        ","+std::to_string(i)+").");
                break;
        }
    }

    else{
        switch(table(j,i).operation){
            case Operation::TANGENT:
                cost= jac_chain(j).n_E() * jac_chain(j).n();
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(parent_ptr, nullptr, &table(j,i), j, i, cost, 0);
                }
                else{
                    sequence.emplace_back(parent_ptr, nullptr, &table(j,i), j, i, cost);
                }

                break;
            case Operation::ADJOINT:
                cost= jac_chain(j).n_E();
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(parent_ptr, nullptr, &table(j,i), j, i, cost* jac_chain(j).m(), cost);
                }
                else{
                    cost = cost*jac_chain(j).m();
                    sequence.emplace_back(parent_ptr, nullptr, &table(j,i), j, i, cost);
                }

                break;
            default:
                throw std::logic_error("Invalid Operation in table entry ("+std::to_string(j)+
                        ","+std::to_string(i)+").");
                break;
        }
    }
}

void graphviz_parser(std::ostream& out, std::vector<Node<Cell_type>>& sequence){
    out<<"digraph G { \n";
    //Iterate backwards on sequence vector
    for (auto it = sequence.rbegin(); it<sequence.rend(); it++){
        
    }

    out<<"}";
} 



/* template<class Cell_type> */
/* void graphviz_parser(std::vector<Node<Cell_type>>& node_sequence){ */

/* } */


/* template <class Cell_type> */
/* void print_acccumulation_sequence(std::vector<Cell_type*>& sequence){ */
/*     std::cout<<"Optimal accumulation sequence: \n"; */
/*     for(auto it = sequence.rbegin(); it<sequence.rend(); it++){ */
/*         (*it)->print(); */
/*     } */
/* } */
#endif
