#include <vector>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <memory>
#include <stdexcept>
#include "table_cell.hpp"
#include "table.hpp"
#include "node.hpp"

#ifndef BEH_OPT_ACCUMULATION
#define BEH_OPT_ACCUMULATION

//sequence vector must be an empty vector during the first call.
template <class Cell_type, class Jacobian_type>
void get_accumulation_seq(std::size_t j, std::size_t i,
    std::vector<Node<Cell_type>>& sequence, Table<Cell_type>& table, 
    std::vector<Jacobian_type>& jac_chain, int seq_idx = -1){
    /* sequence.push_back(&table(j,i)); */
    std::size_t cost = 0;
    //variable used inside the for loops
    std::size_t idx;
    //sequence index
    std::size_t aux_seq_idx;
    if(j>i){
        std::size_t k = table(j,i).k;
        switch(table(j,i).operation){
            case Operation::MULTIPLICATION:
                cost = jac_chain[j].m() * jac_chain[k].m() * jac_chain[i].n();
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(&table(j,i), j, i, cost, 0, seq_idx);
                }
                else{
                    sequence.emplace_back(&table(j,i), j, i, cost, seq_idx);
                }

                aux_seq_idx = sequence.size() - 1;
                seq_idx = static_cast<int>(aux_seq_idx);
                get_accumulation_seq(j, k+1, sequence, table, jac_chain, seq_idx);
                get_accumulation_seq(k, i, sequence, table, jac_chain, aux_seq_idx);

                break;
            case Operation::TANGENT:
                for(idx = k+1; idx<=j; idx++){
                    cost += jac_chain[idx].n_E();    
                }
                cost = cost*jac_chain[i].n();
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(&table(j,i), j, i, cost, 0, seq_idx);
                }
                else{
                    sequence.emplace_back(&table(j,i), j, i, cost, seq_idx);
                }

                seq_idx++;
                get_accumulation_seq(k, i, sequence, table, jac_chain, seq_idx);

                break;
            case Operation::ADJOINT:
                for(idx = i; idx<=k; idx++){
                    cost += jac_chain[idx].n_E();
                }
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(&table(j,i), j, i, cost*jac_chain[j].m(), cost, seq_idx);
                }
                else{
                    cost = cost*jac_chain[j].m();
                    sequence.emplace_back(&table(j,i), j, i, cost, seq_idx);
                }
                seq_idx++;
                get_accumulation_seq(j, k+1, sequence, table, jac_chain, seq_idx);

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
                cost= jac_chain[j].n_E() * jac_chain[j].n();
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(&table(j,i), j, i, cost, 0, seq_idx);
                }
                else{
                    sequence.emplace_back(&table(j,i), j, i, cost, seq_idx);
                }

                break;
            case Operation::ADJOINT:
                cost= jac_chain[j].n_E();
                if constexpr (std::is_same_v<Cell_type, cell_MFDJCPB>){
                    sequence.emplace_back(&table(j,i), j, i, cost* jac_chain[j].m(), cost, seq_idx);
                }
                else{
                    cost = cost*jac_chain[j].m();
                    sequence.emplace_back(&table(j,i), j, i, cost, seq_idx);
                }

                break;
            default:
                throw std::logic_error("Invalid Operation in table entry ("+std::to_string(j)+
                        ","+std::to_string(i)+").");
                break;
        }
    }
}

template<class Cell_type>
void graphviz_parser(std::ofstream& out, std::vector<Node<Cell_type>>& sequence){
    out<<"digraph G { \n";
    std::size_t parent_idx;
    //Iterate backwards on sequence vector
    for (auto it = sequence.rbegin(); it != sequence.rend(); it++){
        // If parent_ptr is equal to nullptr parent pointer is the root node.
        if(it->get_parent_idx() > -1){
            parent_idx = it->get_parent_idx();
            out<<sequence[parent_idx]<< " -> " << *it << '\n';
        }
    }
    out<<"}\n";
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
