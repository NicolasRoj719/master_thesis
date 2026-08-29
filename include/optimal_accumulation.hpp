#ifndef OPTIMAL_ACCUMULATION_HPP
#define OPTIMAL_ACCUMULATION_HPP

#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>

#include "./fill_table.hpp"
#include "./table.hpp"
#include "./table_cell.hpp"


class operation_node{
 public:

    operation_node(std::size_t j, Operation op):
        index_j(j), operation_(op){}

    operation_node(std::size_t j, std::size_t i):
        index_j(j), index_i(i){}

    operation_node(std::size_t j, std::size_t i, Operation op):
        index_j(j), index_i(i), operation_(op){}

    std::size_t get_j() const {return index_j;}

    std::optional<std::size_t> get_i() const {return index_i;}

    std::optional<Operation> operation() const {return operation_;}

    void print() const{

        if(!index_i){
            
            std::cout << "\"[" << index_j << "]: ";
            std::cout << *operation_ << '"';
        }

        else if(operation_){
            
            std::cout << "\"[" << index_j << "," << *index_i;
            std::cout << "]: " << *operation_ <<'"';
        }

        else{
            
            std::cout << "\"[" << index_j << "," << *index_i <<"]\"";
        }
    }

 protected:
    std::size_t index_j;
    std::optional<std::size_t> index_i;
    std::optional<Operation> operation_;
};

template <class Jacobian_T>
class serial_accumulation_sequence{
 public:
    serial_accumulation_sequence(const Table<Jacobian_T>& table,
                                std::size_t chain_length){

        operation_sequence.reserve(chain_length);
        get_accumulation_sequence(table, chain_length);
    }

    const operation_node operator[](std::size_t index){
        
        return operation_sequence[index];
    }

    std::size_t size(){

        return operation_sequence.size();
    }

    void parser_sequence_2_graphviz_format();
 
 protected:
    std::vector<operation_node> operation_sequence;

    void get_accumulation_sequence(const Table<Jacobian_T>& table,
                                    std::size_t chain_length);

    void emplace_back_operation_node(const Table<Jacobian_T>& table,
                                    std::size_t j);

    void emplace_back_operation_node(const Table<Jacobian_T>& table,
                                    std::size_t j, std::size_t i);

};

template <class Jacobian_T>
void serial_accumulation_sequence<Jacobian_T>::emplace_back_operation_node(
        const Table<Jacobian_T>& table, std::size_t j){

    operation_sequence.emplace_back(operation_node{j, table.get_cell(j).operation()});
}

template <class Jacobian_T>
void serial_accumulation_sequence<Jacobian_T>::emplace_back_operation_node(
        const Table<Jacobian_T>& table, std::size_t j, std::size_t i){

    if constexpr (std::is_same_v<Jacobian_T, Jacobian>){

        if(j == i){}
            
        else{

            operation_sequence.emplace_back(operation_node{j,i});

            std::size_t k = table.get_cell(j,i).split_position();
            //Subproblem with lower indices are emplaced first.
            emplace_back_operation_node(table, k, i);

            emplace_back_operation_node(table, j, k+1);

        }
    }

    else{

        if(j == i){
            emplace_back_operation_node(table, j);
        }

        else{

            Operation operation = table.get_cell(j,i).operation();

            operation_sequence.emplace_back(operation_node{j, i, operation});

            std::size_t k = table.get_cell(j,i).split_position();

            if(operation == Operation::MULTIPLICATION){

                //Subproblem with lower indices are emplaced first.
                emplace_back_operation_node(table, k, i);

                emplace_back_operation_node(table, j, k+1);
            }

            else if(operation == Operation::TANGENT){

                emplace_back_operation_node(table, k, i);
            }

            else if(operation == Operation::ADJOINT){
                
                emplace_back_operation_node(table, j, k+1);
            }
        }

    }
}

template <class Jacobian_T>
void serial_accumulation_sequence<Jacobian_T>::get_accumulation_sequence(
        const Table<Jacobian_T>& table, std::size_t chain_length){

    emplace_back_operation_node(table, chain_length - 1, 0);
}

template <class Jacobian_T>
void serial_accumulation_sequence<Jacobian_T>::parser_sequence_2_graphviz_format(){

    if constexpr(!std::is_same_v<Jacobian_T, Jacobian>){

        //Graphviz format.
        std::cout << "digraph G {\n";
        std::cout << "ordering = \"out\"\n";

        std::vector<operation_node> parent_stack;

        bool previous_node_was_a_leaf = false;

        operation_sequence[0].print();
        std::cout << " -> ";

        if(*operation_sequence[0].operation() == Operation::MULTIPLICATION){

            parent_stack.emplace_back(operation_sequence[0]);
        }

        for(std::size_t idx = 1; idx < operation_sequence.size(); idx++){

            //Non leaf node
            if(operation_sequence[idx].get_i()){

                //previous node is its parent.
                if(!previous_node_was_a_leaf){

                    operation_sequence[idx].print();
                    std::cout << '\n';
                }

                //previous node is a leaf on another branch.
                else{

                    previous_node_was_a_leaf = false;

                    //Parent was visited previously.
                    parent_stack.back().print();
                    parent_stack.pop_back();
                    std::cout << " -> ";
                    operation_sequence[idx].print();
                    std::cout << '\n';
                }

                //It has at least one child
                operation_sequence[idx].print();
                std::cout << " -> ";

                if(*operation_sequence[idx].operation() ==
                        Operation::MULTIPLICATION){

                    parent_stack.emplace_back(operation_sequence[idx]);
                }
            }

            else{
                //Last node is its parent.
                if(!previous_node_was_a_leaf){

                    operation_sequence[idx].print();
                    std::cout << '\n';
                }

                else{

                    parent_stack.back().print();
                    parent_stack.pop_back();
                    std::cout << " -> ";
                    operation_sequence[idx].print();
                    std::cout << '\n';
                }

                //Tells next node that previous node was a leaf.
                previous_node_was_a_leaf = true;
            }
        }

        std::cout<< "}\n";
    }

    else{
        //Graphviz format.
        std::cout << "digraph G {\n";
        std::cout << "ordering = \"out\"\n";

        std::vector<operation_node> parent_stack;
        parent_stack.reserve(operation_sequence.size()/2 + operation_sequence.size()%2);

        bool previous_node_was_a_leaf = false;

        operation_sequence[0].print();
        std::cout << " -> ";

        parent_stack.emplace_back(operation_sequence[0]);

        for(std::size_t idx = 1; idx < operation_sequence.size(); idx++){

            //Some nodes may have a single child.
            if((*operation_sequence[idx].get_i() == *parent_stack.back().get_i()) &&
                    operation_sequence[idx].get_j() + 1 == parent_stack.back().get_j()){

                parent_stack.pop_back();
            }

            if((operation_sequence[idx].get_j() == parent_stack.back().get_j()) &&
                    *operation_sequence[idx].get_i() - 1 == *parent_stack.back().get_i()){
                
                parent_stack.pop_back();
            }



            //Non leaf node
            if(operation_sequence[idx].get_j() >
                *operation_sequence[idx].get_i() + 1){

                //previous node is its parent.
                if(!previous_node_was_a_leaf){

                    operation_sequence[idx].print();
                    std::cout << '\n';
                }

                //previous node is a leaf on another branch.
                else{
                    
                    //For the next node current node is not a leaf.
                    previous_node_was_a_leaf = false;

                    //Parent was visited previously.
                    parent_stack.back().print();
                    parent_stack.pop_back();
                    std::cout << " -> ";
                    operation_sequence[idx].print();
                    std::cout << '\n';
                }

                //Since the node is not a leaf it has children.
                operation_sequence[idx].print();
                std::cout << " -> ";

                //Assumption: every node has two children. If the node has a single child
                //then the node will be pop out inside the two ifs at the beginning of the
                // for loop.
                parent_stack.emplace_back(operation_sequence[idx]);

            }
        

            else{
                //previous node is its parent
                if(!previous_node_was_a_leaf){

                    operation_sequence[idx].print();
                    std::cout << '\n';
                }

                else{
                    //Parent was visited previously.
                    parent_stack.back().print();
                    parent_stack.pop_back();
                    std::cout << " -> ";
                    operation_sequence[idx].print();
                    std::cout << '\n';
                }

                //Tells next node that previous node was a leaf.
                previous_node_was_a_leaf = true;
            }
        }

        std::cout<< "}\n";
    }

}

#endif
