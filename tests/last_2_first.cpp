#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

int main(){
  std::ifstream i_file("sparse_data");
  std::ofstream o_file("sparse_data_reversed");
  std::vector<std::string> lines;
  std::string line;
  std::getline(i_file,line);
  o_file << line << '\n';
  while(std::getline(i_file,line)){
    lines.push_back(line);
  }
  std::size_t num_lines = lines.size();
  for(std::size_t i = 0; i<num_lines; i++){
    o_file << lines.back() << '\n';
    lines.pop_back();
  }
  o_file.close();
  return 0;
}
