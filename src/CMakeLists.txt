
# Ubuntu

cmake_minimum_required(VERSION 3.0)
project(CARMI)
set(CMAKE_CXX_STANDARD 14)

file(GLOB stx_btree_sources ../stx_btree/*.h)
add_library(stx_btree STATIC ${stx_btree_sources})
target_include_directories(stx_btree PUBLIC stx_btree)
set_target_properties(stx_btree PROPERTIES LINKER_LANGUAGE CXX)
aux_source_directory(experiment/workload source_list_workload)

add_executable(CARMI experiment.cpp experiment/core.cpp experiment/main_experiment.cpp ${source_list_workload} )
target_link_libraries(CARMI stx_btree)