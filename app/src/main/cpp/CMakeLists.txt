cmake_minimum_required(VERSION 3.22.1)
project(minimind_runtime)

add_library(minimind_runtime SHARED minimind_runtime.cc)

find_library(log-lib log)
target_link_libraries(minimind_runtime ${log-lib})

