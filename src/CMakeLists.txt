
add_executable(model_experiment_1 model_experiment_1.cc)
target_link_libraries(model_experiment_1 pthread)

add_executable(data_experiment_1 data_experiment_1.cc)
target_link_libraries(data_experiment_1 pthread ${Boost_LIBRARIES})
