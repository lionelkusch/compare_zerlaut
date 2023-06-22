Dynamic analysis and comparison between Mean Ad Ex and spiking neurons

* analyse_dynamic: bifurcation analysis of the mean field
  * matlab: \\
    file of matcont (version 7.1) for numerical analysis.
  * print_figure:\\
    function for print or check the result of the bifurcation.
* check derivation: file for determining values where the mean field has negative firing rate
  * derivation: main function
  * helper function: function for plotting and run the model
* fitting procedure: function for fit the transfer function
  * create TF: create transfer function
  * function_fitting_parameters: engine generate data and fit the transfer function  
  * fitting_function_zerlaut: function for fitting the transfer function 
  * generate data: simulate neurons for generating data for the transfer function
  * parameters: parameters use in the study
  * Zerlaut: model of the mean field
* paper figure: generate figure for the paper
  * plot: figure for comparing the difference between mean field and spiking network
    * plot_figure: compare individual simulation
      * helper function: function for figure of comparison between mean field and spiking network
      * plot_low_values: main function for generating data
  * print_compare_network: compare exploration parameter 
    * print_result_compare: print result the comparison from database
* spike_oscilation: response of network simulation to an oscillatory input
  * python_file: library for the simulation and analysis
    * analysis: analyse of the spike trains\\
      analysis_global: function for analyzing\\
      result_class: create a class of saving result
    * parameters: \\
      parameter_default: default parameter for the simulation
    * print:\\
      print_exploration_analysis: plot result of parameter exploration
    * run:\\
      run_exploration: main function for simulation and analyse the simulation\\
      script: concatenate the output of the simulation
    * simulation:\\
      simulation_ring: configure and simulate the network
  * simulation: result of simulation
    * simulation/simulation_b_60: data folder
    * rate_first/rate_rate/rate_rate_1: test for simulation
    * rate_rate_full: run for simulation
* static: simulation with Poisson generator
  * python_file: library for the simulation and analysis
    * analysis: analyse of the spike trains\\
      analysis_global: function for analyzing\\
      result_class: create a class of saving result
    * parameters:\\
      parameter_default: default parameter for the simulation
    * plot:\\
      helper_function: function for helping to change the data\\
      plot_compare_rate_meanfiringrate: compare bifurcation and network steady state\\
      plot_firing_rate_seed: plot variability of result from different seed \\
      plot_mean_firing_rate_std: generate b_mean_var.npy and the plot evolution of the steady state\\
      print_reduce_external: generate file for exploration of coexistence fixed point and plot it\\
      print_spike_train: plot the spike trains\\
      print_study_time_std: generate the data of the variability of the measure of the mean firing rate
                            and plot it
    * run:\\
      run_exploration: main function for simulation and analyse the simulation\\
      script: concatenate the output of the simulation\\
      script_partial: concatenate the output of the simulation in the middle of the simulation
    * simulation:\\
      simulation: configure and simulate the network\\
      simulation_time_evolve: configure and simulate the network where the rate of the poisson generator change over the simulation
  * simulation: result of simulation
    * data: folder with simulation (long, master_seed_X, short, time_reduce, time_reduce_low)\\
      run_first: run simulation for parameter exploration (long, short, master_seed_0)\\
      run_second: run simulation for estimate coexistence of high fixed point (200Hz), time_reduce\\
      run_third: run simulation for estimate coexistence of middle fixed point (50Hz), time_reduce_low
* zerlaut_oscillation: response of mean field to an oscillatory input  
  * python_file: library for the simulation and analysis
    * analysis: analyse of the spike trains\\
      analysis_global: function for analyzing\\
      insert_database: function for inserting the result of the analysis in a database
    * parameters:\\
      parameter_default: default parameter for the simulation
    * plot:\\
      compare_result: first try of figure for compare individual simulation\\
      print_one: figure of the output of the simulation\\
      print_result: result from database
    * run:\\
      tools_simulation: tool for the simulation, get the result, plot result\\
      zerlaut: model of the mean field 
    * simulation:\\
      launch_one: run simulations in serial\\
      launch_simulation: run exploration simulation in parallel
  * simulation: result of simulation\\
    * deterministic: result of the mean field without noise\\
    * stochastic_1e-08: result of simulation with noise (variance 1e-8)\\
    * stochastic_1e-09: result of simulation with noise (variance 1e-9)
      
