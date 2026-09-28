# mutation_choice
Code to support "Mutation choice biases the structure of empirical fitness landscapes."

All simulations are run from jupyter notebooks in /notebooks, with supporting code in the /code folder.
- figure_1.ipynb: Generates Figure 1.
- figure_2.ipynb: Generates Figure 2, the House of Cards figure using simulation results from house_of_cards.ipynb. Results from the simulations are also provided in results/HOC/2026-09-27.
- alternative_simulation_plotting.ipynb: Generates Figures 3-5 using simulation results from RMF_object.ipynb and the LK scripts (see below). Results from the simulations are also provided in results/RMF/2026-09-27 and results/LK/2026-09-21.
- data_plotting.ipynb: Generates Figure 6 using simulations from the three combinatorially complete mutagenesis studies, published in Wu et al. 2016 and Lite et al. 2020. The simulations are conducted with data_analyis.ipynb. Results from the simulations are also provided in results/CMEs/2026-09-27/{study} for each study.
- To produce the LK simulation results:
   1. python3 run_full_sweep.py 1000 100
   2. python3 NiNijBaseline_v2.py '[1,3,7,11,15,25,40,50,55,63,75,85,95]' 1000 100
