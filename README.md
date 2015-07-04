# cuEVOGENET

This is a Final-year Project for a BSc Computer Engineering at Universidad de Sevilla, passed with distinction and looking forward to presenting it to Best Proyect Award as suggested by the university committee.

cuEVOGENET implements a Genetic Algorithm proposed by D. Aguilar-Hidalgo et.at. on [COMPLEX NETWORKS EVOLUTIONARY DYNAMICS USING GENETIC ALGORITHMS] [1]

We use CUDA to take advantage of the data-level parallelism attached to Genetic Algorithms achieving the maximum performance, up to 240x in our tests.

cuEVOGENET implements a wide variety of parallel genetic algoritms as island model, cellular model and hybrid model. It's also possible to choose between an elite selection and roulette wheel selection.

[1]:http://www.worldscientific.com/doi/abs/10.1142/S0218127412501568

## Run
To run the code just copy kernel.cu file into a CUDA project and run it. Be assured that cuBLAS library is linked to the project.
In order to run different configuration, just modify the constants at the top of the code.
