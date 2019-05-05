# SuperNEAT-Python

My first attempt at AI back in high school. The project implements a genetic algorithm called Neuroevolution of Augmenting Topologies (NEAT), which mimics biology in order to evolve neural networks towards an optimum solution. The neural networks are special in that instead of being made up of layers, they have a hidden "web" layer where neurons are free to connect wherever they want, forming neural circuits that feed back to each other.

Neural networks are encoded into objects as "chromosomes", where their genes represent the neural connections. The NEAT algorithm performs functions on a set of these "chromosomes", such as point-mutation, deletions, substitutions, and crossing-over. The networks are then tested on the game/objective, a fitness is calculated, and the process repeats.

For testing on Mario, link to lua script to the BizHawk emulator, run the C# program, and then run the lua script. Then click the mario button on the GUI, and hit start.
