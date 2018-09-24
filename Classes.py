class Pool:
    def __init__(self, sp, gen, innov, currentSp, currentGen, currentFr, maxFit):
        self.species = sp;
        self.generation = gen;
        self.innovation = innov;
        self.currentSpecies = currentSp;
        self.currentGenome = currentGen;
        self.best = None;
        self.currentFrame = currentFr;
        self.maxFitness = maxFit;


class Species:
    def __init__(self):
        self.genomes = []
        self.topFitness = 0
        self.staleness = 0
        self.averageFitness = 0


class Genome:
    def __init__(self, conChance, linkChance, biasChance, nodeChance, enableChance, disableChance, stepSize):
        self.genes = []
        self.fitness = 0.0;
        self.adjustedFitness = 0.0;
        self.network = NeuralNet();
        self.maxneuron = 0;
        self.globalRank = 0;
        self.mutationRates = {}
        self.mutationRates["connections"] = conChance;
        self.mutationRates["link"] = linkChance;
        self.mutationRates["bias"] = biasChance;
        self.mutationRates["node"] = nodeChance;
        self.mutationRates["enable"] = enableChance;
        self.mutationRates["disable"] = disableChance;
        self.mutationRates["step"] = stepSize;


class Gene:
    def __init__(self):
        self.into = 0;
        self.outo = 0;
        self.weight = 0.0;
        self.enabled = True;
        self.innovation = 0;


class NeuralNet:
    def __init__(self):
       self.neurons = []


class Neuron:
    def __init__(self):
        self.incoming = []
        self.loc = None
        self.value = 0.0;

