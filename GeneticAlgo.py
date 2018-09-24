import math
import random
from Classes import *
from pyautogui import *

Inputs = 10; #169 inputs and 1 bias
Outputs = 9; #8 controller keys
MaxNodes = 1000000;

Population = 300;
Stale= 15;

DeltaDisjo= 2.0;
DeltaWeights = 0.4;
DeltaThreshold = 1.0;

MutateConnectionsChance = 0.25;
PerturbChance = 0.90;
CrossoverChance = 0.75;
LinkMutationChance = 2.0;
NodeMutationChance = 0.50;
BiasMutationChance = 0.40;
StepSize = 0.1;
DisableMutationChance = 0.4;
EnableMutationChance = 0.2;

xPressed = False;
sPressed = False;
zPressed = False;
aPressed = False;
uPressed = False;
dPressed = False;
lPressed = False;
rPressed = False;

ThreadCount = 0;


#activation function for each neuron
def Sigmoid(x):

    return 2 / (1 + math.pow(math.E, (-4.9 * x))) - 1;


#returns a unique index for a connection, used in crossover
def NewInnovation():

    pool.innovation = pool.innovation + 1;
    return pool.innovation;


#create a population pool
def NewPool():


    pool = Pool(List<Species>(), 0, Outputs, 0, 0, 0, 0);
    return pool;


#create a species
def NewSpecies():

    species = Species();
    species.topFitness = 0;
    species.staleness = 0;
    species.genomes = List<Genome>();
    species.averageFitness = 0;
    return species;


#create a genome
def NewGenome():

    genome = Genome(MutateConnectionsChance, LinkMutationChance, BiasMutationChance, NodeMutationChance, EnableMutationChance, DisableMutationChance, StepSize);
    return genome;


#return a complete copy of a genome, no references attached
def CopyGenome(genome):

    genome2 = NewGenome();

    for g in range(len(genome.genes)):    
        genome2.genes.append(CopyGene(genome.genes[g]));

    
    genome2.maxneuron = genome.maxneuron;
    genome2.mutationRates["connections"] = genome.mutationRates["connections"];

    genome2.mutationRates["link"] = genome.mutationRates["link"];

    genome2.mutationRates["bias"] = genome.mutationRates["bias"];

    genome2.mutationRates["node"] = genome.mutationRates["node"];

    genome2.mutationRates["enable"] = genome.mutationRates["enable"];

    genome2.mutationRates["disable"] = genome.mutationRates["disable"];


    return genome2;


#returns a randomly mutated genome
def BasicGenome():

    genome = NewGenome();
    genome.maxneuron = Inputs;
    Mutate(genome);
    return genome;


#creates a gene
def NewGene():


    gene = Gene();
    return gene;


#returns a copy of a gene
def CopyGene(gene):


    gene2 = NewGene();

    gene2.into = gene.into;
    gene2.outo = gene.outo;
    gene2.weight = gene.weight;
    gene2.enabled = gene.enabled;
    gene2.innovation = gene.innovation;

    return gene2;


#creates a neuron(equivalent to a node in the genome structure)
def NewNeuron():

    neuron = Neuron();
    return neuron;


#creates a neural network based off a genome's connections and neurons
def GenerateNetwork(genome):
    global Outputs
    global Inputs
    global MaxNodes
    network = NeuralNet();
    network.neurons = [] * (MaxNodes + Outputs)
    for i in range(MaxNodes + Outputs):
        network.neurons[i] = None;

    for i in range(Inputs):
        network.neurons[i] = NewNeuron();

    for o in range(Outputs):
        network.neurons[MaxNodes + o] = NewNeuron();
    
    genome.genes = sorted(genome.genes, lambda x: x.outo)
    for i in range(len(genome.genes)):
        gene = genome.genes[i];
        if (gene.enabled):
        
            if (network.neurons[gene.outo] == None):
            
                network.neurons[gene.outo] = NewNeuron();
            
            neuron = network.neurons[gene.outo];
            neuron.incoming.append(gene);
            if (network.neurons[gene.into] == None):
            
                network.neurons[gene.into] = NewNeuron();
            
        
    
    genome.network = network;


#returns a list of output controls based off an input list for a network
def EvaluateNetwork(network, inputs):

    inputs.append(1);
    if (len(inputs) != Inputs):
    
        print("Incorrect number of neural network inputs.");
        return [];

    for i in range(Inputs):
        network.neurons[i].value = inputs[i];

    for neuron in network.neurons:
        if (neuron == None):
            continue;
        
        sum = 0;
        for j in range(len(neuron.incoming)):
            incoming = neuron.incoming[j];
            other = network.neurons[incoming.into];
            sum += (incoming.weight * other.value);

        if (neuron.incoming.Count > 0):
        
            neuron.value = Sigmoid(sum);
        
    
    outputs = []
    for o in range(Outputs):

        if (network.neurons[MaxNodes + o].value > 0):
        
            outputs.append(True);
        
        else:
        
            outputs.append(False);
        
    
    return outputs;


#returns a list of output controls based off an input list for a network
def EvaluateNetworkDouble(network, inputs, side):

    inputs.append(side);
    if (len(inputs) != Inputs):
    
        print("Incorrect number of neural network inputs.");
        return [];

    for i in range(Inputs):
        network.neurons[i].value = inputs[i];

    for neuron in network.neurons:
        if (neuron == None):
            continue;
        
        sum = 0;
        for j in range(len(neuron.incoming)):

            incoming = neuron.incoming[j];
            other = network.neurons[incoming.into];
            sum += (incoming.weight * other.value);
        
        if (neuron.incoming.Count > 0):
        
            neuron.value = Sigmoid(sum);
        
    
    outputs = []
    for o in range(Outputs):

        outputs.append(network.neurons[MaxNodes + o].value);
    
    return outputs;


#returns the highest index of a genome's connections
def GetHighestInnovation(g):

    highest = 0;
    for i in range(len(g.genes)):
        if (g.genes[i].innovation > highest):
            highest = g.genes[i].innovation;

        
    
    return highest;


#creates child from excess/disjogenes of highest fitness genome and matching genes of random genome
def Crossover(ge1, ge2):

    g1 = None;
    g2 = None;
    if (ge2.fitness > ge1.fitness):
        g1 = ge2;
        g2 = ge1;
    
    else:
        g1 = ge1;
        g2 = ge2;
    
    child = NewGenome();
    innovations2 = []

    for i in range(pool.innovation + 1):

        innovations2.append(None);

    for i in range(len(g2.genes)):

        gene = g2.genes[i];
        innovations2[gene.innovation] = gene;
    

    for i in range(len(g1.genes)):
        gene1 = g1.genes[i];
        gene2 = innovations2[gene1.innovation];
        if (gene2 != None and random.randint(1, 3) == 1 and gene2.enabled):
            child.genes.append(CopyGene(gene2));
        else:
            child.genes.append(CopyGene(gene1));
        
    
    child.maxneuron = math.max(g1.maxneuron, g2.maxneuron);
    child.mutationRates = g1.mutationRates.copy()

    return child;


#get index of a random neuron in a genome
def RandomNeuron(genes, nonInput):

    neurons = bool[MaxNodes + Outputs];
    if (nonInput == False):
        for i in range(Inputs):
            neurons[i] = True;
        
    for i in range(Outputs):
        neurons[MaxNodes + i] = True;

    for i in range(len(genes)):

        if (nonInput == False or genes[i].into >= Inputs):
        
            neurons[genes[i].into] = True;
        
        if (nonInput == False or genes[i].outo >= Inputs):
        
            neurons[genes[i].outo] = True;
        
    
    count = 0;
    for b in neurons:
        if (b):
            count+= 1
        
    

    n = random.randint(0, count + 1);
    for i in range(len(neurons)):
        if (neurons[i]):
        
            n-= 1
            if (n == 0):
                return i;

    return 0;


#checks if a particular gene already exists in a gene set
def ContainsLink(genes, link):

    for i in range(len(genes)):
        gene = genes[i];
        if (gene.into == link.into and gene.outo == link.outo):
        
            return True;
        
    
    return False;


#either slightly perturb or randomly replace each gene in a genome
def PointMutate(genome):


    step = genome.mutationRates["step"]
    for i in range(len(genome.genes)):
        gene = genome.genes[i];
        if (random.random() < PerturbChance):
        
            gene.weight = gene.weight + random.random() * (step * 2 - step);
        
        else:
        
            gene.weight = random.random() * 4 - 2;
        
    


#create a link between unconnected neurons
def LinkMutate(genome, forceBias):

    newLink = NewGene();
    n1 = None;
    n2 = None;
    #find two neurons in genome
    while (True):
    
        n1 = RandomNeuron(genome.genes, False);
        n2 = RandomNeuron(genome.genes, True);
        #make sure not both inputs
        if (n1 < Inputs and n2 < Inputs):
        
            continue;
        
        break;
    

    neuron1 = None
    neuron2 = None
    #verify neuron1 is the input
    if (n2 < Inputs):
    
        neuron1 = n2;
        neuron2 = n1;
    
    else:
    
        neuron1 = n1;
        neuron2 = n2;
    
    newLink.into = neuron1;
    newLink.outo = neuron2;
    #if you want to modify the bias connection
    if (forceBias):
    
        newLink.into = Inputs - 1;
    
    #make sure link doesnt exist
    if (ContainsLink(genome.genes, newLink)):
    
        return;
    
    newLink.innovation = NewInnovation();
    newLink.weight = random.random() * 4 - 2;
    genome.genes.Add(newLink);


#create a neuron by splitting a link into two parts
def NodeMutate(genome):

    if (genome.genes.Count == 0):
    
        return;
    
    genome.maxneuron = genome.maxneuron + 1;
    gene = genome.genes[random.randint(0, genome.genes.Count)];
    #cancel if gene isnt enabled
    if (gene.enabled == False):
    
        return;
    
    #disable connection, create two ones
    gene.enabled = False;
    gene1 = CopyGene(gene);
    gene1.outo = genome.maxneuron;
    gene1.weight = 1;
    gene1.innovation = NewInnovation();
    gene1.enabled = True;
    genome.genes.append(gene1);
    gene2 = CopyGene(gene);
    gene2.into = genome.maxneuron;
    gene2.innovation = NewInnovation();
    gene2.enabled = True;
    genome.genes.append(gene2);


#randomly choose a gene to be enabled/disabled
def EnableDisableMutate(genome, enable):

    candidates = []
    for i in range(len(genome.genes)):
        if (genome.genes[i].enabled != enable):
        
            candidates.append(genome.genes[i]);
        
    
    if (candidates.Count == 0):
    
        return;
    
    gene = candidates[random.randint(0, candidates.Count)];
    gene.enabled = not gene.enabled;


#run all mutations on a genome
def Mutate(genome):

    #alter genome's internal mutation rates
    for i in range(len(genome.mutationRates)):
        if (random.randint(0, 2) == 0):
        
            genome.mutationRates[i] = .95 * genome.mutationRates[i]
        
        else:
        
            genome.mutationRates[i] = 1.05263 * genome.mutationRates[i]
        
    
    if (random.random() < genome.mutationRates["connections"]):
    
        PointMutate(genome);
    
    p = genome.mutationRates["link"]
    while (p > 0):
    
        if (random.random() < p):
        
            LinkMutate(genome, False);
        
        p = p - 1;

    
    p = genome.mutationRates["bias"]
    while (p > 0):
    
        if (random.random() < p):
        
            LinkMutate(genome, True);
        
        p = p - 1;
    
    p = genome.mutationRates["node"]
    while (p > 0):
    
        if (random.random() < p):
        
            NodeMutate(genome);
        
        p = p - 1;
    
    p = genome.mutationRates["enable"]
    while (p > 0):
    
        if (random.random() < p):
        
            EnableDisableMutate(genome, True);
        
        p = p - 1;
    
    p = genome.mutationRates["disable"]
    while (p > 0):
    
        if (random.random() < p):
        
            EnableDisableMutate(genome, False);
        
        p = p - 1;
    


#disjopart of species distance equation
def Disjoint(genes1, genes2):

    i1 = [] * (pool.innovation + 100);
    for i in range(len(genes1)):
        gene = genes1[i];
        i1[gene.innovation] = True;
    
    i2 = [] * (pool.innovation + 100)
    for i in range(len(genes2)):
        gene = genes2[i];
        i2[gene.innovation] = True;
    
    disjointGenes = 0;
    for i in range(len(genes1)):
        gene = genes1[i];
        if (i2[gene.innovation] == False):
        
            disjointGenes = disjointGenes + 1;
        
    for i in range(len(genes2)):
        gene = genes2[i];
        if (i1[gene.innovation] == False):
        
            disjointGenes = disjointGenes + 1;
        
    
    n = math.max(len(genes1), len(genes2));
    if (n == 0):
    
        n = 1;
    
    return disjointGenes / n;


#weight part of species distance equation
def Weights(genes1, genes2):

    i2 = Gene[pool.innovation + 100];
    for i in range(len(genes2)):
        gene = genes2[i];
        i2[gene.innovation] = gene;
    
    sum = 0;
    coincident = 0;
    for i in range(len(genes1)):
        gene = genes1[i];
        if i2[gene.innovation] != None:
        
            gene2 = i2[gene.innovation];
            sum = sum + math.Abs(gene.weight - gene2.weight);
            coincident = coincident + 1;
        
    
    return (sum / coincident);


#check if two genomes are part of the same species
def SameSpecies(genome1, genome2):

    dd = DeltaDisjo* Disjoint(genome1.genes, genome2.genes);
    dw = DeltaWeights * Weights(genome1.genes, genome2.genes);
    return (dd + dw < DeltaThreshold);


#assign a global rank to each genome, from lowest fitness to highest
def RankGlobally():

    globalList = []
    for s in range(len(pool.species)):
        species = pool.species[s];
        for g in range(len(species.genomes)):

            globalList.append(species.genomes[g]);
        
    globalList = sorted(globalList, lambda x : x.fitness)
    for g in range(len(globalList)):

        globalList[g].globalRank = g + 1;
    


#calculate the average rank of a species's genomes
def CalculateAverageFitness(species):

    total = 0;
    for g in range(len(species.genomes)):
        genome = species.genomes[g];
        total = total + genome.globalRank;
    
    species.averageFitness = total / species.genomes.Count;


#calculate the total of all average fitnesses
def TotalAverageFitness():

    total = 0;
    for s in range(len(pool.species)):
        species = pool.species[s];
        total = total + species.averageFitness;
    
    return total;


#remove a certain amount of genomes in every species
def CullSpecies(cutToOne):

    for s in range(len(pool.species)):
        species = pool.species[s];
        species.genomes = sorted(species.genomes, lambda x: x.fitness, reverse=True)
        remaining = math.ceil(len(species.genomes) / 2.0);
        if (cutToOne):
            remaining = 1;
        
        while (len(species.genomes) > remaining):
            species.genomes.pop(len(species.genomes) - 1)
    


#create a genome from a species
def BreedChild(species):

    child = None
    if (random.random() < CrossoverChance):
    
        g1 = species.genomes[random.randint(0, species.genomes.Count)];
        g2 = species.genomes[random.randint(0, species.genomes.Count)];
        child = Crossover(g1, g2);
    
    else:
    
        g = species.genomes[random.randint(0, species.genomes.Count)];
        child = CopyGenome(g);
    
    Mutate(child);
    return child;


#remove species that havent been the best lately
def RemoveStaleSpecies():

    survived = []
    for s in range(len(pool.species)):
        species = pool.species[s];
        species.genomes = sorted(species.genomes, lambda x: x.fitness, reverse=True)
        if (species.genomes[0].fitness > species.topFitness):
        
            species.topFitness = species.genomes[0].fitness;
            species.staleness = 0;

        else:
        
            species.staleness = species.staleness + 1;
        
        if (species.staleness < Stale or species.topFitness >= pool.maxFitness):
        
            survived.append(species);
        
    
    pool.species = survived;


#remove species that are terrible in global ranking 
def RemoveWeakSpecies():
    global pool
    survived = []
    sum = TotalAverageFitness()
    for s in range(len(pool.species)):
        species = pool.species[s];
        breed = math.floor(species.averageFitness / sum * Population);
        if (breed >= 1):
        
            survived.append(species);
        
    
    pool.species = survived;


#determine a species to add a genome to
def AddToSpecies(child):
    global pool
    found= False;
    for s in range(len(pool.species)):
        species = pool.species[s];
        if (found== False and SameSpecies(child, species.genomes[0])):
        
            species.genomes.Add(child);
            found= True;
        
    
    if (found== False):
    
        childSpecies = NewSpecies();
        childSpecies.genomes.append(child);
        pool.species.append(childSpecies);
    


#create a generation of genomes
def NewGeneration():
    global pool
    #cuts each species's population in half
    CullSpecies(False);
    RankGlobally();
    RemoveStaleSpecies();
    RankGlobally();
    for s in range(len(pool.species)):
        species = pool.species[s];
        CalculateAverageFitness(species);
    
    RemoveWeakSpecies();
    sum = TotalAverageFitness();
    #breed the species a certain amount of times proportional to their worth
    children = []
    for s in range(len(pool.species)):
    
        species = pool.species[s];
        breed = math.Floor(species.averageFitness / sum * Population) - 1;
        for i in range(breed):

            children.append(BreedChild(species));
        
    
    #eliminate all but the best members of each species
    CullSpecies(True);
    #breed the top members until population is full again
    while (len(childreN) + len(pool.species) < Population):
    
        species = pool.species[random.randint(0, len(pool.species))];
        children.append(BreedChild(species));
    
    #add all the children to the species
    for c in range(len(children)):
        child = children[c];
        AddToSpecies(child);
    
    pool.generation = pool.generation + 1



#create and fill a pool with random genomes
def InitializePool():
    global pool
    pool = NewPool();
    for i in range(Population):
        basic = BasicGenome();
        AddToSpecies(basic);
    
    InitializeRun();


#generate a network for the current genome
def InitializeRun():
    global pool
    species = pool.species[pool.currentSpecies];
    genome = species.genomes[pool.currentGenome];
    GenerateNetwork(genome);

#evaluate the fitness of the current genome
def EvaluateCurrent(d):

    data = d;
    sp = data[0];
    g = data[1];
    EvaluateMario();
    print("Evaluated");
    ThreadCount+= 1;


#determine the fitness of the current genome for mario
def EvaluateMario():
    global xPressed
    global sPressed
    global zPressed
    global aPressed
    global uPressed
    global dPressed
    global lPressed
    global rPressed
    species = pool.species[pool.currentSpecies];
    genome = species.genomes[pool.currentGenome];
    GenerateNetwork(genome);

    #gui controls
    # Form1.Best= pool.currentGenome;
    # Form1.BestFitness = pool.maxFitness;
    # Form1.SpeciesCount = pool.currentSpecies;
    # Form1.SpeciesTot = pool.species.Count;
    # Form1.CurrentGen = pool.generation;
    # Form1.OverrideBest(genome);

    fitness = 0;
    while (Form1.alive):
        Form1.ActivateApp("EmuHawk");
        inputt = [[]]
        for i in range(len(Form1.inputs)):
            inputt.append([])
        inputs = []
        Form1.inputs = inputt.copy()
        for i in range(len(inputt)):
            for p in range(len(inputt[0])):
                inputs.append(inputt[i][p]);


        #get buttons to press
        output = EvaluateNetwork(genome.network, inputs);
        if(fitness == 0 and output[7] == False):

            Form1.alive = False;

        str = "";
        if (output[0] == True):

            str += "x";

        if (output[1] == True):

            str += "s";

        if (output[2] == True):

            str += "a";

        if (output[3] == True):

            str += "z";

        if (output[4] == True):

            str += "u";

        if (output[5] == True):

            str += "d";

        if (output[6] == True):

            str += "l";

        if (output[7] == True):

            str += "r";

        Form1.InputString = str;
        if (output[0] == True and xPressed == False):
            keyDown('x')
            xPressed = True;

        elif (xPressed):
            keyUp('x')
            xPressed = False;

        if (output[1] == True and sPressed == False):
            keyDown('s')
            sPressed = True;

        elif (sPressed):
            keyUp('s')
            sPressed = False;

        if (output[2] == True and aPressed == False):
            keyDown('a')
            aPressed = True;

        elif (aPressed):
            keyUp('a')
            aPressed = False;

        if (output[3] == True and zPressed == False):
            keyDown('z')
            zPressed = True;

        elif (zPressed):
            keyUp('z')
            zPressed = False;

        if (output[4] == True and uPressed == False):
            keyDown('h')
            uPressed = True;

        elif (output[4] == False and uPressed):
            keyUp('h')
            uPressed = False;

        if (output[5] == True and dPressed == False):
            keyDown('b')
            dPressed = True;

        elif (output[5] == False and dPressed):
            keyUp('d')
            dPressed = False;

        if (output[6] == True and lPressed == False):
            keyDown('l')
            lPressed = True;

        elif (output[6] == False and lPressed):
            keyUp('l')
            lPressed = False;

        if (output[7] == True and rPressed == False):
            keyDown('r')
            rPressed = True;

        elif (output[7] == False and rPressed):
            keyUp('r')
            rPressed = False;
        
    
    if (xPressed):
        keyUp('x')
        xPressed = False;
    
    if (sPressed):
        keyUp('s')
        sPressed = False;
    
    if (aPressed):
        keyUp('a')
        aPressed = False;
    
    if (zPressed):
        keyUp('z')
        zPressed = False;
    
    if (uPressed):
        keyUp('u')
        uPressed = False;
    
    if (dPressed):
        keyUp('d')
        dPressed = False;
    
    if (lPressed):
        keyUp('l')
        lPressed = False;
    
    if (rPressed):
        keyUp('r')
        rPressed = False;
    

    fitness = Form1.marioX;
    if (fitness <= 0):
    
        fitness = -1;
    
    print("Fitness:" + fitness);
    genome.fitness = fitness;
    if (genome.fitness > pool.maxFitness):
    
        pool.maxFitness = genome.fitness;
    


#cycle through all the genomes
def NextGenome():
    global ThreadCount
    pool.current= pool.current+ 1;
    if (pool.current>= len(pool.species[pool.currentSpecies].genomes)):
    
        pool.current= 0;
        pool.current= pool.current+ 1;
        if (pool.current>= len(pool.species.Count)):
        
            ThreadCount = 0;
            NewGeneration();
            pool.best = None;
            pool.maxFitness = 0;
            pool.current= 0;
        
    


#check if a genome's fitness has already been measured
def FitnessAlreadyMeasured():
    global pool
    species = pool.species[pool.currentSpecies];
    genome = species.genomes[pool.currentGenome];

    return (genome.fitness != 0);


#main training loop
def Run():
    global pool
    InitializePool();   
    while (True):
    
        #cycle through genomes until unmeasured one is found
        while (FitnessAlreadyMeasured()):
        
            NextGenome();
        
        #select genome and species
        species = pool.species[pool.currentSpecies];
        genome = species.genomes[pool.currentGenome];
        #evaluate
        tt = [pool.currentSpecies, pool.current];
        EvaluateCurrent(tt);

    

