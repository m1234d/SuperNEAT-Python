import math
import random
from Classes import *
from keys import *

Inputs = 170; #169 inputs and 1 bias
Outputs = 9; #8 controller keys
MaxNodes = 10000;

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

inputs = []
marioX = 0
isAlive = 0
import win32gui
 
def windowEnumerationHandler(hwnd, top_windows):
    top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))
 
def setFocus(name):
    results = []
    top_windows = []
    win32gui.EnumWindows(windowEnumerationHandler, top_windows)
    for i in top_windows:
        if name in i[1].lower():
            win32gui.ShowWindow(i[0],5)
            win32gui.SetForegroundWindow(i[0])
            break

#activation function for each neuron
def Sigmoid(x):

    return 2 / (1 + math.pow(math.e, (-4.9 * x))) - 1;


#returns a unique index for a connection, used in crossover
def NewInnovation():

    pool.innovation = pool.innovation + 1;
    return pool.innovation;


#create a population pool
def NewPool():
    pool = Pool([], 0, Outputs, 0, 0, 0, 0);
    return pool;


#create a species
def NewSpecies():

    species = Species();
    species.topFitness = 0;
    species.staleness = 0;
    species.genomes = []
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
    network.neurons = [None] * (MaxNodes + Outputs)
    for i in range(MaxNodes + Outputs):
        network.neurons[i] = None;

    for i in range(Inputs):
        network.neurons[i] = NewNeuron();

    for o in range(Outputs):
        network.neurons[MaxNodes + o] = NewNeuron();
    
    genome.genes = sorted(genome.genes, key=lambda x: x.outo)
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

        if (len(neuron.incoming) > 0):
        
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
        
        if (len(neuron.incoming) > 0):
        
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
        if (gene2 != None and random.randint(1, 2) == 1 and gene2.enabled):
            child.genes.append(CopyGene(gene2));
        else:
            child.genes.append(CopyGene(gene1));
        
    
    child.maxneuron = max(g1.maxneuron, g2.maxneuron);
    child.mutationRates = g1.mutationRates.copy()

    return child;


#get index of a random neuron in a genome
def RandomNeuron(genes, nonInput):
    global Inputs
    global Outputs
    global MaxNodes
    neurons = [None] * (MaxNodes + Outputs)
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


    n = random.randint(0, count);
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
    genome.genes.append(newLink);


#create a neuron by splitting a link into two parts
def NodeMutate(genome):

    if (len(genome.genes) == 0):
    
        return;
    
    genome.maxneuron = genome.maxneuron + 1;
    value = random.randint(0, len(genome.genes)-1)
    gene = genome.genes[value];
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
        
    
    if (len(candidates) == 0):
    
        return;

    value = random.randint(0, len(candidates)-1)
    gene = candidates[value];
    gene.enabled = not gene.enabled;


#run all mutations on a genome
def Mutate(genome):

    #alter genome's internal mutation rates
    for i,v in genome.mutationRates.items():
        if (random.randint(0, 1) == 0):
        
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

    i1 = [None] * (pool.innovation + 100);
    for i in range(len(genes1)):
        gene = genes1[i];
        i1[gene.innovation] = True;
    
    i2 = [None] * (pool.innovation + 100)
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
        
    
    n = max(len(genes1), len(genes2));
    if (n == 0):
    
        n = 1;
    
    return disjointGenes / n;


#weight part of species distance equation
def Weights(genes1, genes2):
    i2 = [None] * (pool.innovation + 100)
    for i in range(len(genes2)):
        gene = genes2[i];
        i2[gene.innovation] = gene;
    sum = 0;
    coincident = 0;
    for i in range(len(genes1)):
        gene = genes1[i];
        if i2[gene.innovation] != None:
        
            gene2 = i2[gene.innovation];
            sum = sum + abs(gene.weight - gene2.weight);
            coincident = coincident + 1;
    if coincident == 0:
        return 1000000
    
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
        
    globalList = sorted(globalList, key=lambda x : x.fitness)
    for g in range(len(globalList)):

        globalList[g].globalRank = g + 1;
    


#calculate the average rank of a species's genomes
def CalculateAverageFitness(species):

    total = 0;
    for g in range(len(species.genomes)):
        genome = species.genomes[g];
        total = total + genome.globalRank;
    
    species.averageFitness = total / len(species.genomes);


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
        species.genomes = sorted(species.genomes, key=lambda x: x.fitness, reverse=True)
        remaining = math.ceil(len(species.genomes) / 2.0);
        if (cutToOne):
            remaining = 1;
        
        while (len(species.genomes) > remaining):
            species.genomes.pop(len(species.genomes) - 1)
    


#create a genome from a species
def BreedChild(species):

    child = None
    if (random.random() < CrossoverChance):
    
        g1 = species.genomes[random.randint(0, len(species.genomes)-1)];
        g2 = species.genomes[random.randint(0, len(species.genomes)-1)];
        child = Crossover(g1, g2);
    
    else:
    
        g = species.genomes[random.randint(0, len(species.genomes)-1)];
        child = CopyGenome(g);
    
    Mutate(child);
    return child;


#remove species that havent been the best lately
def RemoveStaleSpecies():

    survived = []
    for s in range(len(pool.species)):
        species = pool.species[s];
        species.genomes = sorted(species.genomes, key=lambda x: x.fitness, reverse=True)
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
        
            species.genomes.append(child);
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
        breed = math.floor(species.averageFitness / sum * Population) - 1;
        for i in range(breed):

            children.append(BreedChild(species));
        
    
    #eliminate all but the best members of each species
    CullSpecies(True);
    #breed the top members until population is full again
    while (len(children) + len(pool.species) < Population):
    
        species = pool.species[random.randint(0, len(pool.species)-1)];
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
        print(i)
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
    global ThreadCount
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
    
    global inputs
    global marioX
    global isAlive
    
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
    while(isAlive == 1):
        pass
        
    fitness = 0;
    while (isAlive == 0): #isAlive is set by main.py
        setFocus("EmuHawk");


        #get buttons to press
        inputs2 = []
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                inputs2.append(inputs[i][j])
        output = EvaluateNetwork(genome.network, inputs2);
        if(fitness == 0 and output[7] == False):

            isAlive = 1;

        str1 = "";
        if (output[0] == True):

            str1 += "x";

        if (output[1] == True):

            str1 += "s";

        if (output[2] == True):

            str1 += "a";

        if (output[3] == True):

            str1 += "z";

        if (output[4] == True):

            str1 += "u";

        if (output[5] == True):

            str1 += "d";

        if (output[6] == True):

            str1 += "l";

        if (output[7] == True):

            str1 += "r";

        inputString = str1;
        print(inputString)
        if (output[0] == True and xPressed == False):
            keyDown(VK_X)
            xPressed = True;

        elif (xPressed):
            keyUp(VK_X)
            xPressed = False;

        if (output[1] == True and sPressed == False):
            keyDown(VK_S)
            sPressed = True;

        elif (sPressed):
            keyUp(VK_S)
            sPressed = False;

        if (output[2] == True and aPressed == False):
            keyDown(VK_A)
            aPressed = True;

        elif (aPressed):
            keyUp(VK_A)
            aPressed = False;

        if (output[3] == True and zPressed == False):
            keyDown(VK_Z)
            zPressed = True;

        elif (zPressed):
            keyUp(VK_Z)
            zPressed = False;

        if (output[4] == True and uPressed == False):
            keyDown(VK_U)
            uPressed = True;

        elif (output[4] == False and uPressed):
            keyUp(VK_U)
            uPressed = False;

        if (output[5] == True and dPressed == False):
            keyDown(VK_D)
            dPressed = True;

        elif (output[5] == False and dPressed):
            keyUp(VK_D)
            dPressed = False;

        if (output[6] == True and lPressed == False):
            keyDown(VK_L)
            lPressed = True;

        elif (output[6] == False and lPressed):
            keyUp(VK_L)
            lPressed = False;

        if (output[7] == True and rPressed == False):
            keyDown(VK_R)
            rPressed = True;

        elif (output[7] == False and rPressed):
            keyUp(VK_R)
            rPressed = False;
        
    
    if (xPressed):
        keyUp(VK_X)
        xPressed = False;
    
    if (sPressed):
        keyUp(VK_S)
        sPressed = False;
    
    if (aPressed):
        keyUp(VK_A)
        aPressed = False;
    
    if (zPressed):
        keyUp(VK_Z)
        zPressed = False;
    
    if (uPressed):
        keyUp(VK_U)
        uPressed = False;
    
    if (dPressed):
        keyUp(VK_D)
        dPressed = False;
    
    if (lPressed):
        keyUp(VK_L)
        lPressed = False;
    
    if (rPressed):
        keyUp(VK_R)
        rPressed = False;
    

    fitness = marioX;
    if (fitness <= 0):
    
        fitness = -1;
    
    print("Fitness:" + str(fitness));
    genome.fitness = fitness;
    if (genome.fitness > pool.maxFitness):
    
        pool.maxFitness = genome.fitness;
    


#cycle through all the genomes
def NextGenome():
    global ThreadCount
    pool.currentGenome = pool.currentGenome + 1;
    if (pool.currentGenome >= len(pool.species[pool.currentSpecies].genomes)):
    
        pool.currentGenome = 0;
        pool.currentSpecies = pool.currentSpecies + 1;
        if (pool.currentSpecies >= len(pool.species)):
        
            ThreadCount = 0;
            NewGeneration();
            pool.best = None;
            pool.maxFitness = 0;
            pool.currentSpecies = 0;
        
    


#check if a genome's fitness has already been measured
def FitnessAlreadyMeasured():
    global pool
    species = pool.species[pool.currentSpecies];
    genome = species.genomes[pool.currentGenome];
    
    return (genome.fitness != 0);


#main training loop
def Run():
    global pool
    print("Initializing pool.")
    InitializePool();   
    while (True):
        #cycle through genomes until unmeasured one is found
        while (FitnessAlreadyMeasured()):
            NextGenome();
        
        #select genome and species
        species = pool.species[pool.currentSpecies];
        genome = species.genomes[pool.currentGenome];

        #evaluate
        print("Evaluating " + str(pool.currentSpecies) + ": " + str(pool.currentGenome))
        tt = [pool.currentSpecies, pool.currentGenome];
        EvaluateCurrent(tt);

    

