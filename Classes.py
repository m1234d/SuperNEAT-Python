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
    genes = new List<Gene>();
    double fitness = 0;
    double adjustedFitness = 0;
    NeuralNet network = new NeuralNet();
    int maxneuron = 0;
    int globalRank = 0;
    DataTable mutationRates;
    Genome(double conChance, double linkChance, double biasChance, double nodeChance, double enableChance, double disableChance, double stepSize)
    {
        mutationRates = new DataTable();
        DataColumn cl = new DataColumn("connections");
        DataColumn c2 = new DataColumn("link");
        DataColumn c3 = new DataColumn("bias");
        DataColumn c4 = new DataColumn("node");
        DataColumn c5 = new DataColumn("enable");
        DataColumn c6 = new DataColumn("disable");
        DataColumn c7 = new DataColumn("step");
        mutationRates.Columns.Add(cl);
        mutationRates.Columns.Add(c2);
        mutationRates.Columns.Add(c3);
        mutationRates.Columns.Add(c4);
        mutationRates.Columns.Add(c5);
        mutationRates.Columns.Add(c6);
        mutationRates.Columns.Add(c7);
        DataRow row = mutationRates.NewRow();
        row["connections"] = conChance;
        row["link"] = linkChance;
        row["bias"] = biasChance;
        row["node"] = nodeChance;
        row["enable"] = enableChance;
        row["disable"] = disableChance;
        row["step"] = stepSize;
        mutationRates.Rows.Add(row);
    }

}
class Gene
{
    int into = 0;
    int outo = 0;
    double weight = 0.0;
    bool enabled = true;
    int innovation = 0;
}

class NeuralNet
{
    List<Neuron> neurons = new List<Neuron>();
}

class Neuron
{
    List<Gene> incoming = new List<Gene>();
    Point loc;
    double value = 0;
}
