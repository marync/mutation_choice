import copy as cp
import numpy as np
import itertools

from combinatorial_utility import recode_library, fit_regression_fourier, powerset, int_to_string, compute_Ak
from maxima_functions import count_local_maxima_step
from path_functions import call_paths
from HOC_functions import try_n_times

class RMF :
    """
    RMF Landscape where additive coefficients are drawn from a Normal distribution.
    """
    
    def __init__(self, L, sigma_add, sigma_hoc, rng=None, seed=None, mu=0):
        """
        Args:
            length (int): Number of loci (N).
            mu (float): Mean of the additive fitness effects.
            sigma_add (float): Variation in the additive effects across loci.
            sigma_hoc (float): Scale of the House of Cards (local) noise.
        """
        
        self.L = int (L)
        self.sigma_hoc = sigma_hoc

        # random number generator
        if rng is None :
            self.rng = np.random.default_rng (seed)
        else :
            self.rng = rng
        
        # Draw N additive coefficients once
        self.coefficients = self.rng.normal (loc=mu, scale=sigma_add, size=self.L)
        #self.coefficients = np.ones (self.L) * (sigma_add**2)
        #rint (self.coefficients)

        self.landscape = dict ()

        #self.landscape = self._generate_landscape()


    def generate_libraries (self, M, weighted=False, epsilon=.01, nbatch=10000, ntries=20, scenarios=['adaptive','conditional','random','beneficial'], top=None) :
        """
        """

        self.M = int (M)

        self.ascertained = dict ()
        for key in scenarios :
            self.ascertained[key] = dict ()

        #self.ascertained['size'] = int (M)
        
        #x0, steps, ytrajectory = self.evolve (M, weighted)
        #x0, steps, ytrajectory = try_n_times (ntries, self.evolve, M, weighted)
        x0, steps, ytrajectory = self.evolve (T=M, weighted=weighted, ntries=ntries, top=top)

        if 'adaptive' in scenarios :
            self.ascertained['adaptive']['X'], self.ascertained['adaptive']['y'], Xfull = self.generate_combinatorial_library (x0, steps)
            self.ascertained['adaptive']['mutations'] = cp.deepcopy (steps)
            self.ascertained['adaptive']['A'], self.ascertained['adaptive']['E'] = self.get_fitness_components_wrapper (Xfull)
 
        if 'conditional' in scenarios :
            self.ascertained['conditional']['X'], self.ascertained['conditional']['y'], Xfull, mutations = self.make_conditional_library (x0, ytrajectory[-1], M, epsilon=epsilon, nbatch=nbatch, ntries=ntries)
            self.ascertained['conditional']['mutations'] = cp.deepcopy (mutations)
            self.ascertained['conditional']['A'], self.ascertained['conditional']['E'] = self.get_fitness_components_wrapper (Xfull)
        
        if 'random' in scenarios :
            self.ascertained['random']['X'], self.ascertained['random']['y'], Xfull = self.make_random_library (ancestor=x0, M=M)
            self.ascertained['random']['A'], self.ascertained['random']['E'] = self.get_fitness_components_wrapper (Xfull)
    
        if 'beneficial' in scenarios :
            self.ascertained['beneficial']['X'], self.ascertained['beneficial']['y'], Xfull   = self.make_beneficial_library (x0, M)
            self.ascertained['beneficial']['A'], self.ascertained['beneficial']['E'] = self.get_fitness_components_wrapper (Xfull)
        
        if 'deleterious' in scenarios :
            self.ascertained['deleterious']['X'], self.ascertained['deleterious']['y'], Xfull   = self.make_deleterious_library (x0, M)
            self.ascertained['deleterious']['A'], self.ascertained['deleterious']['E'] = self.get_fitness_components_wrapper (Xfull)



    def compute_statistics (self) :

        for scenario in self.ascertained.keys () :

            anc_idx = np.where (np.sum (self.ascertained[scenario]['X'], axis=1) == 0)[0][0]
            self.ascertained[scenario]['R2'], inter, coefs = fit_regression_fourier (self.ascertained[scenario]['X'], self.ascertained[scenario]['y'])
            self.ascertained[scenario]['Am']   = compute_Ak (inter, coefs, self.M) 
            self.ascertained[scenario]['nMax'] = count_local_maxima_step (self.ascertained[scenario]['X'],
                                                                            self.ascertained[scenario]['y'],
                                                                            anc_idx)
            self.ascertained[scenario]['nPaths'] = call_paths (self.ascertained[scenario]['X'],
                                                               self.ascertained[scenario]['y'])
        



    def get_statistics (self, scenarios=['adaptive','conditional','random','beneficial']) :

        ntypes = len (scenarios)
        
        R2    = np.zeros ((ntypes, self.M+1))
        nMax  = np.zeros ((ntypes, self.M+1))
        nPath = np.zeros (ntypes)
        Am    = np.zeros ((ntypes, self.M+1))

        for i in range (ntypes) :
            R2[i,:]   = cp.deepcopy (self.ascertained[scenarios[i]]['R2'])
            nMax[i,:] = cp.deepcopy (self.ascertained[scenarios[i]]['nMax'])
            nPath[i]  = cp.deepcopy (self.ascertained[scenarios[i]]['nPaths'])
            Am[i,:]   = cp.deepcopy (self.ascertained[scenarios[i]]['Am'])

        return R2, nMax, nPath, Am

    
    def _generate_landscape (self):
        landscape_map = {}
        genotypes = list (product([0, 1], repeat=self.L))
        
        for g_tuple in genotypes:
            genotype = np.array(g_tuple)
            
            # additive component
            additive_fitness = np.sum (genotype * self.coefficients)
            
            # hoc noise
            noise = self.rng.normal (0, self.sigma_hoc)

            landscape_map[g_tuple] = additive_fitness + noise
            
        return landscape_map

    
    def mutate (self, ancestor) :
        """
        ancestor: ancestral genotype
        """
    
        ymuts = np.zeros (self.L) * np.nan
        Muts  = np.zeros ((self.L, self.L), dtype=int)
        # generate one-step mutations
        for i in range (self.L) :
            mut_i    = cp.deepcopy (ancestor)
            mut_i[i] = int (( 0**ancestor[i] ) * ( 1**(1 - ancestor[i]))) # mutate at ith position
            
            # fitness
            Muts[i,:] = cp.deepcopy (mut_i)
            ymuts[i]  = self.get_fitness (mut_i)
    
        return ymuts, Muts

    
    def evolve (self, T, ntries=20, weighted=False, top=None) :

        # sample the ancestor
        x0 = self.rng.binomial (n=1, p=.5, size=self.L)
        #y0 = self.get_fitness (x0)

        steps, fitnesses = try_n_times (ntries, self.go_uphill, x0=x0, T=T, weighted=weighted, top=top)

        return x0, steps, fitnesses


    def go_uphill (self, x0, T, weighted=False, top=None) :
        
        # mutations
        steps   = np.ones (T, dtype=int) * (-1)
        fitness = np.zeros (T)

        # fitness of ancestor
        y0 = self.get_fitness (x0)

        # evolve
        xcur = cp.deepcopy (x0)
        ycur = y0
        step_i = -1
        for t in range (T) :
            ymuts, M = self.mutate (xcur)
                
            # choose step
            while step_i in steps :
                if weighted :
                    weights = ymuts - ycur
                    weights[ymuts < ycur] = 0
                    step_i = self.rng.choice ( np.arange (0, self.L, 1), p=(weights / np.sum (weights)))
                    
                elif top is not None :
                    sorted_mutations = np.argsort (ymuts)
                    step_i = self.rng.choice (sorted_mutations[-top:])
                
                else :
                    step_i = self.rng.choice (np.where (ymuts > ycur)[0])
                    if ymuts[step_i] < ycur :
                        print ((ymuts[step_i], ycur))    
                        raise ValueError("Step cannot be deleterious.")

                if step_i in steps :
                    raise ValueError("Recurrent mutation.")
            
            steps[t]   = step_i
            fitness[t] = ymuts[step_i]
    
            # update current state
            xcur = cp.deepcopy (M[step_i,:])
            ycur = fitness[t]

        # add ancestor to fitness
        fitness_all = np.insert (fitness, 0, y0) # add ancestral fitness
        #print (fitness_all)       
 
        return steps, fitness_all

    
    def get_fitness (self, genotype) :
        g_tuple = int_to_string (genotype)
       
        if g_tuple in self.landscape.keys () :
            fitness = self.landscape.get (g_tuple)
        else :
            fitness  = np.sum ( (2*genotype-1) * self.coefficients)
            fitness += self.rng.normal (0, self.sigma_hoc)
            
            self.landscape[g_tuple] = fitness

        return fitness
        
    
    def get_fitness_components_wrapper (self, G) :
        n, M = G.shape

        A = np.zeros (n)
        E = np.zeros (n)
        for i in range (n) :
            A[i], E[i] = self.get_fitness_components (G[i,:])

        return A, E

 
    def get_fitness_components (self, genotype) :
        g_tuple = int_to_string (genotype)
       
        if g_tuple in self.landscape.keys () :
            fitness  = self.landscape.get (g_tuple)
            additive = np.sum ( (2*genotype-1) * self.coefficients)
            epsilon  = fitness - additive
        else :
            fitness = None

        return additive, epsilon




    def make_random_library (self, M, ancestor=None) :

        if ancestor is None :
            ancestor = self.rng.binomial (n=1, p=.5, size=self.L)

        mutations = self.rng.choice (self.L, size=M, replace=False)

        #print ('random library')
        #print (ancestor)
        #print (mutations)

        return self.generate_combinatorial_library (ancestor, mutations) #, self.coefficients[mutations]


    def make_beneficial_library (self, ancestor, M) :

        y_singles, M_singles = self.mutate (ancestor)

        mutations = np.flip (np.argsort (y_singles))[:M]

        return self.generate_combinatorial_library (ancestor, mutations)
    

    def make_deleterious_library (self, ancestor, M) :

        y_singles, M_singles = self.mutate (ancestor)

        mutations = np.argsort (y_singles)[:M]

        return self.generate_combinatorial_library (ancestor, mutations)
    
    
    def get_coefficients (self):
        
        """Returns the generated additive effects for inspection."""
        return self.coefficients


    def make_conditional_library (self, ancestor, y_target, M, epsilon=0.01, nbatch=1000, ntries=10) :
        """
        Need to search among all possible mutants at distance M from the ancestor for one
        that has a similar fitness to y_target.
        """

        x_cond, y_cond, mutations = self.try_finding_target (ancestor, y_target, M, epsilon, nbatch, ntries)
        #x_cond, y_cond, mutations = self.find_genotypes_at_distance_fast (ancestor, y_target, M, epsilon, nbatch)
        #print (ancestor)
        #print (mutations)
        Xrec, Ylib, Xfull = self.generate_combinatorial_library (ancestor, mutations)
        
        return Xrec, Ylib, Xfull, mutations
        

    def get_shuffled_combinations (self, M) :
        """Suitable for smaller search spaces."""

        combos = list(itertools.combinations(range(self.L), M))
       
        self.rng.shuffle(combos)
        
        return combos


    def sample_sites_batch (self, batch) :
        
        return np.array([self.rng.choice(self.L, self.M, replace=False) for _ in range (batch)])
   

    def find_genotypes_at_distance_fast (self, ancestor, target_fitness, distance, epsilon=0.01, nbatch=1000) :

        # get a batch of locus combinations
        combos = self.sample_sites_batch (batch=nbatch)
        
        success = False
        for indices in combos : 
            
            # Construct the genotype
            genotype = cp.deepcopy (ancestor)
            for idx in indices:
                genotype[idx] = int (( 0**ancestor[idx] ) * ( 1**(1 - ancestor[idx]))) 
                
            # Check fitness
            fitness = self.get_fitness(genotype)
            if abs (fitness - target_fitness) <= epsilon :
                return genotype, fitness, np.array (indices, dtype=int)

        if not success :
            raise Exception ('Target fitness not found.')
                 

    def try_finding_target (self, ancestor, target_fitness, distance, epsilon=0.01, nbatch=1000, ntries=20) :

        last_exception = None
        for attempt in range (ntries):
            try :
                return self.find_genotypes_at_distance_fast (ancestor, target_fitness, distance, epsilon, nbatch) 
            
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt + 1} failed: {e}")

        raise last_exception


    def find_genotypes_at_distance (self, ancestor, target_fitness, distance, epsilon=0.01):
        """
        Searches the shell at Hamming distance M for genotypes near a target fitness.
        
        Args:
            distance (int): Hamming distance from the reference (0,0...0).
            target_fitness (float): The fitness value we are looking for.
            epsilon (float): The tolerance range.
            
        Returns:
            list: Genotypes (as tuples) that satisfy the condition.
        """
            
        # Iterate through all possible ways to choose M mutation sites out of L
        combo_list = self.get_shuffled_combinations (distance)
        #for indices in combinations (range(self.L), distance) :
        for indices in combo_list :

            # Construct the genotype
            genotype = cp.deepcopy (ancestor)
            for idx in indices:
                genotype[idx] = int (( 0**ancestor[idx] ) * ( 1**(1 - ancestor[idx]))) 
                
            # Check fitness
            fitness = self.get_fitness(genotype)
            if abs (fitness - target_fitness) <= epsilon:
                return genotype, fitness, np.array (indices, dtype=int)
                
        return None

 
    def generate_combinatorial_library (self, x0, mutations) :
    
        #mutations_sort = np.sort (mutations)
        mutations_sort = cp.deepcopy (mutations) 
        mutant_combos  = list (powerset (mutations_sort))
        
        N = len (mutant_combos)   

        x0    = np.array (x0, dtype=int)
        Xlib  = np.reshape (np.repeat (x0, N), (len (mutant_combos), self.L), order='F')
        Y     = np.zeros (N)
        
        ct = 0
        for comb in mutant_combos :
            if len (comb) == 0 :
                Y[ct] = self.get_fitness (Xlib[ct,:])

            if len (comb) > 0 :
                for m in comb :
                    Xlib[ct,m] = ( 0**x0[m] ) * ( 1**(1 - x0[m]))
                
                Y[ct] = self.get_fitness (Xlib[ct,:])
            
            ct += 1
    
        Xsub = cp.deepcopy (Xlib[:,mutations_sort])
    
        return recode_library (x0[mutations_sort], Xsub), Y, Xlib


    def recode_library (ancestor, X) :
    
        new = X - ancestor 
        new[new != 0] = 1 
    
        return new 
