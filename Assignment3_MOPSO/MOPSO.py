#!/usr/bin/env python
# coding: utf-8

# In[9]:


#MOPSO
import numpy as np
GLOBAL_BEST=None


# In[10]:


def crowdingDistSorting(children, population):
    """
    Sorts the chldren belonging to population acc. to the crowding distance sorting
    NOTE - Works only for 2 obective functions
    Args:
        Children (2D array of size (?,nvar)) - the subset of population belonging to a particular pareto 
                                                which needs to be sorted
        Population (2D array of size (npop, nvar)) - the current population
    Returns : Sorted Children
    """
    
    
    costValArray = [[child.best.cost1, child.best.cost2] for child in population]

    Smax, Tmax = np.max(costValArray, axis = 0)  
    Smin, Tmin = np.min(costValArray, axis = 0)

    tmpCrowd = []
    
    tmp = np.copy(children)
    #assign inf distance to the extreme points corresponding to all objective functions
    tmp = sorted(children, key=lambda x: population[x].best.cost1)
    tmpCrowd.append((tmp[0] ,float('inf')))
    tmpCrowd.append((tmp[-1], float('inf')))
    min1 = tmp[0]
    max1 = tmp[-1]
    
    tmp = sorted(children, key=lambda x:population[x].best.cost2)
    tmpCrowd.append([tmp[0] ,float('inf')])
    tmpCrowd.append([tmp[-1], float('inf')])
    min2 = tmp[0]
    max2 = tmp[-1]
    
    extremes = {min1, min2, max1, max2}
    for idx, val in enumerate(tmp):
        if val in extremes:
            pass
        else:
            dist = abs(population[tmp[idx+1]].best.cost1 - population[tmp[idx-1]].best.cost1)/(Smax-Smin+0.000001)+                 abs(population[tmp[idx+1]].best.cost2 - population[tmp[idx-1]].best.cost2)/(Tmax-Tmin+0.000001)

            tmpCrowd.append([val, dist])
    
    tmpCrowd = np.array(tmpCrowd)
    newChildren = np.array(sorted(tmpCrowd, key = lambda x:x[1], reverse=True)).squeeze()
    newChildren = newChildren[:,0]
    idx = np.unique(newChildren, return_index = True)[1]
    newChildren = [newChildren[i] for i in sorted(idx)]
    return np.int32(newChildren)


# In[11]:


def atLeast1dominate(front, x):
    """
    Checks wheather atleast one soln in front dominates soln x
    Args:
        front (list) - list of soln(with cost values) belonging to a front
        x (array) - a solution containing the cost values
    Returns:
        True/ False
    """
    for c1, c2 in front:
        if c1>=x[0] and c2<=x[1]:
            return True
    return False
def findFront(idx, paretoFronts, costs):
    """
    Returns the front to which the solution[idx] belongs, and simultaneously updates the Pareto front
    Args:
        idx (int) - the index of the solution whose front no. is to be found
        paretoFronts (dict) - the current paretoFronts, to be updated too.
        costs (2D array)- the cost values array corresponding the current population sorted to descending
                            order of 1 cost function (here lateralSA)
    Return:
        front (int) - the front no. to which soln[idx] belongs
    Update:
        paretoFronts (dict) - assigns idx to ParetoFronts[front]
    """
    l = len(paretoFronts)
    if l==0:
        paretoFronts[1] = [idx]
        return 1
    for i in paretoFronts:
        front = paretoFronts[i]
        if atLeast1dominate([costs[t] for t in front], costs[idx])==False:
            paretoFronts[i].append(idx)
            return idx
    paretoFronts[l+1] = [idx]
    return l+1
def nonDominatedSorting(population, convert2objective):
    """
    Perfroms non-dominated sorting on the population and returns the pareto front, along with the population 
    sorted in escending order of lateralSA 
    NOTE - crowding distance sorting not applied here
    """
    nvar = len(population[0].position)
    npop = len(population)
    
    
    populationSorted = np.array(sorted(population, key = lambda x:x.best.cost2))
    costs = convert2objective(populationSorted)
    paretoFronts = {}
    
    for i in range(npop):
        findFront(i, paretoFronts, costs)
    return paretoFronts, populationSorted

def updateRank(paretoFronts):
    """
    Returns the updated rank of each solution
    Args:
        paretoFronts (dict) - current pareto fronts
    Return:
        rank (dict) - rank of each solution idx
    """
    ranks = {}
    r = 1
    for front in paretoFronts:
        for idx in paretoFronts[front]:
            ranks[idx] = r
            r+=1
    return ranks
def updatePopulation(population, paretoFronts, npop):
    newPop = []
    if len(population)<=npop:
        return population
    count=0
    flag = 0
    for front in paretoFronts:
        if flag==1:
            break
        for x in paretoFronts[front]:
            if count>=npop:
                flag=1
                break
            newPop.append(population[x])
            count+=1
    return np.array(newPop)


# In[12]:

class Best:
    def __init__(self):
        self.position = None
        self.cost1 = None
        self.cost2 = None
class particle:
    def __init__(self):
        self.position = None
        self.velocity = None
        self.cost1 = None
        self.cost2 = None
        self.best = Best()
        


# In[15]:


def initialisePop(npop, cost1, cost2, nvar, varMax, varMin):
    global  GLOBAL_BEST
    GLOBAL_BEST = Best()
    GLOBAL_BEST.cost1 = -float("inf")
    GLOBAL_BEST.cost2 = float("inf")
    population = []
    for i in range(npop):        
        x = particle()
        x.position = np.around(np.random.uniform(low=varMin, high = varMax, size = nvar), 4)
        x.velocity = np.zeros(nvar)
        x.cost1 = cost1(x.position)
        x.cost2 = cost2(x.position)
        x.best.position = x.position
        x.best.cost1 = x.cost1
        x.best.cost2 = x.cost2
        if x.best.cost1 >= GLOBAL_BEST.cost1 and x.best.cost2 <= GLOBAL_BEST.cost2:
            
            GLOBAL_BEST.position = x.best.position
            GLOBAL_BEST.cost1 = x.best.cost1
            GLOBAL_BEST.cost2 = x.best.cost2
        population.append(x)
        del x
    return population


# In[1]:


def mopsoOptimizer(cost1, cost2, applyConstraint,normalize, npop = 50, nvar = 2, varMax=1, varMin=0,
                   niter = 200, c1=2, c2=2, w=1, wdamp=0.99, maxVelocity=100, minVelocity=-1):
    
    global GLOBAL_BEST
    convert2objective = lambda x:np.array([[cost1(i.best.position), cost2(i.best.position)] for i in x])
    pareto_list = []
    
    #initialization
    population = initialisePop(npop, cost1, cost2, nvar, varMax, varMin)
    paretoFronts, population = nonDominatedSorting(population, convert2objective)
    
    #apply constraints to golbal best
    GLOBAL_BEST.position = applyConstraint(GLOBAL_BEST.position)
    GLOBAL_BEST.position = normalize(GLOBAL_BEST.position)
    GLOBAL_BEST.cost1 = cost1(GLOBAL_BEST.position)
    GLOBAL_BEST.cost2 = cost2(GLOBAL_BEST.position)
    
    #crowd sorting of initial pop
    for front in paretoFronts:          
            if len(paretoFronts[front])>1:
                paretoFronts[front] = crowdingDistSorting(paretoFronts[front], population)
    rank = updateRank(paretoFronts)
    #sort the population according to the pareto front
    population = updatePopulation(population,paretoFronts, npop)
    
    for _ in range(niter):
        for i in range(npop):
            population[i].velocity = w*population[i].velocity+c1*np.random.rand()*(population[i].best.position -population[i].position)+c2*np.random.rand()*(GLOBAL_BEST.position - population[i].position)
            
            #velocity limits
            population[i].velocity = np.maximum(population[i].velocity, minVelocity)
            population[i].velocity = np.minimum(population[i].velocity, maxVelocity)
            
            #update position
            population[i].position = population[i].position + population[i].velocity
        
            #position limits
            population[i].position = np.maximum(population[i].position, varMin)
            population[i].position = np.minimum(population[i].position, varMax)
            
            #apply constraint to new population
            for idx in range(npop):
                population[idx].position = applyConstraint(population[idx].position)
                population[idx].position = normalize(population[idx].position)

            #cost-evaluation
            population[i].cost1 =cost1(population[i].position)
            population[i].cost2 =cost2(population[i].position)
            
            #update personal best and global best
            if population[i].cost1 >= population[i].best.cost1 and population[i].cost2 <= population[i].best.cost2:
                
                population[i].best.position = population[i].position
                population[i].best.cost1 = population[i].cost1
                population[i].best.cost2 = population[i].cost2
                             
                
            if population[i].best.cost1 >= GLOBAL_BEST.cost1 and  population[i].best.cost2 <= GLOBAL_BEST.cost2:    
                GLOBAL_BEST.position = population[i].best.position
                GLOBAL_BEST.cost1 = population[i].best.cost1
                GLOBAL_BEST.cost2 = population[i].best.cost2
                
                             
        
        paretoFronts, population = nonDominatedSorting(population, convert2objective)
        
        #crowd-sort
        for front in paretoFronts:
            if len(paretoFronts[front])>1:
                paretoFronts[front] = crowdingDistSorting(paretoFronts[front], population)
        
        #update rank
        rank = updateRank(paretoFronts)
        
        #damping inertia coefficient
        w *= wdamp
        
        #store the current pareto fronts
        pareto_cost = {}
        for front in paretoFronts:
            pareto_cost[front] = [[population[f].cost1, population[f].cost2] for f in paretoFronts[front]]
        pareto_list.append(pareto_cost)
    return (population, pareto_list, GLOBAL_BEST)


# In[ ]:





# In[ ]:




