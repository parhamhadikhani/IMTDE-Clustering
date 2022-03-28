from random import random
from random import sample
from random import uniform
import numpy as np
from Evaluation import *
from sklearn import datasets
import time
import copy
from scipy.stats import cauchy
import random 

# sum squre error
def calc_sse(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)
        dist = np.sum((data[idx] - c)**2)
        distances += dist
    return distances

def quantization_error(centroids: np.ndarray, labels: np.ndarray, data: np.ndarray) -> float:
    error = 0.0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)[0]
        dist = np.linalg.norm(data[idx] - c, axis=1).sum()
        dist /= len(idx)
        error += dist
    error /= len(centroids)
    return error

# Calculate distance between data and centroids
def _calc_distance(data: np.ndarray,centroids) -> np.ndarray:
    distances = []
    for c in centroids:
        for i in range(len(data)):
            distances.append(np.linalg.norm(data[i,:,:] - c))
    distances = list(_divide_chunks(distances, len(data))) 
    distances = np.array(distances)
    distances = np.transpose(distances)
    return distances


def _divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]

# Calculate distance between data and centroids       
def cdist_fast(XA, XB):

    XA_norm = np.sum(XA**2, axis=1)
    XB_norm = np.sum(XB**2, axis=1)
    XA_XB_T = np.dot(XA, XB.T)
    distances = XA_norm.reshape(-1,1) + XB_norm - 2*XA_XB_T
    return distances  
#Predict new data's cluster using minimum distance to centroid
def _predict(data, centroids):
    distance = cdist_fast(data,centroids)
    cluster = _assign_cluster(distance)
    return cluster

#Assign cluster to data based on minimum distance to centroids
def _assign_cluster(distance: np.ndarray):
    cluster = np.argmin(distance, axis=1)
    return cluster

class Vector:
    def __init__(
            self,
            n_cluster: int,
            data: np.ndarray):
        
        index = np.random.choice(list(range(len(data))), n_cluster)
        self.centroids = data[index].copy()
        self.fitness = calc_sse(self.centroids, _predict(data,self.centroids), data)
        self.Qe=quantization_error(self.centroids, _predict(data,self.centroids), data)
        
class LifeTime_archive:
    def __init__(self,vectors,size):
        self.size=size
        self.vectors=vectors
        self.lifetime=[x for x in self.vectors]
        
    def update(self,vector):
        self.lifetime.append(vector)
        while len(self.lifetime)>self.size:
            self.lifetime.pop(0)

    def select(self):
        select=np.random.randint(0,self.size)
        return self.lifetime[select].centroids
      
class IMTDE_Clustering:

    def __init__(
            self,
            n_cluster: int,
            n_vectors: int,
            data: np.ndarray,
            max_iter: int = 500,
            mutate: float=0.5,
            recombination: float = 0.7,
            print_debug: int = 10):
        
        self.n_vectors=n_vectors
        self.n_cluster = n_cluster
        self.data = data
        self.max_iter = max_iter
        self.vectors = []
        self.lifetime=[]
        self.data=data
        self.label=None
        self.gbest_sse = np.inf
        self.quantization_error=np.inf
        self.gbest_centroids = None
        self._init_vectors()
        self.mutate=mutate
        self.recombination=recombination
        self.print_debug = print_debug
        self.winTVPIdx = 0
        self.nFES =np.zeros((3,1))
        self.WinIter = 20
        self.nImpFit=np.ones((3,1))
        self.ImpRate=np.zeros((3,1))
        self.sigma=1
        
    def _init_vectors(self):
        for i in range(self.n_vectors):

            vector = Vector(self.n_cluster, self.data)

            if vector.fitness < self.gbest_sse:
                self.gbest_centroids = vector.centroids.copy()
                self.gbest_sse = vector.fitness
                self.quantization_error=vector.Qe
                self.label=_predict(self.data,vector.centroids).copy()
            self.vectors.append(vector)
    
    
    
    def MUTATION(self,current):
        candidates = list(range(0,self.n_vectors))
        candidates.remove(current)
        random_index = sample(candidates, 3)
        v_donor=self.vectors[random_index[0]].centroids + self.mutate * (self.vectors[random_index[1]].centroids-self.vectors[random_index[2]].centroids)
        return v_donor
        
    def RECOMBINATION(self,v_donor,vector):
        v_trial=np.copy(vector.centroids)
        for k in range(len(v_trial)):
            crossover = random()
            if crossover <= self.recombination:
                v_trial[k]=v_donor[k]
        return v_trial
    
    def GREEDY_SELECTION(self,v_trial,vector,current):
        score_trial=calc_sse(v_trial, _predict(self.data,v_trial), self.data)

        if score_trial < vector.fitness:
            self.vectors[current].centroids = v_trial
            self.vectors[current].fitness=score_trial
    
    
    def distributing_population(self,i):
        if i!=0:
            for w in range(3):
                self.ImpRate[w]=self.nImpFit[w]/self.nFES[w];
            self.winTVPIdx=np.argmax( self.ImpRate)
            self.nImpFit=np.ones((3,1))
            self.ImpRate=np.zeros((3,1))
            self.nFES =np.zeros((3,1))
        
        if self.winTVPIdx == 0:#R_tvp
            G_tvp= self.vectors[0:int(self.n_vectors*.2)]
            L_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
            R_tvp = self.vectors[int(2*0.2*self.n_vectors):]
            
        if self.winTVPIdx == 1:
            G_tvp= self.vectors[0:int(self.n_vectors*.2)]
            R_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
            L_tvp = self.vectors[int(2*0.2*self.n_vectors):]
        
        if self.winTVPIdx == 2:
            R_tvp= self.vectors[0:int(self.n_vectors*.2*2)]
            L_tvp = self.vectors[int(2*0.2*self.n_vectors):int(4*0.2*self.n_vectors)]
            G_tvp = self.vectors[int(4*0.2*self.n_vectors):]
            
        self.nFES = self.nFES + [[len(R_tvp)],[len(L_tvp)],[len(G_tvp)]]
        return R_tvp,L_tvp,G_tvp
    def Population_updating(self,G_tvp,R_tvp,L_tvp):
        if self.winTVPIdx == 0:
            
            self.vectors[0:int(self.n_vectors*.2)]=G_tvp
                
            
            self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]=L_tvp
                
                
            self.vectors[int(2*0.2*self.n_vectors):]=R_tvp
                
            
        if self.winTVPIdx == 1:
            
            self.vectors[0:int(self.n_vectors*.2)]=G_tvp
                
            
            self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]=R_tvp
                
                
            self.vectors[int(2*0.2*self.n_vectors):]=L_tvp
        
        if self.winTVPIdx == 2:
            
            self.vectors[0:int(self.n_vectors*.2*2)]=R_tvp
                
            
            self.vectors[int(2*0.2*self.n_vectors):int(4*0.2*self.n_vectors)]=L_tvp
                
                
            self.vectors[int(4*0.2*self.n_vectors):]=G_tvp
            
            
    def R_TVP(self,r_tvp,ca1):
        index=[]
        #f=F[0:len(r_tvp)]
        M_tril=np.tril(np.ones((self.n_cluster,len(self.data[0,:]))))

        for vector in r_tvp:
            index.append(vector.fitness)
        best=r_tvp[np.argmin(index)]
        worst=r_tvp[np.argmax(index)]
        for i, vector in enumerate(r_tvp):
            M_tril=np.random.permutation(M_tril)
            M_bar=1-M_tril
            vi=vector.centroids +(best.centroids-vector.centroids)+(worst.centroids-vector.centroids)+ca1*(self.lifetime.select()-vector.centroids)
            #vi=vector.centroids +((best.centroids-vector.centroids)+(worst.centroids-vector.centroids))*np.exp(np.random.normal(0, self.sigma))+ca1*(self.lifetime.select()-vector.centroids)
            #vi=np.exp(np.random.normal(vector.centroids, abs((best.centroids-vector.centroids)+(worst.centroids-vector.centroids))))+ca1*(self.lifetime.select()-vector.centroids)
            ui=(M_tril*vector.centroids + M_bar* vi)*np.exp(np.random.normal(0, self.sigma))
            score_trial=calc_sse(ui, _predict(self.data,ui), self.data)

            if score_trial < vector.fitness:
                self.lifetime.update(vector)
                r_tvp[i].centroids = ui
                r_tvp[i].fitness=score_trial
                self.nImpFit[0]+=1
        return r_tvp
    #random.choices(vectors, k=1)[0]
    def L_TVP(self,l_tvp,ca2):
        
        #f=F[0:len(l_tvp)]
        for i, vector in enumerate(l_tvp):
            #cauchy.cdf(vector.centroids)*
            ui=(vector.centroids + (random.choices(l_tvp, k=1)[0].centroids-random.choices(l_tvp, k=1)[0].centroids)+ca2*(self.lifetime.select()-vector.centroids))
            score_trial=calc_sse(ui, _predict(self.data,ui), self.data)

            if score_trial < vector.fitness:
                self.lifetime.update(vector)
                l_tvp[i].centroids = ui
                l_tvp[i].fitness=score_trial
                self.nImpFit[1]+=1
        return l_tvp
    
    def G_TVP(self,g_tvp,ca2):
        M_tril=np.tril(np.ones((self.n_cluster,len(self.data[0,:]))))

        for i, vector in enumerate(g_tvp):
            M_tril=np.random.permutation(M_tril)
            M_bar=1-M_tril
            vi = (self.gbest_centroids + ca2*(random.choices(g_tvp, k=1)[0].centroids-random.choices(g_tvp, k=1)[0].centroids))
            ui=(M_tril*vector.centroids + M_bar* vi)*np.exp(np.random.normal(0, self.sigma))
            score_trial=calc_sse(ui, _predict(self.data,ui), self.data)

            if score_trial < vector.fitness:
                self.lifetime.update(vector)
                g_tvp[i].centroids = ui
                g_tvp[i].fitness=score_trial
                self.nImpFit[2]+=1
        return g_tvp
    
    
    def fit(self):
        
        self.lifetime=LifeTime_archive(self.vectors,self.n_vectors)
        memory_sf = 0.5 * np.ones((len(self.data[0,:]), 1))
        copy=np.mod(self.n_vectors,len(self.data[0,:]))
        history = []
        count=1
        MaxFES = len(self.data[0,:]) * 10000
        MaxGen = MaxFES/self.n_vectors;
        Gen=0
        Mu = np.log(len(self.data[0,:]))
        initial = 0.001
        final = 2
        #The winner-based distributing substep
       
            
        for i in range(self.max_iter):
        # #DE
        #     for vector in self.vectors:
        #         v_donor=self.MUTATION(i)
        #         v_trial=self.RECOMBINATION(v_donor,vector)
        #         self.GREEDY_SELECTION(v_trial,vector,i)
        #     for vector in self.vectors:
        #         if vector.fitness < self.gbest_sse:
        #             self.gbest_centroids = vector.centroids.copy()
        #             self.gbest_sse = vector.fitness
        #             self.label=_predict(self.data,vector.centroids).copy()
        #     history.append(self.gbest_sse)
        # return history,self.label,self.gbest_sse
        # #/DE
            
            
            
            
            
            
        
        ##MTDE
            Gen = Gen +1
            ca1 = 2 - Gen * ((2) /MaxGen)                                        
            ca2 = (initial - (initial - final) * (((MaxGen - Gen)/MaxGen))**Mu)

            #===========================The winner-based distributing substep================
            if i!=0:
                for w in range(3):
                    if w==0:
                        self.ImpRate[w]=(self.nImpFit[w]*1/i)/self.nFES[w]
                    else:
                        self.ImpRate[w]=(self.nImpFit[w]*(1/(self.max_iter-i)))/self.nFES[w]
                self.winTVPIdx=np.argmax( self.ImpRate)
                self.winTVPIdx=0
                #print('-----wintvp-----:',self.winTVPIdx)
                self.nImpFit=np.ones((3,1))
                self.ImpRate=np.zeros((3,1))
                self.nFES =np.zeros((3,1))
            self.winTVPIdx=0
            if self.winTVPIdx == 0:#R_tvp
                # G_tvp= self.vectors[0:int(self.n_vectors*.2)]
                # L_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
                # R_tvp = self.vectors[int(2*0.2*self.n_vectors):]
                
                
                # R_tvp= self.vectors[0:int(self.n_vectors*.2)]
                # G_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
                # L_tvp = self.vectors[int(2*0.2*self.n_vectors):]
                
                
                L_tvp= self.vectors[0:int(self.n_vectors*.2)]
                G_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
                R_tvp = self.vectors[int(2*0.2*self.n_vectors):]
                
                
                
            if self.winTVPIdx == 1:
                G_tvp= self.vectors[0:int(self.n_vectors*.2)]
                R_tvp = self.vectors[int(0.2*self.n_vectors):int(2*0.2*self.n_vectors)]
                L_tvp = self.vectors[int(2*0.2*self.n_vectors):]
            
            if self.winTVPIdx == 2:
                R_tvp= self.vectors[0:int(self.n_vectors*.2*2)]
                L_tvp = self.vectors[int(2*0.2*self.n_vectors):int(4*0.2*self.n_vectors)]
                G_tvp = self.vectors[int(4*0.2*self.n_vectors):]
                
            self.nFES = self.nFES + [[len(R_tvp)],[len(L_tvp)],[len(G_tvp)]]
            
            
            #r_tvp,l_tvp,g_tvp=self.distributing_population(i)
            
            #===========================R-TVP=====================================
            r_tvp=self.R_TVP(R_tvp,ca1)
            #===========================L-TVP=====================================
            l_tvp=self.L_TVP(L_tvp,ca2)
            
            #===========================G-TVP=====================================
            g_tvp=self.G_TVP(G_tvp,ca2)
            #print('G_TVP',len(g_tvp),'R_TVP',len(r_tvp),'L_TVP',len(l_tvp),'***')
            #======================= Population updating =========================================
            self.Population_updating(g_tvp,r_tvp,l_tvp)
            
            for vector in self.vectors:
                if vector.fitness < self.gbest_sse:
                    self.gbest_centroids = vector.centroids.copy()
                    self.gbest_sse = vector.fitness
                    self.quantization_error=vector.Qe
                    self.label=_predict(self.data,vector.centroids).copy()
            history.append(self.gbest_sse)
            if i % self.print_debug == 0:
                print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(
                    i + 1, self.max_iter, self.gbest_sse))
            self.sigma= self.sigma-(1/(self.max_iter)) 
        print('Finish with gbest score {:.18f}'.format(self.gbest_sse))
        return history,self.label,self.gbest_sse,self.quantization_error
        ##/MTDE
 
    
 
    
 
    
         



