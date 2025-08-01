import numpy as np

class TwoEchelonInv:
    """
    the setup of two-echelon inventory model.
    """
    def __init__(self, 
                 h1 = 1.0,     # marginal holding cost for installation 1.
                 p1 = 10.0,    # marginal shortage cost for installation 1.
                 h2 = 1.0,     # marginal holding cost for installation 2.
                 demand_lambda = 5,     # suppose the demand follows possion distribution, \lambda = 5.
                 
                 # initial state variables.    
                 init_x1 = 20.0,     # initial stock on hand for installation 1.
                 init_w1 = 0.0,      # initial stock in transit.
                 init_x2 = 40.0,     # initial system stock. 
                 init_w2 = 0.0,
                 K = 0.0,            # purchase setup one time cost.
                 c = 1.0,            # purchase per unit cost.
                 c1 = 1.0,           # transit cost per unit within installations.
                 seed = None         # add stochasticity.  
                 ):

        self.h1 = h1 
        self.p1 = p1
        self.h2 = h2 
        self.demand_lambda = demand_lambda
        self.init_x1 = init_x1
        self.init_w1 = init_w1
        self.init_x2 = init_x2
        self.init_w2 = init_w2
        self.K = K 
        self.c = c
        self.c1 = c1
        
        if seed is not None:
            np.random.seed(seed)
        self.reset()

        # current state variables.
        self.x1 = init_x1 
        self.w1 = init_w1
        self.x2 = init_x2
        self.w2 = init_w2
        self.t = 0     # tracking the current time step. 

    def reset(self) -> np.ndarray:
        """
        reset the model to initial state, to be used in monte carlo simulation.

        Return: a 3 dimension array containing state variables(np.ndarray)  
        """
        self.x1 = self.init_x1
        self.w1 = self.init_w1
        self.x2 = self.init_x2
        self.w2 = self.init_w2
        self.t = 0

        return np.array([self.x1, self.w1, self.x2, self.w2], dtype=np.float32)

    
    def purchase_cost(self, z: float) -> float:
        """
        compute the purchase cost

        Args:
        z (float): purchase quantity.

        Return:
        float: if z > 0, return setup cost + per unit cost, else no cost.
        """
        return self.K + self.c * z if z > 0 else 0.0
    
    def shortage_storage_cost(self, x: float) -> float:
        """
        compute the shortage and storage cost.

        Args:
        x (float): current order level at installation 1 after the demand is taken (can be positive and negitive).

        Return:
        float: the shortage and storage cost.
        """
        return self.h1 * x if x >= 0 else self.p1 * (-x)

    def step(self, a1: float, a2: float) -> Tuple[np.ndarray, float, bool]:
        """
        calculate the total cost of current time step, and update the new state variables.

        Args:
        a1: order request of installation 1 to be delivered at the beginning of next period.
        a2: order request of installation 2 to be delivered at the beginning of next period.

        Return:
        new states variables and the current step cost.
        """
        #demand = np.random.uniform(low=demand_lambda-1, high=demand_lambda+1)
        #demand = np.random.poisson(demand_lambda)
        demand = np.random.normal(loc=demand_lambda, scale=np.sqrt(1))
            
        #demand = demand_lambda
        # step 1: compute the stock level at installation 2 after receiving the products.
        x2_local = self.x2 - self.w1 - self.x1 + self.w2
        
        # step 2: compute the stock level at installation 1 after demand request.
        self.x1 = self.x1 + self.w1 - demand
        self.w1 = min(a1, max(x2_local, 0))
        cost_w1 = self.c1 * self.w1
        cost_x1 = self.shortage_storage_cost(self.x1)
        
        # step 3: compute the stock level at installation 2 after demand request.
        self.x2 = self.x2 + self.w2 - demand
        x2_local = x2_local - self.w1
        self.w2 = a2
        cost_w2 = self.c1 * self.w2
        cost_x2 = max(self.h2 * x2_local, 0)
        
        total_period_cost = cost_x1 + cost_x2 + cost_w1 + cost_w2 + (a1 - self.w1) * 3.0

        next_state = np.array([self.x1, self.w1, self.x2, self.w2], dtype=np.float32)         

        return (next_state, total_period_cost)
