
    # def update_agent_params(self, t, n, label, w):
    #     """
    #         Update the parameters of this agent.
    #         t: continuous time, t.
    #         n: number of agents within a circle of raduius r 
    #             about the current position of this agent.
    #         label: label (as a natural number) of this agent.

    #         w: heading of this agent averaged over that of 
    #             its neighbors.

    #     """
    #     self.n     = n 
    #     self.w     = w     # averaged over the neighors that surround this agent
    #     self.label = label # update the label of this agent if it has not changed


        # Janvier 03: Use one grid for all agents
        # if grids is None and num_agents==1:
        #     # for every agent, create the grid bounds
        #     grid_mins = [[-1, -1, -np.pi]]
        #     grid_maxs = [[1, 1, np.pi]]   
        #     grids = flockGrid(grid_mins, grid_maxs, dx=.1, num_agents=num_agents, N=grid_nodes)
        # elif grids is None and num_agents>1:
        #     gmin = np.array(([[-1, -1, -np.pi]]),dtype=np.float64).T
        #     gmax = np.array(([[1, 1, np.pi]]),dtype=np.float64).T
        #     grid = createGrid(gmin, gmax, grid_nodes, 2)

        # birds could be on different subspaces of an overall grid
        # if isinstance(grids, list):
        #     self.vehicles = []
        #     lab = 0
        #     for each_grid in grids:
        #         self.vehicles.append(BirdSingle(each_grid,1,1,None, \
        #                                  random.random(), label=lab))
        #         lab += 1
        # else: # all birds are on the same grid
        #     ref_bird = BirdSingle(grids[0], 1, 1, None, \
        #                         random.random(), label=0) 
        #     self.vehicles = [BirdSingle(grids[i], 1, 1, None, \
        #             random.random(), label=i) for i in range(1,num_agents)]

        ## Target set

                
    def get_target(self, reach_rad=1.0, avoid_rad=1.0):
        """
        TODO: Move this to main function.

            Make reference bird the evader and every other bird the pursuer
            owing to the lateral visual anisotropic characteric of starlings.
        """
        # first bird is the evader, so collect its position info
        cur_agent = 0
        evader = self.vehicles[cur_agent]
        target_set = np.zeros((self.N-1,)+(evader.grid.shape), dtype=np.float64)
        payoff_capture = np.zeros((evader.grid.shape), dtype=np.float64)
        # first compute the any pursuer captures an evader
        for pursuer in self.vehicles[1:]: 
            if not np.any(pursuer.center):
                pursuer.center = np.zeros((pursuer.grid.dim, 1))
            elif(numel(pursuer.center) == 1):
                pursuer.center = pursuer.center * np.ones((pursuer.grid.dim, 1), dtype=np.float64)

            #---------------------------------------------------------------------------
            #axis_align must be same for all agents in a flock
            # any pursuer can capture the reference bird
            for i in range(pursuer.grid.dim):
                if(i != pursuer.axis_align):
                    target_set[cur_agent] += (pursuer.grid.xs[i] - evader.grid.xs[i])**2
            target_set[cur_agent] = np.sqrt(target_set[cur_agent])

            # take an element wise min of all corresponding targets now
            if cur_agent >= 1:
                payoff_capture = np.minimum(target_set[cur_agent], target_set[cur_agent-1], dtype=np.float64)
            cur_agent += 1
        payoff_capture -= reach_rad

        # compute the anisotropic value function: this maintains the gap between the pursuers
        # note this is also the avoid set
        target_set = np.zeros((self.N-1,)+(evader.grid.shape), dtype=np.float64)
        payoff_avoid = np.zeros((evader.grid.shape), dtype=np.float64)
        cur_agent = 0
        for vehicle_idx in range(1, len(self.vehicles)-1):
            this_vehicle = self.vehicles[vehicle_idx]
            next_vehicle = self.vehicles[vehicle_idx+1]
            for i in range(this_vehicle.grid.dim):
                if(i != this_vehicle.axis_align):
                    target_set[cur_agent] += (this_vehicle.grid.xs[i] + next_vehicle.grid.xs[i])**2
            target_set[cur_agent] = np.sqrt(target_set[cur_agent])

            # take an element wise min of all corresponding targets now
            if cur_agent >= 1:
                payoff_avoid = np.minimum(target_set[cur_agent], target_set[cur_agent-1], dtype=np.float64)
            cur_agent += 1
        
        payoff_avoid -= avoid_rad

        # now do a union of both the avoid and capture sets
        combo_payoff = np.minimum(payoff_avoid, payoff_capture)

        return combo_payoff



    def do_runge_kutta4(self, cur_state, t_span, M=4, h=2):
        """
            .cur_state: state at time space
            .t_span
            .M: RK steps per interval
            .h: time step
        """
        # integrate the dynamics with 4th order RK scheme
        X = np.array(cur_state) if isinstance(cur_state, list) else cur_state

        for j in range(M):
            if np.any(t_span): # integrate for this much time steps
                hh = (t_span[1]-t_span[0])/10/M
                for h in np.arange(t_span[0], t_span[1], hh):
                    k1 = self.dynamics(X)
                    k2 = self.dynamics(X + h/2 * k1)
                    k3 = self.dynamics(X + h/2 * k2)
                    k4 = self.dynamics(X + h * k3)

                    X  = X+(h/6)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                k1 = self.dynamics(X)
                k2 = self.dynamics(X + h/2 * k1)
                k3 = self.dynamics(X + h/2 * k2)
                k4 = self.dynamics(X + h * k3)

                X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)

        return X


    def dynamics(self, cur_state, disturb=np.zeros((3,1))):
        """
            Dubins Vehicle Dynamics in absolute coordinates.
            Please consult Merz, 1972 for a detailed reference.

            \dot{x}_1 = v cos x_3
            \dot{x}_2 = v sin x_3
            \dot{x}_3 = w
        """
        xdot = [
                self.v_e * np.cos(cur_state[2]) + disturb[0],
                self.v_e * np.sin(cur_state[2]) + disturb[1],
                (cp.ones_like(cur_state[2])/(1+self.valence))*(self.w_e + cp.sum([x.w_e for x in self.neighbors]))+ disturb[2],
        ]

        return cp.asarray(xdot)