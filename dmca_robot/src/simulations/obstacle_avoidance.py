"""

References:
    [1] TODO
    
    [2] S. Caron, “Least squares,” scaron.info. https://scaron.info/doc/
        qpsolvers/least-squares.html#qpsolvers.solve_ls (accessed Oct. 23, 2022).

"""
import numpy as np
from numpy import pi, cos, sin, arctan2, sqrt
from numpy.linalg import norm
from qpsolvers import solve_ls, solve_qp # pip install qpsolvers[all_pypi_solvers]
import matplotlib.pyplot as plt
from time import perf_counter

from helpers.circle_simulation_helpers import circle_points
from helpers.lab_helpers import to_cartesian_velocity, rotation_matrix
from simulations.formation_and_goal_seeking import MultiRobotGoalSeeker

class ObstacleAvoider(MultiRobotGoalSeeker):
    def __init__(self,
            n_robots,
            initial_positions,
            goal_centre,
            goal_heading,
            obstacle_centres,
            obstacle_radii,
            max_control_velocity,
            distance_between_wheels,
            wheel_radius,
            heading_weights,
            safety_distance = None,
            **sim_args
        ):
        """
        The initial interdistances are used as the desired interdistances.
        I.e. the robots are assumed to be initially in the desired formation.
        
        Args:
            obstacle_centres (np.ndarray: shape = n_obstacles x 2): List of
                obstacle centre positions in cartesian coordinates.
            obstacle_radii (np.ndarray: shape = n_obstacles): List of radii
                describing the max distance from the centre of an obstacle to
                the edge of that obstacle, for each obstacle.
            safety_distance (Optional[float]): The minimum
                distance that the centroid of the formation should come within
                the edge of an obstacle. Default: Max radius of the desired formation.
            heading_weights (np.ndarray: shape = n_robots): Array of
                a_i values in [1] which sum to 1 and are used to denote
                the front/heading of the robot formation.
        """
        # Calculate the goal positions by translating the centroid of the initial
        # positions to the origin, rotating all the points, then translating
        # to the goal centre.
        goal_positions = initial_positions - np.mean(initial_positions, axis=0)
        goal_positions = np.matmul(rotation_matrix(goal_heading), goal_positions.T).T
        goal_positions += goal_centre
        super().__init__(n_robots, goal_positions, **sim_args)
        
        # Calculate the number of distinct unordered pairs of robots
        # n choose 2 = 1 + 2 + ... + n - 1 = n(n - 1)/2
        n_pairs = int(self.n_robots * (self.n_robots - 1) / 2)
        
        # E.g. for 4 robots, generate pairs (1, 2), (1, 3),
        # (1, 4), (2, 3), (2, 4), (3, 4)
        self.pairs = np.zeros((n_pairs, 2), dtype=int)
        curr_pair = 0
        for i in range(self.n_robots):
            for j in range(i + 1, self.n_robots):
                self.pairs[curr_pair] = (i, j)
                curr_pair += 1
                
        """Precompute constants defined in section 2 of [1]"""
        # Calculate constants
        r = wheel_radius
        b = distance_between_wheels
        G = np.array([
            [ 1, -0.5*b],
            [ 1,  0.5*b],
            [-1,  0.5*b],
            [-1, -0.5*b]
        ]) / r
        v_max, omega_max = max_control_velocity
        v_min = -v_max
        omega_min = -omega_max
        wl_max = (v_max - 0.5*b*omega_min)/r
        wr_max = (v_max + 0.5*b*omega_max)/r
        wl_min = (v_min - 0.5*b*omega_max)/r
        wr_min = (v_min + 0.5*b*omega_min)/r
        H = np.array([wl_max, wr_max, -wl_min, -wr_min])
        _, desired_interdistances = self.formation_keeping(initial_positions)
        
        # Use the radius of the formation as the safety distance
        if safety_distance is None:
            safety_distance = max(norm(goal_positions - goal_centre, axis=1))
            
        # Assign constants
        # TODO: Tune control gains (lambda)
        self.a = heading_weights
        self.lambda_f = 1
        self.df_d = desired_interdistances
        self.Jlim = np.kron(np.eye(self.n_robots), G)
        self.omega_lim = np.tile(H, self.n_robots)
        self.lambda_p = 1
        self.lambda_theta = 1
        self.lambda_ot = 1
        self.pd = goal_centre
        self.theta_d = goal_heading
        self.pot = obstacle_centres
        # Safe distance from the centre of an obstacle
        self.dst = safety_distance + obstacle_radii
        self.obstacle_radii = obstacle_radii
        
        self.problem1_warm_start = np.zeros((8,))
        self.problem2_warm_start = np.zeros((8,))
        
        
    def B(self, theta):
        """Computes B(theta_i) from equation (5) in [1]"""
        return np.array((
            (cos(theta), -self.L * sin(theta)),
            (sin(theta),  self.L * cos(theta))
        ))
    
    def interdistance(self, i, j, positions):
        """
        Calculates the interdistance dij and interdistance
        unit vector nij between robots i and j.
        Returns:
            (dij, nij)
        """
        difference = positions[j] - positions[i]
        dij = norm(difference)
        return dij, difference.T / dij
    
    def formation_keeping(self, positions):
        """
        Calculate Jf from equation (9) in section 2B of [1]
        as well as the interdistances/formation errors df.
        Returns:
            Jf: Formation Jacobian matrix
            df: Vector of interdistances
        """
        n_pairs = self.pairs.shape[0]
        df = np.zeros(n_pairs)
        Jf = np.zeros((n_pairs, 2*self.n_robots))
        
        # Each row of the Jacobian Jf depends on one (i, j) pair
        for row_id, (i, j) in enumerate(self.pairs):
            dij, nij = self.interdistance(i, j, positions)
            df[row_id] = dij
            Jf[row_id, 2*i:2*(i + 1)] = -nij
            Jf[row_id, 2*j:2*(j + 1)] = nij
        return Jf, df
    
    def goal_seeking(self, positions):
        """
        Calculates variables defined in section 2C of [1].
        
        Returns:
            theta_c: Heading of the formation
            Jn (np.ndarray: shape = 3, 2*self.n_robots): Jacobian
                matrix for navigating the formation to the goal.
        """
        x = positions[:, 0]
        y = positions[:, 1]
        # Sums in the denominator and numerator, respectively, in
        # equation (13) in [1].
        x_sum = np.dot(x, self.a - 1/self.n_robots)
        y_sum = np.dot(y, self.a - 1/self.n_robots)
        
        theta_c = arctan2(y_sum, x_sum)
        
        # Calculate the first two rows of Jn in equation (14)
        # in [1] and set the third row to zeros
        top_part = np.tile(np.eye(2)/self.n_robots, [1, self.n_robots])
        Jn = np.concatenate((top_part, np.zeros((1, 2*self.n_robots))), axis=0)
        
        # Calculate [mxj, myj]
        m = lambda j: np.array([x_sum, y_sum]) * \
            (self.a[j] - 1/self.n_robots) / (x_sum**2 + y_sum**2)
        
        # Fill in the values for the third row of Jn
        for j in range(self.n_robots):
            mxj, myj = m(j)
            Jn[-1, 2*j:2*(j + 1)] = (-myj, mxj)
         
        return theta_c, Jn
    
    def obstacle_avoidance(self, pc):
        """
        Calculates variables defined in section 2D of [1].
        Args:
            pc (np.ndarray: shape = 2): Centroid of the robot positions.
        Returns:
            dot (np.ndarray: shape = n_obstacles x 2): For simplicity, each
                obstacle is considered to be bounded by a circle. So, dot and
                dst are the actual and desired minimum distances of the
                the centre of the formation to the centre of each obstacle rather
                than the edge of the obstacle use in equation (17) in [1].
            Jot (np.ndarray: shape = n_obstacles x (2*n_robots))
        """
        pc = np.expand_dims(pc, axis=0)
        # Equation (17) in [1]:
        diff = pc - self.pot
        dot = norm(diff, axis=1)
        
        # Equation 19 in [1]: ((pc - po)^T / (n dot)) x (I2, I2, ...)
        # diff has shape n_obstacles x 2, so make dot have the same
        # shape for element-wise division
        expanded_dot = np.tile(np.expand_dims(dot, 1), (1, 2))
        # Calculate the (pc - po)^T / 4dot part of equation (19) in [1]
        left_term = diff / (self.n_robots * expanded_dot)
        Jot = np.matmul(left_term, np.tile(np.eye(2), (1, self.n_robots)))
        return dot, Jot
    
    def step_simulation(self, curr_pos, _, curr_heading):
        """
        Based on table 1 in [1] using the same notation,
        except df* (desired inter-distance) is named df_d
        Jx, Jf, df, Jn, pc, theta_c, Jot and dot are from equations
        6, 9, 9, 15, 11, 13, 19 and 17 in section 2 of [1], respectively.
        Args:
            curr_pos (np.ndarray: shape = n_robots x 2)
            curr_vel (np.ndarray: shape = n_robots x 2)
        Returns:
            (next_pos, next_vel, goal_error, formation_error)
        """
        """Calculate variables defined in section 2 of [1]"""
        Jx = np.zeros((2*self.n_robots, 2*self.n_robots))
        for i in range(self.n_robots):
            Jx[2*i:2*(i + 1), 2*i:2*(i + 1)] = self.B(curr_heading[i])
        
        Jf, df = self.formation_keeping(curr_pos)
        theta_c, Jn = self.goal_seeking(curr_pos)
        pc = np.mean(curr_pos, axis=0)
        dot, Jot = self.obstacle_avoidance(pc)
        
        """Solve S2 level 1 from table 1 in [1]"""
        # The solver minimises ||Ru - s||^2 subject to
        # Gu <= h and
        # Au = b [2]
        R = np.matmul(Jf, Jx)
        s = self.lambda_f * (self.df_d - df)
        uf = solve_ls(R, s, G=self.Jlim, h=self.omega_lim,
            solver='osqp', initvals=self.problem1_warm_start)
        self.problem1_warm_start = uf
        
        """Solve S2 level 2 from table 1 in [1]"""
        rank = np.linalg.matrix_rank(R)
        # Remove redundant rows (i.e. no redundant equality constraints) to
        # make A full-rank so that the solver doesn't have errors. 
        A = R[:rank]
        b = np.matmul(A, uf)[:rank]
        
        R = np.matmul(Jn, Jx)
        s = np.append(
            self.lambda_p * (self.pd - pc),
            self.lambda_theta * (self.theta_d - theta_c)
        )
        G = np.concatenate((
            -np.matmul(Jot, Jx),
            self.Jlim
        ), axis=0)
        h = np.concatenate((
            self.lambda_ot * (dot - self.dst),
            self.omega_lim
        ), axis = 0)
        # G = self.Jlim
        # h = self.omega_lim
        P = np.matmul(R.T, R)
        #print("Rank = {}".format(np.linalg.matrix_rank(np.concatenate((P, A.T, G.T), axis=1))))
        un = solve_ls(R, s, G=G, h=h, A=A, b=b,
            solver='osqp', initvals=self.problem2_warm_start)
        self.problem2_warm_start = un
        # Convert control velocity to position and cartesian velocity
        control_vel = un.reshape((self.n_robots, 2))
        new_heading = curr_heading + self.dynamics_time_step * control_vel[:, 1]
        new_vel = to_cartesian_velocity(new_heading, control_vel, self.L)
        new_pos = curr_pos + self.dynamics_time_step * new_vel
        
        goal_error = np.array([norm(pc - self.pd), norm(theta_c - self.theta_d)])
        # Use error in millimetres
        interdistance_error = 1e3 * np.abs(self.df_d - df)
        return new_pos, new_vel, goal_error, interdistance_error
        
    def setup_plot(self, *args):
        """
        Start with the MultiRobotGoalSeeker plot and make adjustments
        specific for this simulation as necessary.
        """
        times, positions = args[:2]
        super().setup_plot(legend=False, *args)
        plt.suptitle("Avoiding Obstacles Whilst Maintaining Formation")
        
        # Add obstacles to the trajectory plot
        plt.subplot(221)
        for centre, obstacle_radius, safety_radius in zip(self.pot, self.obstacle_radii, self.dst):
            obstacle = circle_points(centre, obstacle_radius)
            safety_boundary = circle_points(centre, safety_radius)
            plt.plot(obstacle[:, 0], obstacle[:, 1], 'g', lw=0.5)
            plt.plot(safety_boundary[:, 0], safety_boundary[:, 1], 'k:', lw=0.3)
        
        # Plot the trajectory of the centroid of the robots
        centroid = np.mean(positions, axis=1)
        plt.plot(centroid[:, 0], centroid[:, 1], 'k--', lw=0.5)
        
        handles, _ = plt.get_legend_handle_lables()
        trajectory_legend = [r"$\mathbf{{x}}_{}$".format(robot + 1) for robot in range(self.n_robots)]
        trajectory_legend.extend([r"\mathbf{{x}}_c", "start", "end", "goal", "obstacle", "safety margin"])
        # Trajectory handles
        filtered_handles = [handles[5*i] for i in range(self.n_robots)]
        # Centroid handle
        filtered_handles.append(handles[-1])
        # Start, end and goal handles
        filtered_handles.extend(handles[2:6])
        # Obstacle handles
        plt.legend(trajectory_legend[-3:-1])
        
        plt.subplot(222)
        plt.title("Interdistance Errors")
        plt.ylabel("Distance (mm)")
        plt.legend([r"$d_{{{:d}{:d}}}$".format(i + 1, j + 1) for i, j in self.pairs])
        time_lims = [0, self.simulation_time]
        plt.xlim(time_lims)
        
        plt.subplot(223)
        plt.xlim(time_lims)
        
        plt.subplot(224)
        plt.xlim(time_lims)
        
        plt.tight_layout()
        
if __name__ == "__main__":

    initial_positions = np.array([
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1]
    ])
    simulator = ObstacleAvoider(
        n_robots = 4,
        initial_positions = initial_positions,
        goal_centre = np.array([4, 4]),
        goal_heading = pi/2,
        obstacle_centres = np.array([[2, 1.5], [1, 5]]),
        safety_distance = 0.1 + 1/sqrt(2),
        obstacle_radii = np.array([0.5, 0.7]),
        max_control_velocity = np.array([0.1, 1]),
        distance_between_wheels = 0.35,
        wheel_radius = 0.1,
        heading_weights = np.array([0.5, 0.5, 0, 0]),
        simulation_time = 30,
        dynamics_time_step = 0.1
    )
    
    start_time = perf_counter()
    # Don't need to set max control velocity for run_simulation is it
    # is handled by step_simulation as a constraint in the HQP problem
    times, _, _, positions, velocities, goal_errors, formation_errors = \
        simulator.run_simulation(initial_positions)
    
    duration = perf_counter() - start_time
    print("Took {:.5f}s for {} time steps ({:.5f} s/s)".format(duration, len(times), duration / simulator.simulation_time))
    
    # Set up data for animation
    simulator.setup_plot(times, positions, velocities, goal_errors, formation_errors)
    
    # if avg_lines:
    #     lines.extend(avg_lines)
        
    plt.show()