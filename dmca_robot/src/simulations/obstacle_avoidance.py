"""

References:
    [1] TODO
    
    [2] S. Caron, “Least squares,” scaron.info. https://scaron.info/doc/
        qpsolvers/least-squares.html#qpsolvers.solve_ls (accessed Oct. 23, 2022).

"""
import numpy as np
from numpy import pi, cos, sin, arctan2, sqrt
from numpy.linalg import norm
from qpsolvers import solve_ls
import matplotlib.pyplot as plt

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
            safety_distances,
            max_control_velocity,
            distance_between_wheels,
            wheel_radius,
            heading_weights,
            **sim_args
        ):
        """
        The initial interdistances are used as the desired interdistances.
        I.e. the robots are assumed to be initially in the desired formation.
        
        Args:
            obstacle_centres (np.ndarray: shape = n_obstacles x 2): List of
                obstacle centre positions in cartesian coordinates.
            safety_distances (np.ndarray: shape = n_obstacles): The minimum
                distance that the robot should come towards the centre of
                the object.
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
        wl_max = (v_max - 0.5*b*omega_max)/r
        wr_max = (v_max + 0.5*b*omega_max)/r
        
        # Let wl_min = -wl_max and wr_min = -wr_max
        H = np.array([wl_max, wr_max, wl_max, wr_max])
        _, desired_interdistances = self.formation_keeping(initial_positions)
        
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
        self.dst = safety_distances
        
        
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
        x_sum = np.dot(x, self.a - 1/self.n_robots)
        y_sum = np.dot(y, self.a - 1/self.n_robots)
        
        theta_c = arctan2(x_sum, y_sum)
        
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
            Jn[-1, 2*j:2*(j + 1)] = (-mxj, myj)
         
        return theta_c, Jn
    
    def obstacle_avoidance(self, pc):
        """
        Calculates variables defined in section 2D of [1].
        Args:
            positions (np.ndarray: shape = n_robots x 2):
                Current positions of all robots.
            pc (np.ndarray: shape = 2): Centroid of the robot positions.
        """
        pc = np.expand_dims(pc, axis=0)
        diff = pc - self.pot
        dot = norm(diff, axis=1)
        
        # Equation 19 in [1]: ((pc - po)^T / (n dot)) x (I2, I2, ...)
        # diff has shape Mx2, so make dot have the same
        # shape for element-wise division
        expanded_dot = np.tile(np.expand_dims(dot, 1), (1, 2))
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
        uf = solve_ls(R, s, G=self.Jlim, h=self.omega_lim, solver='cvxopt')
        
        """Solve S2 level 2 from table 1 in [1]"""
        A = R
        b = np.matmul(A, uf)
        R = np.matmul(Jn, Jx)
        s = np.array([
            self.lambda_p * (self.pd - pc),
            self.lambda_theta * (self.theta_d - theta_c)
        ])
        G = np.concatenate((
            -np.matmul(Jot, Jx),
            self.Jlim
        ), axis=0)
        h = np.concatenate((
            self.lambda_ot * (dot - self.dst),
            self.omega_lim
        ), axis = 0)
        un = solve_ls(R, s, G, h, A, b, solver='cvxopt')
        
        # Convert control velocity to position and cartesian velocity
        control_vel = un.reshape((self.n_robots, 2))
        new_heading = curr_heading + self.dynamics_time_step * control_vel[:, 1]
        new_vel = to_cartesian_velocity(new_heading, control_vel, self.L)
        new_pos = curr_pos + self.dynamics_time_step * new_vel
        
        goal_error = np.array([norm(pc - self.pd), norm(theta_c - self.theta_d)])
        return new_pos, new_vel, goal_error, df
        
    def setup_plot(self, *args):
        super().setup_plot(*args)
        
        # Add obstacles to the trajectory plot
        plt.subplot(221)
        for centre, radius in zip(self.pot, self.dst):
            obstacle = circle_points(centre, radius)
            plt.plot(obstacle[:, 0], obstacle[:, 1])
            
        plt.subplot(222)
        plt.ylabel("Interdistances", fontsize='large')
        
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
        obstacle_centres = np.array([[2, 1.5]]),
        safety_distances = 0.75,
        max_control_velocity = np.array([0.1, 1]),
        distance_between_wheels = 0.35,
        wheel_radius = 0.1,
        heading_weights = np.array([0.5, 0.5, 0, 0])
    )
    
    # Don't need to set max control velocity for run_simulation is it
    # is handled by step_simulation as a constraint in the HQP problem
    simulator.run_simulation(initial_positions)