"""
Simulation from thesis A refactored to be compatible with the 
pioneer robots and python 2.7.

Recreate the results of the nonholonomic robot simulation in [1],
except changing the formation keeping part of the loss function
to something that is both (approximately) translation and rotation
invariant. This is because the formation keeping part in [1] is
only translation invariant and hence causes the formation to be
completely reorganised when only rotation is required.

Author: Mark Amos
Date: 4/03/2022
References:
    [1] W. Ren and E. Atkins, "Distributed multi-vehicle coordinated 
        control via local information exchange", International Journal 
        of Robust and Nonlinear Control, vol. 17, no. 10-11, pp. 
        1002-1033, 2007. Available: 10.1002/rnc.1147 [Accessed 4 March 2022].
    [2] R. Olfati-Saber and R. Murray, "Consensus Problems in Networks 
        of Agents With Switching Topology and Time-Delays", IEEE 
        Transactions on Automatic Control, vol. 49, no. 9, pp. 1520-1533, 
        2004. Available: 10.1109/tac.2004.834113 [Accessed 8 March 2022].
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from enum import Enum

from second_order_different_topologies import create_graph
from helpers.circle_simulation_helpers import cartesian_to_polar, polar_to_cartesian
class MultiRobotGoalSeeker():
    def __init__(self):
        self.n_robots = 4
        self.graph_id = 5 # 4 for original, 5 for average consensus
        
        # Step size for finite derivative approximation. Requires a small value 
        # (e.g. 2e-4) for AVG_POLAR mode, but can use a larger value (e.g. 5e-3) 
        # otherwise.
        self.dynamics_time_step  = 1e-3
        self.simulation_time     = 10 # How long to run the simulation for
        # Only plot every <FRAME_SKIP>th frame for the animation
        self.frame_skip = 5
        self.A = 5
        self.m = 10
        self.J = 0.15
        self.d = 0.1

        """Weights"""
        self.K = 5 # Identical weights for all edges
        self.gamma = 1
        self.radius_formation_weight = 1
        self.angle_formation_weight = 100
        self.centre_consensus_weight = 1
        self.max_velocity = 1

        """Initial and goal formations"""
        # The goal is a shifted (centre), scaled (radius), rotated version
        # of the starting formation.
        self.start = Formation((5, 0), 2, 135)
        # TODO: Convert offset_angle to formation_angle
        # How many degrees the starting formation has to rotate by in 
        # order to give the goal (i.e. angle relative to START_ANGLE). 
        # self.goal_offset_angle = 145
        self.goal = Formation((0, 4), 2, 145)

        class Formation():
            def __init__(self, centre, radius, angle):
                self.centre = np.array(centre)
                self.radius = radius
                # Angle of the first node about the centre relative to the positive
                # x-axis. The other nodes are distribute evenly around the circle.
                self.angle = angle
            
        """Plot and file names"""
        class FormationType(Enum):
            GOAL        = "Goal"
            IDEAL_POLAR = "Ideal Polar"
            AVG_POLAR   = "Polar with Mean"

        class FormationScenario(Enum):
            ELLIPSE = 0
            ROTATE  = 1
            DEFAULT = 2

        FORMATION = FormationType.GOAL
        SCENARIO  = FormationScenario.DEFAULT

        SAVE_FIG  = True
        FILE_NAME = "{}gid_{}gamma_{}wr_{}_formation".format(
            self.graph_id, self.gamma, self.radius_formation_weight, FORMATION.value
        )

        # Plot title
        sup_title = r"Robot Trajectories for $\gamma = {}$".format(self.gamma)
        # Add a plot id to this number to get the number for a subplot
        plot_start = 220 

        if FORMATION != FormationType.GOAL:
            sup_title += ", $w_r = {}$ " \
                "and {} Formation Error".format(self.radius_formation_weight, FORMATION.value)
            plot_start = 230
                
        
        
    def generate_node_formation(self, centre, radii, angle_offset):
        """
        Get N_ROBOTS points, evenly distributed around 
        circle starting from angle.
        
        Args:
            centre ((float, float)): The centroid of the formation.
            radii (float | Callable): A constant radius or polar function used to 
                generate the radii.
            angle_offset (float): The offset in degrees of the first node. 
                The remaining nodes will be evenly distributed.
        Returns:
            (np.ndarray)
        """
        angle_difference = 2*np.pi / self.n_robots
        angles = [np.deg2rad(angle_offset) + i*angle_difference \
            for i in range(self.n_robots)]
        get_radius = radii if isinstance(radii, Callable) else lambda _ : radii
        return np.array([
            centre + polar_to_cartesian((get_radius(theta), theta)) \
                for theta in angles
        ])

    def pos(node_id):
        """
        Args:
            node_id (int)
        Returns:
            (np.ndarray): [x_hi, y_hi] ~= [x, y]
        """
        return positions[-1][node_id]

    def local_centre(node_id: int) -> np.ndarray:
        """The coordinates that node <node_id> thinks 
        is the current centre of the formation.
        """
        return centres[-1][node_id] 

    def pos_error(node_id: int) -> np.ndarray:
        """Calculates the error in the position relative to the goal.
        This is used to make each node approach the destination at the 
        same rate as its neighbours.
        Return:
            [x_ei, y_ei] = [x_hi - x_hi^d, y_hi - y_hi^d], from page 25 in [1]
        """
        return pos(node_id) - goal[node_id]

    def velocity(node_id: int) -> np.ndarray:
        """
        Return: [xdot_hi, ydot_hi]
        """
        return velocities[-1][node_id]

    def cartesian_to_polar_graph(
        cartesian_coords, centre = None
    ) -> tuple[tuple[float, float], np.ndarray]:
        """
        Convert the cartesian coordinates of all nodes to polar 
        coordinates according to:
            centre = mean(x), forall x
            r_i = || x_i - centre ||
            theta_i = angle(x_i - centre)
        
        Return:
            centre (Optional[np.ndarray]): Coordinates (x, y) of the centroid of the given 
                coordinates. This is used as the reference point that 
                each radius is measured from.
            cartesian_coords (dim: N_ROBOTS x 2): Row i represents the 
                coordinates for node i.
        """
        if not centre:
            centre = np.mean(cartesian_coords, axis=0)
        polar_coords = np.zeros(shape=cartesian_coords.shape)
        for i, pos in enumerate(cartesian_coords):
            polar_coords[i] = cartesian_to_polar(pos - centre)
            
        return centre, polar_coords

    def polar_formation_error(self, i, centre):
        """
        Get the direction (in cartesian coordinates) that a node 
        needs to move in, in order to minimise the error between 
        the current and goal formations.
        
        Args:
            i (int): Index of the node to compute the error for.
            centre: The reference point that the formation is 
                defined relative to and rotates around.
        Return:
            3x2 ndarray with rows corresponding to total, unweighted radius 
            and unweighted angle errors respectively.
        """
        cartesian_offset = pos(i) - centre
        r_magnitude = np.linalg.norm(cartesian_offset)
        # Prevent division by 0
        # HACK: Nodes may get stuck if they pass through the centre
        if r_magnitude == 0:
            return np.zeros(shape=(3, 2))
        
        # Find the error in a node's position when aligning 
        # the goal and current formation centres
        curr_polar_pos = cartesian_to_polar(cartesian_offset)
        curr_polar_goal = polar_goals[i, :]
        dr, dtheta = curr_polar_goal - curr_polar_pos
        # Get the principal argument (note in the report and literature, 
        # (-pi, pi] is used, but [-pi, pi) is used here for simplicity).
        dtheta = ((dtheta + np.pi) % (2*np.pi)) - np.pi
        
        # Get the unit radius and tangent (in anticlockwise direction) 
        # vectors, respectively, in cartesian coordinates.
        r_hat = cartesian_offset / r_magnitude
        tan_hat = np.array((-r_hat[1], r_hat[0])) # Perpendicular to r_hat
        
        radius_error_vector = dr * r_hat
        angle_error_vector = dtheta * tan_hat
        weighted_radius_error = self.radius_formation_weight * radius_error_vector
        weighted_angle_error = self.angle_formation_weight * angle_error_vector
        error_vector = weighted_radius_error + weighted_angle_error
        return np.array((error_vector, radius_error_vector, angle_error_vector))
        
    def goal_error(self, i) -> np.ndarray:
        """
        Terms in the objective function that don't 
        depend on each node's neighbours.
        Args:
            i (int): Node id
        Return:
            (np.ndarray): [ge_x, ge_y] where ge_x = -a_x*x_ei - gamma_x*a_x*xdot_hi
        """
        # Terms in objective function that depend only on i
        goto_goal = -self.A * self.pos_error(i)
        stop_when_goal_reached = -self.A * self.gamma * self.velocity(i)
        
        #return stop_when_goal_reached
        return goto_goal + stop_when_goal_reached

    def formation_error(self, i):
        """
        Terms that depend on both a node and its neighbours.
        Args:
            i (int): 0-indexed id of the node to compute the error for.
        Returns:
            (np.ndarray) 3x2 ndarray where the first row is the formation error
            and the other two rows are additional errors for plotting.
        """
        neighbours = G[i]

        formation_err   = np.zeros(shape=(3, 2))
        formation_term  = None

        if FORMATION == FormationType.GOAL:
            # Original formation term from [1]
            formation_term = lambda node: self.pos_error(node) + self.gamma * self.velocity(node)
        
        else: # Formation keeping in polar coordinates
            if FORMATION == FormationType.IDEAL_POLAR:
                avg_pos = np.mean(positions[-1], axis=0)
            else:
                # Assumes that the average consensus algorithm will have
                # faster dynamics than the main second order system.
                avg_pos = local_centre(i)
            
            formation_err = polar_formation_error(i, avg_pos)
        
        if formation_term:
            j_parts = sum([formation_term(j) for j in neighbours])
            i_parts = -len(neighbours)*formation_term(i)
            formation_err[0] += K*(j_parts + i_parts)
        
        return formation_err
        
    def objective_function(self, i):
        """
        Overall objective/error function (u) for change in velocity.
        
        Args:
            i (int)
        Returns:
            (np.ndarray)
        """
        ge = self.goal_error(i)
        fe = self.formation_error(i)
        total_error = ge + fe[0]
        return ge, fe, total_error

    def polar_ellipse(a, b, theta):
        return a*b / np.sqrt((b*np.cos(theta))**2 + (a*np.sin(theta))**2)

    def get_new_centre(i: int, delta_pos: np.ndarray) -> np.ndarray:
        """
        Calculate the centre of the formation according to node i using
        a first-order consensus objective function.
        """
        neighbours = G[i]
        # Node i knows how much it moved, so update its contribution to the centre value
        new_local_centre = local_centre(i) + delta_pos
        # Rely on the consensus protocol to get the contributions 
        # of the other nodes to the centre value
        u = K*(sum([local_centre(j) for j in neighbours]) \
            - len(neighbours)*new_local_centre)
        new_local_centre += DYNAMICS_TIME_STEP * CENTRE_CONSENSUS_WEIGHT * u
        return new_local_centre

    def run_simulation(self) -> None:
        global positions
        global velocities
        global goal_errors
        global formation_errors
        global centres
        
        # Append lists in the format used for this simulation
        sim_append = lambda full_list, new_list : \
            np.concatenate((full_list, np.expand_dims(np.array(new_list), 0)))
        
        for _ in times:
            new_positions = []
            new_velocities = []
            new_goal_errors = []
            new_formation_errors = []
            new_centres = []
            
            for i in range(self.n_robots):
                # Calculate the state for the ith robot for the current 
                # time-step using a finite difference
                delta_pos = self.dynamics_time_step * velocity(i)
                new_positions.append(pos(i) + delta_pos)
                ge, fe, total_error = objective_function(i)
                new_goal_errors.append(ge)
                new_formation_errors.append(fe)
                
                # Saturate the velocity to simulate the maximum speed of the robot
                new_velocities.append(np.clip(
                    velocity(i) + DYNAMICS_TIME_STEP*total_error,
                    -MAX_VELOCITY, MAX_VELOCITY
                ))
                
                new_centres.append(get_new_centre(i, delta_pos))
                
            positions        = sim_append(positions, new_positions)
            velocities       = sim_append(velocities, new_velocities)
            goal_errors      = sim_append(goal_errors, new_goal_errors)
            formation_errors = sim_append(formation_errors, new_formation_errors)
            centres          = sim_append(centres, new_centres)

    def get_bounding_box(arrays: Iterable):
        """
        Given multiple arrays of (x, y) coordinates, calculates the
        coordinates of a box that includes all of the points.
        
        NaN and inf values are ignored.
        
        Args:
            arrays: Tuple of np.ndarrays, each with dimensions Nx2
        Return:
            xlims, ylims
        """
        for i in arrays:
            if i.shape[1] != 2:
                raise ValueError("Array had invalid dimensions")
        
        # Split into x and y components
        good_array_values = lambda array : \
            not (np.isnan(array).any() or np.isinf(array).any())
        x_values = [i[:, 0] for i in arrays if good_array_values(i)]
        y_values = [i[:, 1] for i in arrays if good_array_values(i)]
        x_min = np.floor(np.min(x_values)) - 1
        x_max = np.ceil(np.max(x_values))
        y_min = np.floor(np.min(y_values)) - 1
        y_max = np.ceil(np.max(y_values)) + 1
        return (x_min, x_max), (y_min, y_max)
        
    def setup_plot():
        """Draw plot except for the animation."""
        # Setup
        ax: tuple[plt.Axes, plt.Axes]
        n_cols = 2 if FORMATION == FormationType.GOAL else 3
        _, ax = plt.subplots(2, n_cols, figsize=(4*n_cols, 8)) 
            
        plt.suptitle(sup_title, fontsize='x-large')
        
        """Plot robot trajectories"""
        plt.subplot(plot_start + 1)
        plt.title("Robot Trajectories")
        plt.xlabel("x (m)", fontsize='large')
        plt.ylabel("y (m)", fontsize='large')
        
        # Calculate rough bounds for the trajectories 
        # using the initial position, final position and goal and goal
        xlims, ylims = get_bounding_box((goal, positions[0], positions[-1]))
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.grid(True, linewidth=0.15)
        
        pos_lines       = []
        start_markers   = []
        end_markers     = []
        goal_markers    = []
        handles         = []
        x_coords = positions[:, :, 0]
        y_coords = positions[:, :, 1]
        x_goal, y_goal = list(zip(*goal))
        COLOURS = ["r", "b", "k", "m"]
        
        for id, colour in enumerate(COLOURS):
            # Create lines with no points to be used for the animation 
            pos_lines.extend(plt.plot([], [], f'{colour}', linewidth=1))
            
            # Plot a complete, thin version of the trajectories to be animated
            plt.plot(x_coords[:, id], y_coords[:, id], f'{colour}:', linewidth=0.4)
            
            # Plot markers for start, actual end and desired end points respectively
            start_markers.extend(plt.plot(x_coords[0, id], 
                y_coords[0, id], f'{colour}o', ls='', ms=4))
            
            end_markers.extend(plt.plot(x_coords[-1, id], 
                y_coords[-1, id], f'{colour}s', ls='', ms=4))
            
            goal_markers.extend(plt.plot(x_goal[id],
                y_goal[id], f'{colour}*', ls='', ms=4))
            
        # Plot the legend for the first plot
        labels = ["robot {}".format(i + 1) for i in range(N_ROBOTS)]
        labels.extend(["start", "end", "goal"])
        handles.extend(pos_lines)
        handles.extend([start_markers[2], end_markers[2], goal_markers[2]])
        
        """Plot the internal average consensus values of each robot"""
        avg_lines = None

        # Don't animate the average position when using the protocol from [1]
        if FORMATION != FormationType.GOAL:
            plt.subplot(plot_start + 2)
            avg_lines = []
            x_coords = centres[:, :, 0]
            y_coords = centres[:, :, 1]
            
            for id, colour in enumerate(COLOURS):
                # Create lines with no points to be used for the animation 
                avg_lines.extend(plt.plot([], [], f'{colour}', linewidth=1))
                
                # Plot a complete, thin version of the 
                # trajectories to be animated
                plt.plot(x_coords[:, id], y_coords[:, id], 
                    f'{colour}:', linewidth=0.4)
                
                # Plot markers for start, actual end and 
                # desired end points respectively
                plt.plot(x_coords[0, id], y_coords[0, id], 
                    f'{colour}o', ls='', ms=4)
                
                plt.plot(x_coords[-1, id], y_coords[-1, id], 
                    f'{colour}s', ls='', ms=4)
                
            # By definition of consensus, there is only one goal
            plt.plot(goal_avg[0], goal_avg[1], f'k*', ls='', ms=4)
            
            plt.title("Average Consensus Trajectories")
            plt.xlabel("x (m)", fontsize='large')
            plt.ylabel("y (m)", fontsize='large')
        
            # Calculate limits
            xlims, ylims = get_bounding_box((goal, centres[0], centres[-1]))
            plt.xlim(xlims)
            plt.ylim(ylims)
            plt.grid(True, linewidth=0.15)
        
        """Plot velocity"""
        plt.subplot(plot_start + n_cols + 1)
        plt.title("Robot Velocities")
        for id, colour in enumerate(COLOURS):
            plt.plot(times, np.linalg.norm(velocities[:, id], axis=1), 
                f'{colour}', lw=1)
            
        plt.xlabel("Time (s)", fontsize='large')
        plt.ylabel("Norm of velocity (m/s)", fontsize='large')
        plt.grid(True, lw=0.15)
        
        """Error plots"""
        plt.subplot(plot_start + n_cols + 2)
        plt.title("Total Goal Error")
        for id, colour in enumerate(COLOURS):
            plt.plot(times, np.linalg.norm(goal_errors[:, id], axis=1), 
                f'{colour}', lw=1)
            
        plt.xlabel("Time (s)", fontsize='large')
        plt.ylabel("Norm of goal error", fontsize='large')
        plt.grid(True, lw=0.15)  
        
        # Different formation error plot for goal 
        # vs other formation error terms
        if FORMATION == FormationType.GOAL:
            formation_err_norm = np.linalg.norm(formation_errors[:, :, 0, :], axis=2)
            plt.subplot(plot_start + 2)
            plt.title("Formation Error")
            for id, colour in enumerate(COLOURS):
                plt.plot(times, formation_err_norm[:, id], f'{colour}', lw=1)
            plt.ylabel("Norm of formation error", fontsize='large')
            
        else:
            avg_formation_errors = np.mean(formation_errors, axis=1)
            avg_formation_magnitude = np.linalg.norm(avg_formation_errors, axis=2)
            plt.subplot(plot_start + 6)
            plt.title("Unweighted Formation Errors")
            plt.plot(times, avg_formation_magnitude[:, 1] + avg_formation_magnitude[:, 2], lw=1)
            plt.plot(times, avg_formation_magnitude[:, 1:], lw=1)
            plt.legend(["Total", "Radius", "Angle"])
            plt.ylabel("Mean over all nodes of formation error norms", fontsize='medium')

        plt.xlabel("Time (s)", fontsize='large')
        plt.grid(True, lw=0.15)
        
        """Legend"""
        if FORMATION == FormationType.GOAL:
            ax[1, 1].legend(
                labels          = labels,
                handles         = handles,
                loc             = 'upper left', 
                borderpad       = 0.2,
                ncol            = 1,
                title           = "Legend",
                title_fontsize  = 'large',
                bbox_to_anchor   = (1, 1.4)
            )
            plt.tight_layout(h_pad=-5, rect=(0, 0, 1.01, 1))
        else:
            plt.axis('off')
            ax[0, 2].legend(
                labels          = labels,
                handles         = handles,
                loc             = 'center', 
                borderpad       = 2,
                ncol            = 2,
                title           = "Legend",
                title_fontsize  = 'large'
            )
            plt.tight_layout()
        
        return pos_lines, avg_lines

    x_data, y_data =[], []

    def init_animation() -> list:
        plt.subplot(plot_start + 1)
        return lines

    def animate_trajectories(time_step: int) -> list:
        """Handle one time step of the animation."""
        end = time_step + 1
        
        # Assign x and y coordinates to be plotted
        x = data[:end, :, 0]
        y = data[:end, :, 1]
        for i, line in enumerate(lines):
            line.set_data(x[:, i], y[:, i])
        
        print("{}\r".format(time_step), end="")
        return lines

if __name__ == "__main__":
    # Set up robots
    G = create_graph(GRAPH_ID)
    
    # Dimension of positions and velocities: Time x node x dim
    # Points for positions and goals are 
    # distributed evenly around a circle
    
    # Initial and goal formations that allow each 
    # robot to move symmetrically
    if SCENARIO == FormationScenario.ELLIPSE:
        ellipse_formation = lambda angle: polar_ellipse(0.75, 1.5, angle)
        positions = np.array([
            generate_node_formation(START_CENTRE, ellipse_formation, 0)])
        goal = generate_node_formation(START_CENTRE, START_RADIUS, 0)
    
    # Rotate to reach goal
    elif SCENARIO == FormationScenario.ROTATE:
        positions = np.array([generate_node_formation(0, 1, 0)])
        goal = generate_node_formation(0, 1, 80)
    
    else:
        # Original goal and positions
        positions = np.array([generate_node_formation(\
            START_CENTRE, START_RADIUS, START_ANGLE)])
        goal = generate_node_formation(GOAL_CENTRE, \
            GOAL_RADIUS, START_ANGLE + GOAL_OFFSET_ANGLE)
    
    goal_avg, polar_goals = cartesian_to_polar_graph(goal)
    velocities = np.array([[(0, 0) for _ in range(N_ROBOTS)]])
    goal_errors = np.zeros(shape=(0, N_ROBOTS, 2))
    formation_errors = np.zeros(shape=(0, N_ROBOTS, 3, 2))
    centres = positions.copy()
    
    # Simulate
    times = np.arange(0, SIMULATION_TIME, DYNAMICS_TIME_STEP)
    run_simulation()
    positions = positions[:-1]
    velocities = velocities[:-1]
    centres = centres[:-1]
    
    # Set up data for animation
    lines, avg_lines = setup_plot()
    if avg_lines:
        lines.extend(avg_lines)
    data = np.concatenate((positions, centres), axis=1)
    
    anim = animation.FuncAnimation(
        fig         = plt.gcf(),
        init_func   = init_animation,
        func        = animate_trajectories, 
        frames      = list(range(0, len(times), FRAME_SKIP)),
        interval    = 1,
        blit        = True
    )
    
    if SAVE_FIG:
        anim.save(FILE_NAME + ".mp4", fps=144, dpi=350)
        plt.savefig(FILE_NAME + ".pdf")
        
    plt.show()