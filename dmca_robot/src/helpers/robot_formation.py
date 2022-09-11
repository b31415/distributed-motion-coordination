
import numpy as np
from numbers import Number
from helpers.circle_simulation_helpers import polar_to_cartesian

def generate_node_formation(n_robots, centre, radii, angle_offset):
    """
    Get <self.n_robots> points, evenly distributed around 
    circle starting from angle.
    
    Args:
        n_robots (float)
        centre ((float, float)): The centroid of the formation.
        radii (float | Callable): A constant radius or polar function used to 
            generate the radii.
        angle_offset (float): The offset in radians of the first node. 
            The remaining nodes will be evenly distributed.
    Returns:
        (np.ndarray)
    """
    angle_difference = 2*np.pi / n_robots
    angles = [angle_offset + i*angle_difference \
        for i in range(n_robots)]
    get_radius = (lambda _ : radii) if isinstance(radii, Number) else radii
    return np.array([
        centre + polar_to_cartesian((get_radius(theta), theta)) \
            for theta in angles
    ])

def polar_ellipse(a, b, theta):
    """
    Args:
        a (float): Semi-major axis
        b (float): Semi-minor axis
        theta (float): Angle in radians
    """
    return a*b / np.sqrt((b*np.cos(theta))**2 + (a*np.sin(theta))**2)

def generate_ellipse(
    n_robots,
    centre,
    angle_offset,
    x_axis_radius,
    y_axis_radius
):
    """
    Args:
        angle_offset (float): Angle in radians that the ellipse is rotated by.
        y_axis_radius (Optional[float]): If None, use the same radius as 
            the x-axis radius to produce a circle.
    """
    if y_axis_radius is None:
        y_axis_radius = x_axis_radius
        
    ellipse_point = lambda angle: polar_ellipse(x_axis_radius, y_axis_radius, angle)
    positions = generate_node_formation(n_robots, centre, ellipse_point, angle_offset)
    return positions

def generate_circle(n_robots, centre, angle_offset, radius):
    return generate_ellipse(n_robots, centre, angle_offset, radius, radius)