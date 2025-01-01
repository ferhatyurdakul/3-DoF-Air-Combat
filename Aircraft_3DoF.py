import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Aircraft:
    def __init__(self, x, y, h, v, psi, gamma):
        # Positions
        self.x = x
        self.y = y
        self.h = h

        # Velocity
        self.v = v

        # Angles
        self.psi = psi     # Heading angle
        self.gamma = gamma # Flight path angle

        # Constants
        self.dt = 0.01
        self.g  = 9.81

        # Log list
        self.log = {
        "x": [x],
        "y": [y],
        "h": [h],
        "v": [v],
        "psi": [psi],
        "gamma": [gamma]}

    def update(self, nx, nz, mu):
        # Calculations
        # Iterating the values for 1 second using dt
        for _ in np.arange(0, 1, self.dt):
            x_dot = self.v * np.cos(self.psi) * np.cos(self.gamma)
            y_dot = self.v * np.sin(self.psi) * np.cos(self.gamma)
            h_dot = self.v * np.sin(self.gamma)

            v_dot = self.g * (nx - np.sin(self.gamma))
            psi_dot = (self.g * nz / self.v) * (np.sin(mu) / np.cos(self.gamma))
            gamma_dot = (self.g / self.v) * (nz * np.cos(mu) - np.cos(self.gamma))

            self.x += x_dot * self.dt
            self.y += y_dot * self.dt
            self.h += h_dot * self.dt

            self.v += v_dot * self.dt
            self.psi += psi_dot * self.dt
            self.gamma += gamma_dot * self.dt

            # Udpating the logs
            self.log["x"].append(self.x)
            self.log["y"].append(self.y)
            self.log["h"].append(self.h)
            self.log["v"].append(self.v)
            self.log["psi"].append(self.psi)
            self.log["gamma"].append(self.gamma)

    def WEZ(self, px, py, ph, aperture=20, height=300):
        """
        Checks if a given point is within the Weapon Engagement Zone (WEZ) cone.
        Parameters:
            px, py, ph : float
                Coordinates of the target point (x, y, altitude).
            aperture : float
                Cone aperture angle in degrees.
            height : float
                Height (length) of the cone.
        Returns:
            bool : True if the point is in WEZ, False otherwise.
        """
        # Vector from aircraft to point
        dx = px - self.x
        dy = py - self.y
        dh = ph - self.h

        # Distance to the point
        point_distance = np.sqrt(dx**2 + dy**2 + dh**2)

        # Direction of the point relative to aircraft heading
        heading_vector = np.array([np.cos(self.psi) * np.cos(self.gamma),
                                   np.sin(self.psi) * np.cos(self.gamma),
                                   np.sin(self.gamma)])
        point_vector = np.array([dx, dy, dh]) / point_distance  # Normalize to unit vector

        # Compute the angle between the heading vector and point vector
        cosine_angle = np.dot(heading_vector, point_vector)
        angle = np.degrees(np.arccos(cosine_angle))

        # Distance from the tip of the cone to the bottom surface of the cone with the angle of point
        cone_distance = height / np.cos(np.radians(angle))

        # Check if point is within aperture angle
        return angle <= aperture / 2 and point_distance <= cone_distance 
    
    def render(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Initialize the plot elements
        line, = ax.plot([], [], [], 'b-', label="Trajectory")
        point, = ax.plot([], [], [], 'ro', label="F-16")
        heading_vector_line, = ax.plot([], [], [], 'g-', label="Heading Vector")
        ax.legend()

        # Setting axis labels and title
        ax.set_xlabel("X Position [m]")
        ax.set_ylabel("Y Position [m]")
        ax.set_zlabel("Altitude [m]")
        ax.set_title("F-16 Aircraft Movement")

        def update(frame):
            # Update trajectory
            line.set_data(self.log["x"][:frame+1], self.log["y"][:frame+1])
            line.set_3d_properties(self.log["h"][:frame+1])

            # Update current position
            point.set_data([self.log["x"][frame]], [self.log["y"][frame]])
            point.set_3d_properties([self.log["h"][frame]])

            # Calculate heading vector
            heading_length = 100  # Length of the heading vector
            heading_vector = np.array([
                np.cos(self.log["psi"][frame]) * np.cos(self.log["gamma"][frame]),
                np.sin(self.log["psi"][frame]) * np.cos(self.log["gamma"][frame]),
                np.sin(self.log["gamma"][frame])
            ]) * heading_length

            heading_x = [self.log["x"][frame], self.log["x"][frame] + heading_vector[0]]
            heading_y = [self.log["y"][frame], self.log["y"][frame] + heading_vector[1]]
            heading_z = [self.log["h"][frame], self.log["h"][frame] + heading_vector[2]]

            heading_vector_line.set_data(heading_x, heading_y)
            heading_vector_line.set_3d_properties(heading_z)

            # Dynamically adjust the axis limits to follow the aircraft
            ax.set_xlim(self.log["x"][frame] - 200, self.log["x"][frame] + 200)
            ax.set_ylim(self.log["y"][frame] - 200, self.log["y"][frame] + 200)
            ax.set_zlim(self.log["h"][frame] - 200, self.log["h"][frame] + 200)

            return line, point, heading_vector_line

        ani = FuncAnimation(fig, update, frames=len(self.log["x"]), interval=50, blit=False)

        # Display the animation
        plt.show()

    def trajectory_plot(self):
        """
        Plots the entire trajectory of the aircraft in 3D space.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(self.log["x"], self.log["y"], self.log["h"], label="Trajectory", color='b')

        # Setting axis labels and title
        ax.set_xlabel("X Position [m]")
        ax.set_ylabel("Y Position [m]")
        ax.set_zlabel("Altitude [m]")
        ax.set_title("F-16 Aircraft Full Trajectory")
        ax.legend()

        # Show the plot
        plt.show()

if __name__ == "__main__":
    F16 = Aircraft(0, 0, 1000, 350, 0, 0)

    # Simulate the movement of F16 for 20 seconds
    for i in np.arange(10):
        if i < -10:
            F16.update(1.0, 6.0, 1.22)
        else:
            F16.update(0, 0, 0)

    F16.trajectory_plot()
    F16.render()
