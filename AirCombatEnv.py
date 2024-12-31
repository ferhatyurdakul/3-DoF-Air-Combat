import numpy as np
from Aircraft_3DoF import Aircraft
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class F16Environment(gym.Env):
    """
    Custom Gym environment for simulating two F-16 aircrafts.
    """
    def __init__(self):
        super(F16Environment, self).__init__()

        # Action space: nx, nz, mu for both aircrafts in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation space: Positions, velocities, and angles for both aircrafts
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),  # 6 parameters for each aircraft
            dtype=np.float32
        )

        # Limits for nx, nz, and mu for the F-16
        self.nx_limits = [0.0, 3.0]   # Longitudinal load factor
        self.nz_limits = [0.0, 9.0]   # Normal load factor (G-forces)
        self.mu_limits = [-5*np.pi/12, 5*np.pi/12]  # Bank angle in radians

        # Initialize two aircraft
        self.aircraft1 = Aircraft(0, 0, 1000, 250, 0, 0)
        self.aircraft2 = Aircraft(0, 1000, 1000, 250, np.pi, 0)

        # Constants
        self.distance_limit = 1000

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.aircraft1 = Aircraft(0, 0, 1000, 250, 0, 0)
        self.aircraft2 = Aircraft(0, 1000, 1000, 250, np.pi, 0)
        
        observation = self._get_observation()
        return observation

    def step(self, action):
        """
        Execute one time step within the environment.
        Parameters:
            action: np.array
                Action values for both aircrafts in range [-1, 1].
        Returns:
            observation: np.array
                The updated state of the environment.
            reward: float
                Reward value based on actions and state.
            done: bool
                Whether the simulation has ended.
            info: dict
                Additional information.
        """
        # Scale actions to the real nx, nz, and mu values
        nx1 = self._scale_action(action[0], self.nx_limits)
        nz1 = self._scale_action(action[1], self.nz_limits)
        mu1 = self._scale_action(action[2], self.mu_limits)

        nx2 = self._scale_action(action[3], self.nx_limits)
        nz2 = self._scale_action(action[4], self.nz_limits)
        mu2 = self._scale_action(action[5], self.mu_limits)

        # Update aircraft states
        self.aircraft1.update(nx1, nz1, mu1)
        # self.aircraft2.update(nx2, nz2, mu2) # TODO: Aircraft Stationary

        # Check wez
        if self.aircraft1.WEZ(self.aircraft2.x, self.aircraft2.y, self.aircraft2.h):
            print("Aircraft 1 wins")
            done = True
            reward = 10
        if self.aircraft2.WEZ(self.aircraft1.x, self.aircraft1.y, self.aircraft1.h):
            print("Aircraft 2 wins")
            done = True
            reward = -5
        else:
            done = False
            reward = 0

        # Get the new observation
        observation = self._get_observation()

        # Calculate reward (can be customized further)
        reward += self._calculate_reward()

        info = {}
        return observation, reward, done, info

    def _scale_action(self, action, limits):
        """
        Scale an action from [-1, 1] to the real range specified by limits.
        """
        return limits[0] + (action + 1) * (limits[1] - limits[0]) / (2)

    def _get_observation(self):
        """
        Combine the states of both aircraft into a single observation.
        """
        obs1 = [
            self.aircraft1.x, self.aircraft1.y, self.aircraft1.h,
            self.aircraft1.v, self.aircraft1.psi, self.aircraft1.gamma
        ]
        obs2 = [
            self.aircraft2.x, self.aircraft2.y, self.aircraft2.h,
            self.aircraft2.v, self.aircraft2.psi, self.aircraft2.gamma
        ]
        return np.array(obs1 + obs2, dtype=np.float32)

    def _calculate_reward(self):
        """
        Placeholder reward function. Customize based on objectives.
        """
        distance = np.sqrt((self.aircraft1.x - self.aircraft2.x) ** 2 +
                           (self.aircraft1.y - self.aircraft2.y) ** 2 +
                           (self.aircraft1.y - self.aircraft2.y) ** 2)

        if distance > self.distance_limit:
            distance = self.distance_limit

        reward = 1 - distance/self.distance_limit

        return reward

    def render(self):
        """
        Render the movements of both aircrafts continuously during the simulation using their current states,
        including headings and Weapon Engagement Zones (WEZ).
        """
        if not hasattr(self, '_fig'):
            # Initialize the figure and axis only once
            self._fig = plt.figure(figsize=(10, 7))
            self._ax = self._fig.add_subplot(111, projection='3d')

            # Initialize plot elements for aircraft 1
            self._line1, = self._ax.plot([], [], [], 'b-', label="Aircraft 1 Trajectory")
            self._point1, = self._ax.plot([], [], [], 'ro', label="Aircraft 1")
            self._heading1, = self._ax.plot([], [], [], 'g-', label="Aircraft 1 Heading")
            self._wez1, = self._ax.plot([], [], [], 'g--', label="Aircraft 1 WEZ")

            # Initialize plot elements for aircraft 2
            self._line2, = self._ax.plot([], [], [], 'g-', label="Aircraft 2 Trajectory")
            self._point2, = self._ax.plot([], [], [], 'yo', label="Aircraft 2")
            self._heading2, = self._ax.plot([], [], [], 'r-', label="Aircraft 2 Heading")
            self._wez2, = self._ax.plot([], [], [], 'r--', label="Aircraft 2 WEZ")

            # Set labels and title
            self._ax.set_xlabel("X Position [m]")
            self._ax.set_ylabel("Y Position [m]")
            self._ax.set_zlabel("Altitude [m]")
            self._ax.set_title("Aircraft Movements")
            self._ax.legend()

        def compute_heading_and_wez(aircraft, heading_length=100, cone_height=500, aperture=np.radians(2 / 2)):
            # Compute heading vector
            heading_vector = np.array([
                np.cos(aircraft.psi) * np.cos(aircraft.gamma),
                np.sin(aircraft.psi) * np.cos(aircraft.gamma),
                np.sin(aircraft.gamma)
            ]) * heading_length

            heading_x = [aircraft.x, aircraft.x + heading_vector[0]]
            heading_y = [aircraft.y, aircraft.y + heading_vector[1]]
            heading_z = [aircraft.h, aircraft.h + heading_vector[2]]

            # Compute WEZ cone
            cone_x, cone_y, cone_z = [], [], []
            for h in np.linspace(0, cone_height, 10):
                radius = h * np.tan(aperture)
                for theta in np.linspace(0, 2 * np.pi, 36):
                    offset_x = radius * np.cos(theta)
                    offset_y = radius * np.sin(theta)

                    cone_x.append(aircraft.x + h * heading_vector[0] / heading_length + offset_x)
                    cone_y.append(aircraft.y + h * heading_vector[1] / heading_length + offset_y)
                    cone_z.append(aircraft.h + h * heading_vector[2] / heading_length)

            return heading_x, heading_y, heading_z, cone_x, cone_y, cone_z

        # Update trajectory and heading for Aircraft 1
        self._line1.set_data([self.aircraft1.x], [self.aircraft1.y])
        self._line1.set_3d_properties([self.aircraft1.h])
        self._point1.set_data([self.aircraft1.x], [self.aircraft1.y])
        self._point1.set_3d_properties([self.aircraft1.h])

        heading1_x, heading1_y, heading1_z, wez1_x, wez1_y, wez1_z = compute_heading_and_wez(self.aircraft1)
        self._heading1.set_data(heading1_x, heading1_y)
        self._heading1.set_3d_properties(heading1_z)
        self._wez1.set_data(wez1_x, wez1_y)
        self._wez1.set_3d_properties(wez1_z)

        # Update trajectory and heading for Aircraft 2
        self._line2.set_data([self.aircraft2.x], [self.aircraft2.y])
        self._line2.set_3d_properties([self.aircraft2.h])
        self._point2.set_data([self.aircraft2.x], [self.aircraft2.y])
        self._point2.set_3d_properties([self.aircraft2.h])

        heading2_x, heading2_y, heading2_z, wez2_x, wez2_y, wez2_z = compute_heading_and_wez(self.aircraft2)
        self._heading2.set_data(heading2_x, heading2_y)
        self._heading2.set_3d_properties(heading2_z)
        self._wez2.set_data(wez2_x, wez2_y)
        self._wez2.set_3d_properties(wez2_z)

        # Dynamically adjust axis limits to fit both aircraft
        all_x = [self.aircraft1.x, self.aircraft2.x]
        all_y = [self.aircraft1.y, self.aircraft2.y]
        all_h = [self.aircraft1.h, self.aircraft2.h]

        self._ax.set_xlim(min(all_x) - 200, max(all_x) + 200)
        self._ax.set_ylim(min(all_y) - 200, max(all_y) + 200)
        self._ax.set_zlim(min(all_h) - 200, max(all_h) + 200)

        # Redraw the figure
        plt.pause(0.01)

    def trajectory_plot(self):
        """
        Plots the entire trajectory of both aircrafts in 3D space using their current positions.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory for Aircraft 1
        ax.plot(self.aircraft1.log["x"], self.aircraft1.log["y"], self.aircraft1.log["h"], label="Aircraft 1 Trajectory", color='b')
        ax.scatter(self.aircraft1.log["x"][-1], self.aircraft1.log["y"][-1], self.aircraft1.log["h"][-1], color='r', label="Aircraft 1 Final Position")

        # Plot trajectory for Aircraft 2
        ax.plot(self.aircraft2.log["x"], self.aircraft2.log["y"], self.aircraft2.log["h"], label="Aircraft 2 Trajectory", color='g')
        ax.scatter(self.aircraft2.log["x"][-1], self.aircraft2.log["y"][-1], self.aircraft2.log["h"][-1], color='y', label="Aircraft 2 Final Position")

        # Set axis labels and title
        ax.set_xlabel("X Position [m]")
        ax.set_ylabel("Y Position [m]")
        ax.set_zlabel("Altitude [m]")
        ax.set_title("3D Trajectories of Aircrafts")
        ax.legend()

        # Show the plot
        plt.show()

# Example usage
if __name__ == "__main__":
    env = F16Environment()
    obs = env.reset()
    done = False

    for i in range(10):
        action = np.random.uniform(0, 0, size=(6,))
        obs, reward, done, info = env.step([-1,-1,0,0,0,0])
        
    env.aircraft1.render()

    env.trajectory_plot()