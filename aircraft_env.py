import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional
from datetime import datetime
import pickle

class AircraftEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, 
                 render_mode: Optional[str] = None, 
                 mode: str = "manual",  # "manual" or "bc"
                 model=None):  # For BC model
        super().__init__()
        
        # Environment constants
        self.CONSTANT_SPEED = 150.0  # Fixed speed
        self.ALTITUDE = 500.0
        self.MAX_YAW_RATE = np.pi/2
        self.SENSOR_RANGE = 500.0
        self.SENSOR_ANGLES = [-60, -30, 0, 30, 60]
        self.WAYPOINT_RANGE = 1000.0
        self.WAYPOINT_RADIUS = 75.0
        
        # Operation mode and model
        self.mode = mode
        self.model = model
        
        # State initialization
        self.state = None
        self.current_waypoint = None
        self.distance_to_waypoint = float('inf')
        self.previous_distance_to_waypoint = None
        self.previous_heading = None
        self.boundary_distances = np.zeros(4)  # [north, east, south, west]
        
        # Demonstration recording
        self.demonstration_data = []
        self.recording = False
        
        # Setup spaces
        self._setup_spaces()
        
        # Metrics tracking
        self.metrics = {
            'episode_length': 0,
            'total_reward': 0.0,
            'collision_count': 0,
            'success_count': 0,
            'boundary_violations': 0
        }
        
        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.camera_offset = np.array([400, 300])
        self.zoom = 0.2
        
        # Initialize obstacles
        self.obstacles = []
        self.reset()

    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # State space: [x, y, yaw, sensor_readings, boundary_distances]
        self.observation_space = spaces.Box(
            low=np.array([
                -np.inf, -np.inf,  # position
                -np.pi,            # yaw
                *[0.0] * len(self.SENSOR_ANGLES),  # sensor readings
                *[0.0] * 4         # boundary distances [north, east, south, west]
            ]),
            high=np.array([
                np.inf, np.inf,    # position
                np.pi,             # yaw
                *[self.SENSOR_RANGE] * len(self.SENSOR_ANGLES),
                *[self.WAYPOINT_RANGE * 2] * 4  # max boundary distances
            ]),
            dtype=np.float32
        )
        
        # Action space: [yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-0.1]),
            high=np.array([0.1]),
            dtype=np.float32
        )

    def get_sensor_readings(self):
        """Get sensor readings"""
        readings = []
        
        for angle in self.SENSOR_ANGLES:
            abs_angle = self.state[2] + np.deg2rad(angle)
            direction = np.array([
                np.cos(abs_angle),
                np.sin(abs_angle)
            ])
            
            min_distance = self.SENSOR_RANGE
            
            # Check obstacles
            for obstacle in self.obstacles:
                relative_pos = obstacle['position'][:2] - self.state[:2]
                half_size = obstacle['size'] / 2
                
                # Check intersection with obstacle boundaries
                for axis in range(2):
                    if abs(direction[axis]) > 1e-6:  # Avoid division by zero
                        for sign in [-1, 1]:
                            d = (relative_pos[axis] + sign * half_size) / direction[axis]
                            if d > 0:  # Only consider points ahead of the sensor
                                intersection = self.state[:2] + direction * d
                                rel_int = intersection - obstacle['position'][:2]
                                if abs(rel_int[1-axis]) <= half_size:
                                    min_distance = min(min_distance, d)
            
            # Check boundaries
            boundary = self.WAYPOINT_RANGE
            pos = self.state[:2]
            
            # Check each boundary line (top, right, bottom, left)
            boundaries = [
                (boundary - pos[1]) / direction[1] if abs(direction[1]) > 1e-6 else float('inf'),  # North
                (boundary - pos[0]) / direction[0] if abs(direction[0]) > 1e-6 else float('inf'),  # East
                (-boundary - pos[1]) / direction[1] if abs(direction[1]) > 1e-6 else float('inf'), # South
                (-boundary - pos[0]) / direction[0] if abs(direction[0]) > 1e-6 else float('inf')  # West
            ]
            
            # Only consider positive distances (in front of the sensor)
            # and check if intersection point is within boundary limits
            for i, d in enumerate(boundaries):
                if d > 0:
                    intersection = pos + direction * d
                    if i % 2 == 0:  # North/South boundaries
                        if abs(intersection[0]) <= boundary:
                            min_distance = min(min_distance, d)
                    else:  # East/West boundaries
                        if abs(intersection[1]) <= boundary:
                            min_distance = min(min_distance, d)
            
            readings.append(min_distance)
        
        return np.array(readings)

    def _update_boundary_distances(self):
        """Update distances to boundaries"""
        x, y = self.state[:2]
        boundary = self.WAYPOINT_RANGE
        
        # Calculate distances to boundaries [north, east, south, west]
        # Fix: The original calculation was incorrect for south and west boundaries
        self.boundary_distances = np.array([
            boundary - y,    # distance to north boundary (correct)
            boundary - x,    # distance to east boundary (correct)
            boundary + y,    # distance to south boundary (was incorrect)
            boundary + x     # distance to west boundary (was incorrect)
        ])
        
    def _check_collision(self):
        """Check for collisions with obstacles or boundaries"""
        # Check boundary collision
        if np.min(self.boundary_distances) < 0:
            self.metrics['boundary_violations'] += 1
            return True
            
        # Check obstacle collision
        for obstacle in self.obstacles:
            relative_pos = self.state[:2] - obstacle['position'][:2]
            if np.linalg.norm(relative_pos) < obstacle['size'] / 2:
                return True
                
        return False

    def _calculate_reward(self, collision):
        """Calculate reward based on progress and safety"""
        if collision:
            return -100.0  # Severe collision penalty
        
        # Calculate progress towards goal
        current_distance = np.linalg.norm(self.current_waypoint - self.state[:2])
        delta_distance = self.previous_distance_to_waypoint - current_distance
        
        # Progress reward
        progress_reward = 20.0 * delta_distance
        
        # Heading alignment reward
        direction_to_waypoint = np.arctan2(
            self.current_waypoint[1] - self.state[1],
            self.current_waypoint[0] - self.state[0]
        )
        heading_diff = abs((direction_to_waypoint - self.state[2] + np.pi) % (2 * np.pi) - np.pi)
        heading_reward = -10.0 * (heading_diff / np.pi)
        
        # Boundary distance reward
        min_boundary_dist = np.min(self.boundary_distances)
        boundary_threshold = 200.0
        if min_boundary_dist < boundary_threshold:
            boundary_penalty = -20.0 * (1.0 - min_boundary_dist/boundary_threshold)
        else:
            boundary_penalty = 0.0
        
        # Obstacle avoidance reward
        min_obstacle_dist = min([
            np.linalg.norm(self.state[:2] - obstacle['position'][:2]) - obstacle['size']/2
            for obstacle in self.obstacles
        ])
        obstacle_threshold = 150.0
        if min_obstacle_dist < obstacle_threshold:
            obstacle_penalty = -15.0 * (1.0 - min_obstacle_dist/obstacle_threshold)
        else:
            obstacle_penalty = 0.0
        
        # Combine rewards
        reward = (
            progress_reward +     # Progress towards goal
            heading_reward +      # Heading alignment
            boundary_penalty +    # Boundary avoidance
            obstacle_penalty      # Obstacle avoidance
        )
        
        # Goal proximity bonus
        if current_distance < self.WAYPOINT_RADIUS * 2:
            reward += 20.0 * (1.0 - current_distance/(self.WAYPOINT_RADIUS * 2))
        
        return reward

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset metrics
        self.metrics = {
            'episode_length': 0,
            'total_reward': 0.0,
            'collision_count': 0,
            'success_count': 0,
            'boundary_violations': 0
        }
        
        # Reset aircraft state
        self.state = np.array([
            0.0,    # x position
            0.0,    # y position
            0.0,    # yaw (heading)
            *[self.SENSOR_RANGE] * len(self.SENSOR_ANGLES),  # sensor readings
            *[self.WAYPOINT_RANGE] * 4  # initial boundary distances
        ], dtype=np.float32)
        
        # Generate waypoint
        min_distance = 500
        max_distance = self.WAYPOINT_RANGE * 0.8  # Slightly reduced to avoid boundary issues
        
        while True:
            angle = self.np_random.uniform(0, 2 * np.pi)
            distance = self.np_random.uniform(min_distance, max_distance)
            
            waypoint = np.array([
                np.cos(angle) * distance,
                np.sin(angle) * distance
            ])
            
            if min_distance <= np.linalg.norm(waypoint) <= max_distance:
                self.current_waypoint = waypoint
                break
        
        # Initialize distances
        self.distance_to_waypoint = np.linalg.norm(self.current_waypoint - self.state[:2])
        self.previous_distance_to_waypoint = self.distance_to_waypoint
        self._update_boundary_distances()
        
        # Generate obstacles
        self.obstacles = []
        for _ in range(5):
            while True:
                position = np.array([
                    self.np_random.uniform(-800, 800),  # Reduced range to avoid boundaries
                    self.np_random.uniform(-800, 800),
                    self.ALTITUDE
                ])
                
                dist_to_start = np.linalg.norm(position[:2])
                dist_to_waypoint = np.linalg.norm(position[:2] - self.current_waypoint)
                
                if dist_to_start > 200 and dist_to_waypoint > 200:
                    self.obstacles.append({
                        'position': position,
                        'size': self.np_random.uniform(100, 200)
                    })
                    break
        
        if self.render_mode == "human":
            self._init_renderer()
        
        return self.state.copy(), {}

    def step(self, action):
        """Execute environment step"""
        if self.mode == "manual":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.recording:
                        self.save_demonstration()
                    self.close()
                    return self.state, 0.0, True, False, {}
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        if not self.recording:
                            print("Started recording demonstration")
                            self.recording = True
                            self.demonstration_data = []
                        else:
                            self.save_demonstration()
                            print("Stopped recording demonstration")
                    elif event.key == pygame.K_ESCAPE:
                        if self.recording:
                            self.save_demonstration()
                        return self.state, 0.0, True, False, {}
            
            action = self.handle_manual_control()
            if self.recording:
                self.demonstration_data.append((self.state.copy(), action))
        elif self.mode == "bc":
            action = np.array(action).flatten()
        
        self.metrics['episode_length'] += 1
        
        # Store previous state
        previous_position = self.state[:2].copy()
        self.previous_distance_to_waypoint = np.linalg.norm(self.current_waypoint - previous_position)
        
        # Update state
        dt = 0.1
        self.state[2] = float((self.state[2] + action[0]) % (2 * np.pi))
        
        velocity = np.array([
            np.cos(self.state[2]),
            np.sin(self.state[2])
        ]) * self.CONSTANT_SPEED * dt
        
        self.state[0:2] += velocity
        self.state[3:3+len(self.SENSOR_ANGLES)] = self.get_sensor_readings()
        self._update_boundary_distances()
        self.state[3+len(self.SENSOR_ANGLES):] = self.boundary_distances
        
        # Update distances and check conditions
        self.distance_to_waypoint = np.linalg.norm(self.current_waypoint - self.state[:2])
        collision = self._check_collision()
        
        if collision:
            self.metrics['collision_count'] += 1
        
        waypoint_reached = self.distance_to_waypoint < self.WAYPOINT_RADIUS
        if waypoint_reached:
            self.metrics['success_count'] += 1
            reward = 1000.0
            print("Waypoint reached! Success!")
        else:
            reward = self._calculate_reward(collision)
        
        timeout = self.metrics['episode_length'] >= 500
        done = collision or waypoint_reached or timeout
        
        if timeout:
            reward = -100.0
            print("Episode timeout")
        
        self.metrics['total_reward'] += reward
        
        if self.render_mode == "human":
            self.render()
        
        return self.state.copy(), reward, done, False, self.metrics

    def handle_manual_control(self):
        """Handle keyboard inputs"""
        keys = pygame.key.get_pressed()
        yaw_rate = 0.0
        
        if keys[pygame.K_LEFT]:
            yaw_rate = 0.1
        if keys[pygame.K_RIGHT]:
            yaw_rate = -0.1
            
        return np.array([yaw_rate])

    def save_demonstration(self):
        """Save recorded demonstration"""
        if len(self.demonstration_data) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expert_demo_{timestamp}.pkl"
            
            observations = np.array([data[0] for data in self.demonstration_data])
            actions = np.array([data[1] for data in self.demonstration_data])
            
            demonstration = {
                'observations': observations,
                'actions': actions
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(demonstration, f)
            
            print(f"Demonstration saved to {filename}")
            print(f"Recorded {len(self.demonstration_data)} timesteps")
            self.demonstration_data = []
            self.recording = False

    def _init_renderer(self):
        """Initialize pygame renderer"""
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Aircraft Navigation - Press R to record, ESC to reset")
        self.clock = pygame.time.Clock()

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.screen is None:
                self._init_renderer()
            
            self.screen.fill((200, 200, 200))
            
            def world_to_screen(pos):
                screen_pos = self.camera_offset + pos * self.zoom
                return int(screen_pos[0]), int(screen_pos[1])
            
            # Draw boundary box
            boundary = self.WAYPOINT_RANGE
            corners = [
                (-boundary, -boundary),
                (boundary, -boundary),
                (boundary, boundary),
                (-boundary, boundary)
            ]
            screen_corners = [world_to_screen(np.array(corner)) for corner in corners]
            pygame.draw.lines(self.screen, (255, 0, 0), True, screen_corners, 2)
            
            # Draw boundary distances
            aircraft_pos_screen = world_to_screen(self.state[:2])
            for i, (dx, dy) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):  # N, E, S, W
                end_pos = self.state[:2] + np.array([dx, dy]) * self.boundary_distances[i]
                end_pos_screen = world_to_screen(end_pos)
                color = (100, 100, 255) if self.boundary_distances[i] < 200 else (180, 180, 255)
                pygame.draw.line(self.screen, color, aircraft_pos_screen, end_pos_screen, 1)
            
            # Draw grid
            grid_size = 100
            grid_range = 2000
            for x in range(-grid_range, grid_range + 1, grid_size):
                start = world_to_screen(np.array([x, -grid_range]))
                end = world_to_screen(np.array([x, grid_range]))
                pygame.draw.line(self.screen, (220, 220, 220), start, end)
            for y in range(-grid_range, grid_range + 1, grid_size):
                start = world_to_screen(np.array([-grid_range, y]))
                end = world_to_screen(np.array([grid_range, y]))
                pygame.draw.line(self.screen, (220, 220, 220), start, end)
            
            # Draw starting point
            start_pos = world_to_screen(np.array([0, 0]))
            pygame.draw.circle(self.screen, (0, 255, 0), start_pos, 10)
            
            # Draw obstacles
            for obstacle in self.obstacles:
                pos = world_to_screen(obstacle['position'][:2])
                size = int(obstacle['size'] * self.zoom)
                rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
                pygame.draw.rect(self.screen, (128, 128, 128), rect)
            
            # Draw waypoint with pulsing effect
            waypoint_pos = world_to_screen(self.current_waypoint)
            waypoint_radius = int(self.WAYPOINT_RADIUS * self.zoom)
            pulse = (np.sin(pygame.time.get_ticks() * 0.005) + 1) * 0.5
            waypoint_color = (255, int(185 + 70 * pulse), 0)
            pygame.draw.circle(self.screen, waypoint_color, waypoint_pos, waypoint_radius)
            
            # Draw aircraft
            aircraft_pos = world_to_screen(self.state[:2])
            heading = self.state[2]
            
            # Draw path to waypoint
            pygame.draw.line(self.screen, (100, 100, 100), aircraft_pos, waypoint_pos, 1)
            
            # Draw aircraft triangle
            triangle_size = 15
            points = []
            points.append((
                aircraft_pos[0] + int(triangle_size * np.cos(heading)),
                aircraft_pos[1] + int(triangle_size * np.sin(heading))
            ))
            wing_angle = 2.6
            points.append((
                aircraft_pos[0] + int(triangle_size * np.cos(heading + wing_angle)),
                aircraft_pos[1] + int(triangle_size * np.sin(heading + wing_angle))
            ))
            points.append((
                aircraft_pos[0] + int(triangle_size * np.cos(heading - wing_angle)),
                aircraft_pos[1] + int(triangle_size * np.sin(heading - wing_angle))
            ))
            pygame.draw.polygon(self.screen, (255, 0, 0), points)
            
            # Draw sensor rays
            for angle, reading in zip(self.SENSOR_ANGLES, self.state[3:3+len(self.SENSOR_ANGLES)]):
                abs_angle = self.state[2] + np.deg2rad(angle)
                direction = np.array([np.cos(abs_angle), np.sin(abs_angle)])
                end_pos = self.state[:2] + direction * reading
                ray_end = world_to_screen(end_pos)
                
                color = (255, 255, 0) if reading < self.SENSOR_RANGE else (200, 200, 0)
                pygame.draw.line(self.screen, color, aircraft_pos, ray_end, 2)
                
                if reading < self.SENSOR_RANGE:
                    pygame.draw.circle(self.screen, (255, 0, 0), ray_end, 3)

            # Draw info panel
            self._draw_info_panel()
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _draw_info_panel(self):
        """Draw information panel"""
        info_surface = pygame.Surface((200, 160))  # Reduced height since we removed boundary info
        info_surface.fill((240, 240, 240))
        info_surface.set_alpha(230)
        self.screen.blit(info_surface, (10, 10))

        font = pygame.font.Font(None, 24)
        y_offset = 20
        line_spacing = 25

        def draw_text(text, y_pos, color=(0, 0, 0)):
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (20, y_pos))

        # Essential information
        if self.recording:
            draw_text("Recording...", y_offset, (255, 0, 0))
            y_offset += line_spacing

        mode_text = "Manual" if self.mode == "manual" else "BC"
        draw_text(f"Mode: {mode_text}", y_offset)
        y_offset += line_spacing
        
        draw_text(f"Speed: {self.CONSTANT_SPEED:.1f} m/s", y_offset)
        y_offset += line_spacing
        
        draw_text(f"Goal dist: {self.distance_to_waypoint:.1f}m", y_offset)
        y_offset += line_spacing
        
        draw_text(f"Reward: {self.metrics['total_reward']:.1f}", y_offset)
        
        # Controls at bottom
        if self.mode == "manual":
            controls_surface = pygame.Surface((200, 80))
            controls_surface.fill((240, 240, 240))
            controls_surface.set_alpha(230)
            self.screen.blit(controls_surface, (10, 490))
            
            y_offset = 500
            controls = [
                "←/→: Turn",
                "R: Record",
                "ESC: Reset"
            ]
            
            for control in controls:
                text_surface = font.render(control, True, (100, 100, 100))
                self.screen.blit(text_surface, (20, y_offset))
                y_offset += 20

    def close(self):
        """Close the environment"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    env = AircraftEnv(render_mode="human", mode="manual")
    try:
        obs, info = env.reset()
        while True:
            obs, reward, terminated, truncated, info = env.step(np.zeros(1))
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nExiting...")
        env.close()
    finally:
        env.close()