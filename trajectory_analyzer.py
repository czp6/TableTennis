"""
Trajectory analysis module for table tennis ball tracking.
Calculates position, velocity, acceleration, and 3D trajectory estimation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass
class TrajectoryPoint:
    """Container for trajectory data at a single point in time."""
    frame_number: int
    time: float  # seconds
    position_2d: Tuple[float, float]  # (x, y) in pixels
    position_3d: Optional[Tuple[float, float, float]] = None  # (x, y, z) in real units
    velocity_2d: Optional[Tuple[float, float]] = None  # pixels/second
    velocity_3d: Optional[Tuple[float, float, float]] = None  # units/second
    acceleration_2d: Optional[Tuple[float, float]] = None  # pixels/second²
    acceleration_3d: Optional[Tuple[float, float, float]] = None  # units/second²
    speed: float = 0.0  # magnitude of velocity


class TrajectoryAnalyzer:
    """Analyzes ball trajectory from position data."""

    def __init__(self, fps: float = 240.0, ball_diameter_mm: float = 40.0):
        """
        Initialize trajectory analyzer.

        Args:
            fps: Frames per second of the video
            ball_diameter_mm: Real diameter of table tennis ball in millimeters (standard is 40mm)
        """
        self.fps = fps
        self.ball_diameter_mm = ball_diameter_mm
        self.trajectory_points: List[TrajectoryPoint] = []
        self.pixels_per_mm: Optional[float] = None
        self.smoothing_window = 5  # Frames to smooth for velocity/acceleration

    def calibrate_scale(self, ball_radius_pixels: float):
        """
        Calibrate the pixel-to-millimeter scale using detected ball size.

        Args:
            ball_radius_pixels: Detected ball radius in pixels
        """
        ball_diameter_pixels = ball_radius_pixels * 2
        self.pixels_per_mm = ball_diameter_pixels / self.ball_diameter_mm

    def add_position(self, frame_number: int, x: float, y: float, ball_radius: Optional[float] = None):
        """
        Add a position measurement.

        Args:
            frame_number: Frame number
            x: X coordinate in pixels
            y: Y coordinate in pixels
            ball_radius: Ball radius in pixels (for calibration)
        """
        time = frame_number / self.fps

        # Calibrate scale if we have ball radius
        if ball_radius and self.pixels_per_mm is None:
            self.calibrate_scale(ball_radius)

        point = TrajectoryPoint(
            frame_number=frame_number,
            time=time,
            position_2d=(x, y)
        )

        # Initialize spin attributes
        point.rps = 0.0
        point.topspin_rps = 0.0
        point.sidespin_rps = 0.0
        point.direction = "none"

        self.trajectory_points.append(point)
        return point

    def calculate_derivatives(self):
        """Calculate velocity and acceleration for all trajectory points with improved smoothing."""
        if len(self.trajectory_points) < 2:
            return

        # Smooth positions with larger window
        smoothed_positions = self._smooth_positions()

        # Calculate velocities
        for i in range(len(self.trajectory_points)):
            if i == 0:
                # Forward difference for first point
                if len(self.trajectory_points) > 1:
                    dt = self.trajectory_points[1].time - self.trajectory_points[0].time
                    dx = smoothed_positions[1][0] - smoothed_positions[0][0]
                    dy = smoothed_positions[1][1] - smoothed_positions[0][1]
                    vx = dx / dt if dt > 0 else 0
                    vy = dy / dt if dt > 0 else 0
                else:
                    vx, vy = 0, 0
            elif i == len(self.trajectory_points) - 1:
                # Backward difference for last point
                dt = self.trajectory_points[i].time - self.trajectory_points[i-1].time
                dx = smoothed_positions[i][0] - smoothed_positions[i-1][0]
                dy = smoothed_positions[i][1] - smoothed_positions[i-1][1]
                vx = dx / dt if dt > 0 else 0
                vy = dy / dt if dt > 0 else 0
            else:
                # Central difference for middle points
                dt = self.trajectory_points[i+1].time - self.trajectory_points[i-1].time
                dx = smoothed_positions[i+1][0] - smoothed_positions[i-1][0]
                dy = smoothed_positions[i+1][1] - smoothed_positions[i-1][1]
                vx = dx / dt if dt > 0 else 0
                vy = dy / dt if dt > 0 else 0

            self.trajectory_points[i].velocity_2d = (vx, vy)
            self.trajectory_points[i].speed = np.sqrt(vx**2 + vy**2)

            # Convert to 3D if calibrated (m/s units)
            if self.pixels_per_mm:
                vx_m_s = vx / self.pixels_per_mm / 1000
                vy_m_s = vy / self.pixels_per_mm / 1000
                self.trajectory_points[i].velocity_3d = (vx_m_s, vy_m_s, 0)  # Assuming z velocity is 0 for perpendicular view

        # Calculate accelerations with smoothing
        velocity_history = [p.velocity_2d for p in self.trajectory_points if p.velocity_2d]

        if len(velocity_history) > self.smoothing_window:
            # Smooth velocity for better acceleration calculation
            smoothed_velocity = []
            half_window = self.smoothing_window // 2

            for i in range(len(velocity_history)):
                start = max(0, i - half_window)
                end = min(len(velocity_history), i + half_window + 1)

                vx_sum = sum(velocity_history[j][0] for j in range(start, end))
                vy_sum = sum(velocity_history[j][1] for j in range(start, end))
                count = end - start

                smoothed_velocity.append((vx_sum / count, vy_sum / count))
        else:
            smoothed_velocity = velocity_history

        # Calculate acceleration from smoothed velocity
        for i in range(len(self.trajectory_points)):
            if i == 0 or i == len(self.trajectory_points) - 1:
                self.trajectory_points[i].acceleration_2d = (0, 0)
                if self.pixels_per_mm:
                    self.trajectory_points[i].acceleration_3d = (0, 0, 0)
            else:
                dt = self.trajectory_points[i+1].time - self.trajectory_points[i-1].time

                if dt > 0:
                    # Use central difference on smoothed velocities
                    v1 = smoothed_velocity[i-1]
                    v2 = smoothed_velocity[i+1]

                    ax = (v2[0] - v1[0]) / dt
                    ay = (v2[1] - v1[1]) / dt
                    self.trajectory_points[i].acceleration_2d = (ax, ay)

                    if self.pixels_per_mm:
                        ax_m_s2 = (ax / self.pixels_per_mm / 1000)
                        ay_m_s2 = (ay / self.pixels_per_mm / 1000)
                        self.trajectory_points[i].acceleration_3d = (ax_m_s2, ay_m_s2, 0)

    def _smooth_positions(self) -> List[Tuple[float, float]]:
        """Smooth position data using moving average."""
        if len(self.trajectory_points) < self.smoothing_window:
            return [(p.position_2d[0], p.position_2d[1]) for p in self.trajectory_points]

        smoothed = []
        half_window = self.smoothing_window // 2

        for i in range(len(self.trajectory_points)):
            start = max(0, i - half_window)
            end = min(len(self.trajectory_points), i + half_window + 1)

            x_sum = sum(self.trajectory_points[j].position_2d[0] for j in range(start, end))
            y_sum = sum(self.trajectory_points[j].position_2d[1] for j in range(start, end))
            count = end - start

            smoothed.append((x_sum / count, y_sum / count))

        return smoothed

    def estimate_3d_trajectory(self):
        """
        Estimate 3D trajectory assuming camera is perpendicular to ball path.
        Uses ball size changes to estimate depth (z-axis).
        """
        # This is a simplified estimation
        # In reality, you'd need proper camera calibration
        pass

    def calculate_launch_kinematics(self, launch_frame: int = 0):
        """
        Calculate physics properties at the launch (paddle contact) point.

        Args:
            launch_frame: Frame number where launch occurs

        Returns:
            Dictionary with physics properties
        """
        if launch_frame < 0 or launch_frame >= len(self.trajectory_points):
            return None

        launch_point = self.trajectory_points[launch_frame]

        physics = {
            'frame_number': launch_point.frame_number,
            'time': launch_point.time
        }

        # Ball properties (standard table tennis)
        mass = 0.0027  # kg
        radius = 0.020  # meters (40mm diameter)
        moment_of_inertia = 0.4 * mass * radius**2  # Approximate for hollow sphere

        # Linear velocity and momentum - converted to m/s
        if launch_point.velocity_3d:
            vx, vy, vz = launch_point.velocity_3d
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            momentum = mass * speed
            kinetic_energy = 0.5 * mass * speed**2
            physics.update({
                'velocity_m_s': speed,
                'velocity_components': (vx, vy, vz),
                'momentum_kg_m_s': momentum,
                'kinetic_energy_j': kinetic_energy
            })

        # Angular velocity (from spin rate) - assuming we have spin data linked to trajectory
        # This requires integration with spin tracking
        physics['angular_velocity_rad_s'] = None
        physics['rotational_energy_j'] = None
        physics['total_mechanical_energy_j'] = kinetic_energy if 'kinetic_energy_j' in physics else None

        # Trajectory angle (from horizontal)
        if launch_point.velocity_2d:
            vx, vy = launch_point.velocity_2d
            angle_deg = np.degrees(np.arctan2(vy, vx))
            physics['launch_angle_deg'] = angle_deg

        # Acceleration at launch (force estimation) - converted to m/s²
        if launch_point.acceleration_3d:
            ax, ay, az = launch_point.acceleration_3d
            accel_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
            force = mass * accel_magnitude
            physics.update({
                'acceleration_m_s2': accel_magnitude,
                'force_n': force
            })

        return physics

    def find_launch_point(self, method: str = 'auto') -> int:
        """
        Find the likely launch (paddle contact) point in the trajectory.

        Args:
            method: 'auto' or 'manual'

        Returns:
            Frame number of estimated launch point
        """
        if method == 'auto' and len(self.trajectory_points) > 0:
            # Analyze trajectory points to find launch
            if len(self.trajectory_points) < 5:
                return 0

            # First, check if we have RPS values in spin data (which indicate rotation)
            # For now, use speed and acceleration analysis
            speeds = []
            accelerations = []
            frames = []

            for i, point in enumerate(self.trajectory_points):
                if point.speed > 0 and hasattr(point, 'rps'):
                    frames.append(i)
                    speeds.append(point.speed)

            if len(frames) > 0:
                # Find where speed increases rapidly - this could be launch
                speed_diffs = []
                for i in range(1, len(frames)):
                    diff = speeds[i] - speeds[i-1]
                    speed_diffs.append((frames[i], diff))

                if speed_diffs:
                    max_diff_frame, max_diff = max(speed_diffs, key=lambda x: x[1])
                    if max_diff > 0:
                        print(f"Launch likely at frame {max_diff_frame} (speed increase: {max_diff:.2f} px/s)")
                        return max_diff_frame

            # Fallback: Use simple heuristic for first valid trajectory point
            for i, point in enumerate(self.trajectory_points):
                if hasattr(point, 'rps') and point.rps > 0.1:  # Ball is rotating
                    print(f"Launch at first rotating frame: {i}")
                    return i

            return 0
        return 0

    def get_statistics(self) -> dict:
        """Get trajectory statistics in real-world units."""
        if not self.trajectory_points:
            return {}

        # Calculate based on real-world units if available
        if self.pixels_per_mm:
            stats = self._get_real_world_statistics()
        else:
            stats = self._get_pixel_statistics()

        return stats

    def _get_real_world_statistics(self) -> dict:
        """Get statistics in real-world units (m, m/s, m/s²)."""
        stats = {
            'total_points': len(self.trajectory_points),
            'duration': self.trajectory_points[-1].time if self.trajectory_points else 0,
        }

        # Speed in m/s
        speeds = []
        for p in self.trajectory_points:
            if p.speed > 0 and self.pixels_per_mm:
                speeds.append(p.speed / self.pixels_per_mm / 1000)

        if speeds:
            stats['avg_speed_m_s'] = sum(speeds) / len(speeds)
            stats['max_speed_m_s'] = max(speeds)
            stats['min_speed_m_s'] = min(speeds)

        # Acceleration in m/s²
        accelerations = []
        for p in self.trajectory_points:
            if p.acceleration_2d:
                accel_magnitude = np.sqrt(p.acceleration_2d[0]**2 + p.acceleration_2d[1]**2)
                if self.pixels_per_mm:
                    accelerations.append(accel_magnitude / self.pixels_per_mm / 1000)

        if accelerations:
            stats['max_acceleration_m_s2'] = max(accelerations)
            stats['avg_acceleration_m_s2'] = sum(accelerations) / len(accelerations)

        # Calculate total distance traveled in meters
        if len(self.trajectory_points) > 1:
            distance_m = 0.0
            for i in range(1, len(self.trajectory_points)):
                dx_px = self.trajectory_points[i].position_2d[0] - self.trajectory_points[i-1].position_2d[0]
                dy_px = self.trajectory_points[i].position_2d[1] - self.trajectory_points[i-1].position_2d[1]
                distance_px = np.sqrt(dx_px**2 + dy_px**2)
                distance_m += distance_px / self.pixels_per_mm / 1000

            stats['total_distance_m'] = distance_m

        return stats

    def _get_pixel_statistics(self) -> dict:
        """Get statistics in pixel units (fallback if calibration fails)."""
        stats = {
            'total_points': len(self.trajectory_points),
            'duration': self.trajectory_points[-1].time if self.trajectory_points else 0,
        }

        speeds = [p.speed for p in self.trajectory_points if p.speed > 0]

        if speeds:
            stats['avg_speed_px_s'] = sum(speeds) / len(speeds)
            stats['max_speed_px_s'] = max(speeds)
            stats['min_speed_px_s'] = min(speeds)

        accelerations = [
            np.sqrt(p.acceleration_2d[0]**2 + p.acceleration_2d[1]**2)
            for p in self.trajectory_points
            if p.acceleration_2d
        ]

        if accelerations:
            stats['max_acceleration_px_s2'] = max(accelerations)
            stats['avg_acceleration_px_s2'] = sum(accelerations) / len(accelerations)

        return stats

    def get_trajectory_path(self) -> List[Tuple[int, int]]:
        """Get trajectory as list of (x, y) points for drawing."""
        return [(int(p.position_2d[0]), int(p.position_2d[1])) for p in self.trajectory_points]
