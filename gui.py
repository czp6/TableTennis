#!/usr/bin/env python3
"""
GUI interface for table tennis ball spin rate analysis with real-time visualization.
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from pathlib import Path
from spin_tracker import SpinTracker, SpinData
from trajectory_analyzer import TrajectoryAnalyzer


class SpinTrackerGUI:
    """GUI application for spin rate analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("Table Tennis Ball Analyzer")
        self.root.geometry("1200x800")

        self.tracker = None
        self.trajectory_analyzer = None
        self.video_path = None
        self.is_playing = False
        self.current_frame_idx = 0
        self.processed_data = []
        self.analysis_mode = "spin"  # "spin" or "trajectory"
        self.needs_rotation = True  # Rotate 90 degrees clockwise
        self.zoom_enabled = False  # Zoom to ball region
        self.zoom_margin = 3.0  # Multiplier for ball radius to determine crop size

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # File selection
        ttk.Button(
            control_frame,
            text="Load Video",
            command=self.load_video
        ).pack(side=tk.LEFT, padx=5)

        # FPS input
        ttk.Label(control_frame, text="FPS:").pack(side=tk.LEFT, padx=5)
        self.fps_var = tk.StringVar(value="240")
        ttk.Entry(
            control_frame,
            textvariable=self.fps_var,
            width=8
        ).pack(side=tk.LEFT, padx=5)

        # Mode selector
        ttk.Label(control_frame, text="Mode:").pack(side=tk.LEFT, padx=(20, 5))
        self.mode_var = tk.StringVar(value="spin")
        mode_combo = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            values=["spin", "trajectory"],
            state="readonly",
            width=12
        )
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Zoom toggle
        self.zoom_var = tk.BooleanVar(value=False)
        zoom_check = ttk.Checkbutton(
            control_frame,
            text="Zoom to Ball",
            variable=self.zoom_var,
            command=self.on_zoom_toggle
        )
        zoom_check.pack(side=tk.LEFT, padx=5)

        # Rotation toggle
        self.rotation_var = tk.BooleanVar(value=True)
        rotation_check = ttk.Checkbutton(
            control_frame,
            text="Rotate 90°",
            variable=self.rotation_var
        )
        rotation_check.pack(side=tk.LEFT, padx=5)

        # Process button
        self.process_btn = ttk.Button(
            control_frame,
            text="Process Video",
            command=self.process_video,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)

        # Export button
        self.export_btn = ttk.Button(
            control_frame,
            text="Export CSV",
            command=self.export_csv,
            state=tk.DISABLED
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Frame labeler button
        ttk.Button(
            control_frame,
            text="Label Frames",
            command=self.launch_frame_labeler
        ).pack(side=tk.LEFT, padx=5)

        # Main content area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Video display
        video_frame = ttk.LabelFrame(content_frame, text="Video", padding="10")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame, text="No video loaded", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Video controls
        video_controls = ttk.Frame(video_frame)
        video_controls.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.play_btn = ttk.Button(
            video_controls,
            text="Play",
            command=self.toggle_play,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.frame_slider = ttk.Scale(
            video_controls,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.on_slider_change
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_slider.state(['disabled'])

        self.frame_label = ttk.Label(video_controls, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=5)

        # Right side - Statistics and info
        info_frame = ttk.LabelFrame(content_frame, text="Analysis Results", padding="10")
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        info_frame.config(width=300)

        # Create notebook for different analysis modes
        self.notebook = ttk.Notebook(info_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Spin analysis tab
        spin_tab = ttk.Frame(self.notebook)
        self.notebook.add(spin_tab, text="Spin Analysis")

        # Current frame info (spin)
        current_frame = ttk.LabelFrame(spin_tab, text="Current Frame", padding="10")
        current_frame.pack(fill=tk.X, pady=5)

        self.current_rps_label = ttk.Label(current_frame, text="RPS: --", font=("Arial", 14, "bold"))
        self.current_rps_label.pack()

        self.current_rpm_label = ttk.Label(current_frame, text="RPM: --")
        self.current_rpm_label.pack()

        self.current_angle_label = ttk.Label(current_frame, text="Angle: --°")
        self.current_angle_label.pack()

        # Overall statistics (spin)
        stats_frame = ttk.LabelFrame(spin_tab, text="Overall Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)

        self.avg_rps_label = ttk.Label(stats_frame, text="Avg RPS: --")
        self.avg_rps_label.pack(anchor=tk.W)

        self.max_rps_label = ttk.Label(stats_frame, text="Max RPS: --")
        self.max_rps_label.pack(anchor=tk.W)

        self.min_rps_label = ttk.Label(stats_frame, text="Min RPS: --")
        self.min_rps_label.pack(anchor=tk.W)

        self.frames_analyzed_label = ttk.Label(stats_frame, text="Frames: --")
        self.frames_analyzed_label.pack(anchor=tk.W)

        # Trajectory analysis tab
        trajectory_tab = ttk.Frame(self.notebook)
        self.notebook.add(trajectory_tab, text="Trajectory Analysis")

        # Detection parameters tab with scrollbar
        params_tab = ttk.Frame(self.notebook)
        self.notebook.add(params_tab, text="Detection Params")

        # Create canvas and scrollbar for scrolling
        params_canvas = tk.Canvas(params_tab, highlightthickness=0)
        params_scrollbar = ttk.Scrollbar(params_tab, orient="vertical", command=params_canvas.yview)
        scrollable_params_frame = ttk.Frame(params_canvas)

        scrollable_params_frame.bind(
            "<Configure>",
            lambda e: params_canvas.configure(scrollregion=params_canvas.bbox("all"))
        )

        params_canvas.create_window((0, 0), window=scrollable_params_frame, anchor="nw")
        params_canvas.configure(yscrollcommand=params_scrollbar.set)

        params_canvas.pack(side="left", fill="both", expand=True)
        params_scrollbar.pack(side="right", fill="y")

        # Enable trackpad and mousewheel scrolling (cross-platform)
        def scroll_canvas(event):
            # macOS trackpad sends delta values directly
            if event.num == 4 or event.delta > 0:
                params_canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                params_canvas.yview_scroll(1, "units")
            return "break"

        # Bind scrolling events for all platforms
        params_canvas.bind_all("<MouseWheel>", scroll_canvas)  # Windows/macOS
        params_canvas.bind_all("<Button-4>", scroll_canvas)    # Linux scroll up
        params_canvas.bind_all("<Button-5>", scroll_canvas)    # Linux scroll down

        # Detection method weights
        weights_frame = ttk.LabelFrame(scrollable_params_frame, text="Detection Weights", padding="10")
        weights_frame.pack(fill=tk.X, pady=5)

        ttk.Label(weights_frame, text="YOLOv8 Weight:").pack(anchor=tk.W)
        self.yolo_weight_var = tk.DoubleVar(value=1.0)
        yolo_weight_slider = ttk.Scale(
            weights_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.yolo_weight_var,
            command=self.on_param_change
        )
        yolo_weight_slider.pack(fill=tk.X, pady=2)
        self.yolo_weight_label = ttk.Label(weights_frame, text="1.00")
        self.yolo_weight_label.pack(anchor=tk.W)

        ttk.Label(weights_frame, text="Hough Weight:").pack(anchor=tk.W, pady=(10, 0))
        self.hough_weight_var = tk.DoubleVar(value=0.0)
        hough_weight_slider = ttk.Scale(
            weights_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.hough_weight_var,
            command=self.on_param_change
        )
        hough_weight_slider.pack(fill=tk.X, pady=2)
        self.hough_weight_label = ttk.Label(weights_frame, text="0.00")
        self.hough_weight_label.pack(anchor=tk.W)

        ttk.Label(weights_frame, text="Hybrid Weight:").pack(anchor=tk.W, pady=(10, 0))
        self.hybrid_weight_var = tk.DoubleVar(value=0.0)
        hybrid_weight_slider = ttk.Scale(
            weights_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.hybrid_weight_var,
            command=self.on_param_change
        )
        hybrid_weight_slider.pack(fill=tk.X, pady=2)
        self.hybrid_weight_label = ttk.Label(weights_frame, text="0.00")
        self.hybrid_weight_label.pack(anchor=tk.W)

        # Segmentation display toggle
        self.show_segmentation_var = tk.BooleanVar(value=True)
        seg_check = ttk.Checkbutton(
            weights_frame,
            text="Show Segmentation Mask",
            variable=self.show_segmentation_var
        )
        seg_check.pack(anchor=tk.W, pady=(10, 0))

        # Debug mode toggle
        self.debug_mode_var = tk.BooleanVar(value=False)
        debug_check = ttk.Checkbutton(
            weights_frame,
            text="Debug Mode (print mask info)",
            variable=self.debug_mode_var
        )
        debug_check.pack(anchor=tk.W, pady=(5, 0))

        # YOLOv8 parameters
        yolo_frame = ttk.LabelFrame(scrollable_params_frame, text="YOLOv8 Parameters", padding="10")
        yolo_frame.pack(fill=tk.X, pady=5)

        ttk.Label(yolo_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.yolo_conf_var = tk.DoubleVar(value=0.25)
        yolo_conf_slider = ttk.Scale(
            yolo_frame,
            from_=0.01,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.yolo_conf_var,
            command=self.on_param_change
        )
        yolo_conf_slider.pack(fill=tk.X, pady=2)
        self.yolo_conf_label = ttk.Label(yolo_frame, text="0.25")
        self.yolo_conf_label.pack(anchor=tk.W)

        ttk.Label(yolo_frame, text="Min Mask Area (pixels):").pack(anchor=tk.W, pady=(10, 0))
        self.yolo_min_mask_area_var = tk.IntVar(value=100)
        yolo_min_mask_area_slider = ttk.Scale(
            yolo_frame,
            from_=10,
            to=1000,
            orient=tk.HORIZONTAL,
            variable=self.yolo_min_mask_area_var,
            command=self.on_param_change
        )
        yolo_min_mask_area_slider.pack(fill=tk.X, pady=2)
        self.yolo_min_mask_area_label = ttk.Label(yolo_frame, text="100")
        self.yolo_min_mask_area_label.pack(anchor=tk.W)

        ttk.Label(yolo_frame, text="Circle Radius Scale:").pack(anchor=tk.W, pady=(10, 0))
        self.yolo_radius_scale_var = tk.DoubleVar(value=0.8)
        yolo_radius_scale_slider = ttk.Scale(
            yolo_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.yolo_radius_scale_var,
            command=self.on_param_change
        )
        yolo_radius_scale_slider.pack(fill=tk.X, pady=2)
        self.yolo_radius_scale_label = ttk.Label(yolo_frame, text="0.80")
        self.yolo_radius_scale_label.pack(anchor=tk.W)

        # Hough Circle parameters
        hough_frame = ttk.LabelFrame(scrollable_params_frame, text="Hough Circle Parameters", padding="10")
        hough_frame.pack(fill=tk.X, pady=5)

        ttk.Label(hough_frame, text="Param1 (Edge threshold):").pack(anchor=tk.W)
        self.hough_param1_var = tk.IntVar(value=50)
        hough_param1_slider = ttk.Scale(
            hough_frame,
            from_=10,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.hough_param1_var,
            command=self.on_param_change
        )
        hough_param1_slider.pack(fill=tk.X, pady=2)
        self.hough_param1_label = ttk.Label(hough_frame, text="50")
        self.hough_param1_label.pack(anchor=tk.W)

        ttk.Label(hough_frame, text="Param2 (Accumulator threshold):").pack(anchor=tk.W, pady=(10, 0))
        self.hough_param2_var = tk.IntVar(value=30)
        hough_param2_slider = ttk.Scale(
            hough_frame,
            from_=10,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.hough_param2_var,
            command=self.on_param_change
        )
        hough_param2_slider.pack(fill=tk.X, pady=2)
        self.hough_param2_label = ttk.Label(hough_frame, text="30")
        self.hough_param2_label.pack(anchor=tk.W)

        ttk.Label(hough_frame, text="Min Radius:").pack(anchor=tk.W, pady=(10, 0))
        self.hough_min_radius_var = tk.IntVar(value=10)
        hough_min_radius_slider = ttk.Scale(
            hough_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            variable=self.hough_min_radius_var,
            command=self.on_param_change
        )
        hough_min_radius_slider.pack(fill=tk.X, pady=2)
        self.hough_min_radius_label = ttk.Label(hough_frame, text="10")
        self.hough_min_radius_label.pack(anchor=tk.W)

        ttk.Label(hough_frame, text="Max Radius:").pack(anchor=tk.W, pady=(10, 0))
        self.hough_max_radius_var = tk.IntVar(value=150)
        hough_max_radius_slider = ttk.Scale(
            hough_frame,
            from_=50,
            to=300,
            orient=tk.HORIZONTAL,
            variable=self.hough_max_radius_var,
            command=self.on_param_change
        )
        hough_max_radius_slider.pack(fill=tk.X, pady=2)
        self.hough_max_radius_label = ttk.Label(hough_frame, text="150")
        self.hough_max_radius_label.pack(anchor=tk.W)

        # Hybrid Detector parameters
        hybrid_frame = ttk.LabelFrame(scrollable_params_frame, text="Hybrid Detector Parameters", padding="10")
        hybrid_frame.pack(fill=tk.X, pady=5)

        ttk.Label(hybrid_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.hybrid_conf_var = tk.DoubleVar(value=0.6)
        hybrid_conf_slider = ttk.Scale(
            hybrid_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.hybrid_conf_var,
            command=self.on_param_change
        )
        hybrid_conf_slider.pack(fill=tk.X, pady=2)
        self.hybrid_conf_label = ttk.Label(hybrid_frame, text="0.60")
        self.hybrid_conf_label.pack(anchor=tk.W)

        ttk.Label(hybrid_frame, text="Orange H Min:").pack(anchor=tk.W, pady=(10, 0))
        self.hybrid_orange_h_min_var = tk.IntVar(value=5)
        hybrid_orange_h_min_slider = ttk.Scale(
            hybrid_frame,
            from_=0,
            to=180,
            orient=tk.HORIZONTAL,
            variable=self.hybrid_orange_h_min_var,
            command=self.on_param_change
        )
        hybrid_orange_h_min_slider.pack(fill=tk.X, pady=2)
        self.hybrid_orange_h_min_label = ttk.Label(hybrid_frame, text="5")
        self.hybrid_orange_h_min_label.pack(anchor=tk.W)

        ttk.Label(hybrid_frame, text="Orange H Max:").pack(anchor=tk.W, pady=(10, 0))
        self.hybrid_orange_h_max_var = tk.IntVar(value=25)
        hybrid_orange_h_max_slider = ttk.Scale(
            hybrid_frame,
            from_=0,
            to=180,
            orient=tk.HORIZONTAL,
            variable=self.hybrid_orange_h_max_var,
            command=self.on_param_change
        )
        hybrid_orange_h_max_slider.pack(fill=tk.X, pady=2)
        self.hybrid_orange_h_max_label = ttk.Label(hybrid_frame, text="25")
        self.hybrid_orange_h_max_label.pack(anchor=tk.W)

        ttk.Label(hybrid_frame, text="Orange S Min:").pack(anchor=tk.W, pady=(10, 0))
        self.hybrid_orange_s_min_var = tk.IntVar(value=100)
        hybrid_orange_s_min_slider = ttk.Scale(
            hybrid_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.hybrid_orange_s_min_var,
            command=self.on_param_change
        )
        hybrid_orange_s_min_slider.pack(fill=tk.X, pady=2)
        self.hybrid_orange_s_min_label = ttk.Label(hybrid_frame, text="100")
        self.hybrid_orange_s_min_label.pack(anchor=tk.W)

        ttk.Label(hybrid_frame, text="Orange V Min:").pack(anchor=tk.W, pady=(10, 0))
        self.hybrid_orange_v_min_var = tk.IntVar(value=100)
        hybrid_orange_v_min_slider = ttk.Scale(
            hybrid_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.hybrid_orange_v_min_var,
            command=self.on_param_change
        )
        hybrid_orange_v_min_slider.pack(fill=tk.X, pady=2)
        self.hybrid_orange_v_min_label = ttk.Label(hybrid_frame, text="100")
        self.hybrid_orange_v_min_label.pack(anchor=tk.W)

        ttk.Label(hybrid_frame, text="White V Min:").pack(anchor=tk.W, pady=(10, 0))
        self.hybrid_white_v_min_var = tk.IntVar(value=180)
        hybrid_white_v_min_slider = ttk.Scale(
            hybrid_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.hybrid_white_v_min_var,
            command=self.on_param_change
        )
        hybrid_white_v_min_slider.pack(fill=tk.X, pady=2)
        self.hybrid_white_v_min_label = ttk.Label(hybrid_frame, text="180")
        self.hybrid_white_v_min_label.pack(anchor=tk.W)

        ttk.Label(hybrid_frame, text="White S Max:").pack(anchor=tk.W, pady=(10, 0))
        self.hybrid_white_s_max_var = tk.IntVar(value=50)
        hybrid_white_s_max_slider = ttk.Scale(
            hybrid_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.hybrid_white_s_max_var,
            command=self.on_param_change
        )
        hybrid_white_s_max_slider.pack(fill=tk.X, pady=2)
        self.hybrid_white_s_max_label = ttk.Label(hybrid_frame, text="50")
        self.hybrid_white_s_max_label.pack(anchor=tk.W)

        # Launch point control
        launch_frame = ttk.LabelFrame(trajectory_tab, text="Launch Point", padding="10")
        launch_frame.pack(fill=tk.X, pady=5)

        ttk.Label(launch_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        self.launch_frame_var = tk.IntVar(value=0)
        self.launch_frame_entry = ttk.Entry(
            launch_frame,
            textvariable=self.launch_frame_var,
            width=8
        )
        self.launch_frame_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            launch_frame,
            text="Use Current Frame",
            command=self.set_launch_to_current
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            launch_frame,
            text="Auto Detect",
            command=self.auto_detect_launch
        ).pack(side=tk.LEFT, padx=5)

        # Current frame info (trajectory)
        traj_current_frame = ttk.LabelFrame(trajectory_tab, text="Current Frame", padding="10")
        traj_current_frame.pack(fill=tk.X, pady=5)

        self.current_position_label = ttk.Label(traj_current_frame, text="Position: --", font=("Arial", 12, "bold"))
        self.current_position_label.pack()

        self.current_velocity_label = ttk.Label(traj_current_frame, text="Velocity: --")
        self.current_velocity_label.pack()

        self.current_speed_label = ttk.Label(traj_current_frame, text="Speed: --")
        self.current_speed_label.pack()

        self.current_acceleration_label = ttk.Label(traj_current_frame, text="Acceleration: --")
        self.current_acceleration_label.pack()

        # Launch point physics
        self.launch_physics_frame = ttk.LabelFrame(trajectory_tab, text="Launch Physics", padding="10")
        self.launch_physics_frame.pack(fill=tk.X, pady=5)

        self.launch_speed_label = ttk.Label(self.launch_physics_frame, text="Speed: --")
        self.launch_speed_label.pack(anchor=tk.W)

        self.launch_angle_label = ttk.Label(self.launch_physics_frame, text="Angle: --")
        self.launch_angle_label.pack(anchor=tk.W)

        self.launch_force_label = ttk.Label(self.launch_physics_frame, text="Force: --")
        self.launch_force_label.pack(anchor=tk.W)

        self.launch_energy_label = ttk.Label(self.launch_physics_frame, text="Energy: --")
        self.launch_energy_label.pack(anchor=tk.W)

        # Overall statistics (trajectory)
        traj_stats_frame = ttk.LabelFrame(trajectory_tab, text="Overall Statistics", padding="10")
        traj_stats_frame.pack(fill=tk.X, pady=5)

        self.avg_speed_label = ttk.Label(traj_stats_frame, text="Avg Speed: --")
        self.avg_speed_label.pack(anchor=tk.W)

        self.max_speed_label = ttk.Label(traj_stats_frame, text="Max Speed: --")
        self.max_speed_label.pack(anchor=tk.W)

        self.max_accel_label = ttk.Label(traj_stats_frame, text="Max Accel: --")
        self.max_accel_label.pack(anchor=tk.W)

        self.distance_label = ttk.Label(traj_stats_frame, text="Distance: --")
        self.distance_label.pack(anchor=tk.W)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            info_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=10)

        self.status_label = ttk.Label(info_frame, text="Ready", wraplength=250)
        self.status_label.pack(pady=5)

    def load_video(self):
        """Load a video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.video_path = file_path
        self.status_label.config(text=f"Loaded: {Path(file_path).name}")
        self.process_btn.config(state=tk.NORMAL)

        # Load first frame as preview
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        if ret:
            # Rotate frame 90 degrees clockwise if rotation enabled
            if self.rotation_var.get():
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            self.display_frame(frame)
        cap.release()

    def on_mode_change(self, event=None):
        """Handle mode change between spin and trajectory."""
        self.analysis_mode = self.mode_var.get()

        # Switch to appropriate notebook tab
        if self.analysis_mode == "spin":
            self.notebook.select(0)
        else:
            self.notebook.select(1)

        # Refresh display if video is processed
        if self.processed_data and self.tracker:
            self.show_frame(self.current_frame_idx)

    def on_zoom_toggle(self):
        """Handle zoom toggle."""
        self.zoom_enabled = self.zoom_var.get()

    def on_param_change(self, event=None):
        """Handle detection parameter changes in real-time."""
        # Update label displays
        self.yolo_weight_label.config(text=f"{self.yolo_weight_var.get():.2f}")
        self.hough_weight_label.config(text=f"{self.hough_weight_var.get():.2f}")
        self.hybrid_weight_label.config(text=f"{self.hybrid_weight_var.get():.2f}")
        self.yolo_conf_label.config(text=f"{self.yolo_conf_var.get():.2f}")
        self.yolo_min_mask_area_label.config(text=f"{self.yolo_min_mask_area_var.get()}")
        self.yolo_radius_scale_label.config(text=f"{self.yolo_radius_scale_var.get():.2f}")
        self.hough_param1_label.config(text=f"{self.hough_param1_var.get()}")
        self.hough_param2_label.config(text=f"{self.hough_param2_var.get()}")
        self.hough_min_radius_label.config(text=f"{self.hough_min_radius_var.get()}")
        self.hough_max_radius_label.config(text=f"{self.hough_max_radius_var.get()}")
        self.hybrid_conf_label.config(text=f"{self.hybrid_conf_var.get():.2f}")
        self.hybrid_orange_h_min_label.config(text=f"{self.hybrid_orange_h_min_var.get()}")
        self.hybrid_orange_h_max_label.config(text=f"{self.hybrid_orange_h_max_var.get()}")
        self.hybrid_orange_s_min_label.config(text=f"{self.hybrid_orange_s_min_var.get()}")
        self.hybrid_orange_v_min_label.config(text=f"{self.hybrid_orange_v_min_var.get()}")
        self.hybrid_white_v_min_label.config(text=f"{self.hybrid_white_v_min_var.get()}")
        self.hybrid_white_s_max_label.config(text=f"{self.hybrid_white_s_max_var.get()}")

        # Update tracker parameters if tracker exists
        if self.tracker:
            self.tracker.yolo_weight = self.yolo_weight_var.get()
            self.tracker.hough_weight = self.hough_weight_var.get()
            self.tracker.hybrid_weight = self.hybrid_weight_var.get()
            self.tracker.yolo_conf_threshold = self.yolo_conf_var.get()
            self.tracker.yolo_min_mask_area = self.yolo_min_mask_area_var.get()
            self.tracker.yolo_radius_scale = self.yolo_radius_scale_var.get()
            self.tracker.yolo_debug = self.debug_mode_var.get()
            self.tracker.hough_param1 = self.hough_param1_var.get()
            self.tracker.hough_param2 = self.hough_param2_var.get()
            self.tracker.hough_min_radius = self.hough_min_radius_var.get()
            self.tracker.hough_max_radius = self.hough_max_radius_var.get()
            self.tracker.hybrid_confidence_threshold = self.hybrid_conf_var.get()
            self.tracker.hybrid_orange_h_min = self.hybrid_orange_h_min_var.get()
            self.tracker.hybrid_orange_h_max = self.hybrid_orange_h_max_var.get()
            self.tracker.hybrid_orange_s_min = self.hybrid_orange_s_min_var.get()
            self.tracker.hybrid_orange_v_min = self.hybrid_orange_v_min_var.get()
            self.tracker.hybrid_white_v_min = self.hybrid_white_v_min_var.get()
            self.tracker.hybrid_white_s_max = self.hybrid_white_s_max_var.get()

            # Show feedback that parameters were updated
            self.status_label.config(text="Parameters updated - detection re-running...")

            # Refresh current frame if video is loaded
            if self.processed_data:
                self.show_frame(self.current_frame_idx)
                self.status_label.config(text="Parameters applied")
            else:
                self.status_label.config(text="Parameters updated (process video to see changes)")
        else:
            self.status_label.config(text="Load video first to apply parameters")

    def set_launch_to_current(self):
        """Set launch point to current frame."""
        self.launch_frame_var.set(self.current_frame_idx)
        self.update_launch_physics()

    def auto_detect_launch(self):
        """Automatically detect launch point."""
        if self.trajectory_analyzer:
            launch_frame = self.trajectory_analyzer.find_launch_point()
            self.launch_frame_var.set(launch_frame)
            self.update_launch_physics()

    def update_launch_physics(self):
        """Update launch point physics properties."""
        if not self.trajectory_analyzer:
            return

        launch_frame = self.launch_frame_var.get()
        if launch_frame < 0 or launch_frame >= len(self.trajectory_analyzer.trajectory_points):
            messagebox.showwarning("Warning", "Launch frame is out of valid range")
            return

        physics = self.trajectory_analyzer.calculate_launch_kinematics(launch_frame)
        if physics:
            if 'velocity_m_s' in physics:
                self.launch_speed_label.config(text=f"Speed: {physics['velocity_m_s']:.2f} m/s")
            if 'launch_angle_deg' in physics:
                self.launch_angle_label.config(text=f"Angle: {physics['launch_angle_deg']:.1f}°")
            if 'force_n' in physics:
                self.launch_force_label.config(text=f"Force: {physics['force_n']:.2f} N")
            if 'kinetic_energy_j' in physics:
                self.launch_energy_label.config(text=f"Energy: {physics['kinetic_energy_j']:.3f} J")

    def process_video(self):
        """Process the video in a separate thread."""
        if not self.video_path:
            return

        try:
            fps = float(self.fps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid FPS value")
            return

        self.process_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing video...")
        self.progress_var.set(0)

        # Run processing in separate thread
        thread = threading.Thread(target=self._process_video_thread, args=(fps,))
        thread.daemon = True
        thread.start()

    def _process_video_thread(self, fps):
        """Thread function for video processing."""
        try:
            self.tracker = SpinTracker(
                self.video_path,
                fps=fps,
                rotate_90_cw=self.rotation_var.get()
            )
            self.trajectory_analyzer = TrajectoryAnalyzer(fps=fps)

            # Rotate video based on user setting
            self.needs_rotation = self.rotation_var.get()

            def progress_callback(current, total):
                progress = (current / total) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()

            self.processed_data = self.tracker.process_video(progress_callback)

            # Build trajectory data from spin data
            for data in self.processed_data:
                if data.ball_position and data.ball_radius:
                    # Adjust coordinates if video is rotated
                    if self.needs_rotation:
                        x, y = data.ball_position
                        # Rotate coordinates 90 degrees clockwise
                        # This is a simplification - need to know actual video dimensions
                        # For now, assuming portrait orientation with height > width
                        pass
                    traj_point = self.trajectory_analyzer.add_position(
                        data.frame_number,
                        data.ball_position[0],
                        data.ball_position[1],
                        data.ball_radius
                    )

                    # Store spin information in trajectory point for analysis
                    if hasattr(traj_point, 'trajectory_points'):
                        # Find the created trajectory point
                        for p in self.trajectory_analyzer.trajectory_points:
                            if p.frame_number == data.frame_number:
                                p.rps = data.rps
                                p.topspin_rps = data.topspin_rps
                                p.sidespin_rps = data.sidespin_rps
                                p.direction = data.direction

            # Calculate trajectory derivatives
            self.trajectory_analyzer.calculate_derivatives()

            # Update UI on completion
            self.root.after(0, self._on_processing_complete)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))

    def _on_processing_complete(self):
        """Called when video processing is complete."""
        self.status_label.config(text="Processing complete!")
        self.process_btn.config(state=tk.NORMAL)
        self.play_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        self.frame_slider.state(['!disabled'])
        self.frame_slider.config(to=self.tracker.frame_count - 1)

        # Update statistics
        self.update_statistics()

        # Update launch physics with auto-detected launch point
        if self.trajectory_analyzer:
            self.auto_detect_launch()

        # Show first frame
        self.current_frame_idx = 0
        self.show_frame(0)

    def update_statistics(self):
        """Update the statistics display."""
        if not self.processed_data:
            return

        # Update spin statistics
        valid_spins = [s.rps for s in self.processed_data if abs(s.rps) > 0.1]

        if valid_spins:
            avg_rps = sum(valid_spins) / len(valid_spins)
            max_rps = max(valid_spins, key=abs)
            min_rps = min(valid_spins, key=abs)

            self.avg_rps_label.config(text=f"Avg RPS: {avg_rps:.2f} ({avg_rps*60:.1f} RPM)")
            self.max_rps_label.config(text=f"Max RPS: {max_rps:.2f} ({max_rps*60:.1f} RPM)")
            self.min_rps_label.config(text=f"Min RPS: {min_rps:.2f} ({min_rps*60:.1f} RPM)")

        self.frames_analyzed_label.config(text=f"Frames: {len(self.processed_data)}")

        # Update trajectory statistics
        if self.trajectory_analyzer:
            traj_stats = self.trajectory_analyzer.get_statistics()

            if self.trajectory_analyzer.pixels_per_mm:
                if 'avg_speed_m_s' in traj_stats:
                    self.avg_speed_label.config(text=f"Avg Speed: {traj_stats['avg_speed_m_s']:.2f} m/s")
                if 'max_speed_m_s' in traj_stats:
                    self.max_speed_label.config(text=f"Max Speed: {traj_stats['max_speed_m_s']:.2f} m/s")
                if 'max_acceleration_m_s2' in traj_stats:
                    self.max_accel_label.config(text=f"Max Accel: {traj_stats['max_acceleration_m_s2']:.2f} m/s²")
                if 'total_distance_m' in traj_stats:
                    self.distance_label.config(text=f"Distance: {traj_stats['total_distance_m']:.2f} m")
                if 'duration' in traj_stats:
                    duration_label = ttk.Label(self.distance_label.master, text=f"Duration: {traj_stats['duration']:.3f} s")
                    duration_label.pack(anchor=tk.W)
            else:
                # Fallback to pixel units if not calibrated
                if 'avg_speed_px_s' in traj_stats:
                    self.avg_speed_label.config(text=f"Avg Speed: {traj_stats['avg_speed_px_s']:.1f} px/s")
                if 'max_speed_px_s' in traj_stats:
                    self.max_speed_label.config(text=f"Max Speed: {traj_stats['max_speed_px_s']:.1f} px/s")
                if 'max_acceleration_px_s2' in traj_stats:
                    self.max_accel_label.config(text=f"Max Accel: {traj_stats['max_acceleration_px_s2']:.1f} px/s²")
                if 'duration' in traj_stats:
                    self.distance_label.config(text=f"Duration: {traj_stats['duration']:.3f} s")

    def show_frame(self, frame_idx):
        """Display a specific frame with annotations."""
        if not self.tracker:
            return

        self.tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.tracker.cap.read()

        if not ret:
            return

        # Rotate frame if needed
        if hasattr(self, 'needs_rotation') and self.needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Re-run detection on current frame with current parameters
        ball_detection = self.tracker.detect_ball(frame)

        # Find corresponding spin data from cached results
        spin_data = None
        for data in self.processed_data:
            if data.frame_number == frame_idx:
                spin_data = data
                break

        # Update spin_data with new detection results
        if spin_data:
            if ball_detection:
                spin_data.ball_position = (ball_detection[0], ball_detection[1])
                spin_data.ball_radius = ball_detection[2]
            else:
                # Clear old detection if new detection returns None
                spin_data.ball_position = None
                spin_data.ball_radius = None

        # Apply zoom if enabled
        crop_offset = (0, 0)
        if self.zoom_enabled and spin_data and spin_data.ball_position and spin_data.ball_radius:
            frame, crop_offset = self._crop_to_ball(frame, spin_data)

        # Draw detection method indicator
        self._draw_detection_info(frame)

        # Draw annotations based on mode
        if self.analysis_mode == "spin":
            frame = self._draw_spin_annotations(frame, spin_data, crop_offset)
        else:  # trajectory mode
            frame = self._draw_trajectory_annotations(frame, frame_idx, spin_data, crop_offset)

        self.display_frame(frame)
        self.frame_label.config(text=f"Frame: {frame_idx}/{self.tracker.frame_count}")

    def _draw_detection_info(self, frame):
        """Draw detection method information overlay."""
        if not self.tracker:
            return

        y_offset = 30
        line_height = 25

        # Title
        cv2.putText(
            frame,
            "Detection Methods:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        y_offset += line_height

        # YOLOv8
        yolo_weight = self.yolo_weight_var.get()
        color = (0, 255, 0) if yolo_weight > 0 else (100, 100, 100)
        cv2.putText(
            frame,
            f"YOLOv8: {yolo_weight:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        y_offset += line_height

        # Hough
        hough_weight = self.hough_weight_var.get()
        color = (0, 255, 0) if hough_weight > 0 else (100, 100, 100)
        cv2.putText(
            frame,
            f"Hough: {hough_weight:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        y_offset += line_height

        # Hybrid
        hybrid_weight = self.hybrid_weight_var.get()
        color = (0, 255, 0) if hybrid_weight > 0 else (100, 100, 100)
        cv2.putText(
            frame,
            f"Hybrid: {hybrid_weight:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    def _crop_to_ball(self, frame, spin_data):
        """Crop frame to ball region for zoom view."""
        x, y = spin_data.ball_position
        r = spin_data.ball_radius if spin_data.ball_radius else 30

        # Calculate crop region
        margin = int(r * self.zoom_margin)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + margin)
        y2 = min(frame.shape[0], y + margin)

        # Crop frame
        cropped = frame[y1:y2, x1:x2]

        # Return cropped frame and offset for coordinate adjustment
        return cropped, (x1, y1)

    def _draw_spin_annotations(self, frame, spin_data, crop_offset=(0, 0)):
        """Draw spin-related annotations on frame."""
        offset_x, offset_y = crop_offset

        # Draw segmentation mask if enabled and available
        if self.show_segmentation_var.get() and self.tracker and self.tracker.yolo_detector:
            mask = self.tracker.yolo_detector.get_last_mask()
            if mask is not None:
                # Create colored overlay for segmentation
                mask_overlay = np.zeros_like(frame)
                # Use semi-transparent green for the mask
                mask_overlay[mask > 0.5] = [0, 255, 0]
                # Blend with original frame
                frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.3, 0)

        if spin_data and spin_data.ball_position:
            x, y = spin_data.ball_position
            # Adjust coordinates for crop offset
            x -= offset_x
            y -= offset_y
            r = spin_data.ball_radius if spin_data.ball_radius else 20

            # Draw ball circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Draw arrow from ball center to orange centroid (rotation indicator)
            if spin_data.orange_centroid:
                ox, oy = spin_data.orange_centroid
                ox -= offset_x
                oy -= offset_y

                # Draw arrow from ball center pointing toward orange center
                cv2.arrowedLine(frame, (x, y), (ox, oy), (0, 165, 255), 3, tipLength=0.3)

                # Draw orange centroid marker
                cv2.circle(frame, (ox, oy), 5, (0, 165, 255), -1)
                cv2.putText(
                    frame,
                    "ORANGE",
                    (ox + 10, oy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2
                )

            # Draw orange centroid
            if spin_data.orange_centroid:
                ox, oy = spin_data.orange_centroid
                ox -= offset_x
                oy -= offset_y
                cv2.circle(frame, (ox, oy), 5, (0, 165, 255), -1)
                cv2.line(frame, (x, y), (ox, oy), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    "ORANGE",
                    (ox + 10, oy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2
                )

            # Draw white centroid
            if spin_data.white_centroid:
                wx, wy = spin_data.white_centroid
                wx -= offset_x
                wy -= offset_y
                cv2.circle(frame, (wx, wy), 5, (255, 255, 255), -1)
                cv2.line(frame, (x, y), (wx, wy), (200, 200, 200), 2)
                cv2.putText(
                    frame,
                    "WHITE",
                    (wx + 10, wy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            # Display spin info with direction
            info_y = y - r - 20
            cv2.putText(
                frame,
                f"Total: {spin_data.rps:.2f} RPS",
                (x + r + 10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            info_y += 20
            cv2.putText(
                frame,
                f"Topspin: {abs(spin_data.topspin_rps):.2f} RPS",
                (x + r + 10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            info_y += 20
            cv2.putText(
                frame,
                f"Sidespin: {abs(spin_data.sidespin_rps):.2f} RPS",
                (x + r + 10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2
            )

            info_y += 20
            cv2.putText(
                frame,
                f"Type: {spin_data.direction}",
                (x + r + 10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            # Update current frame info
            self.current_rps_label.config(text=f"Total: {spin_data.rps:.2f} RPS")
            self.current_rpm_label.config(text=f"Topspin: {abs(spin_data.topspin_rps):.2f} | Sidespin: {abs(spin_data.sidespin_rps):.2f}")
            self.current_angle_label.config(text=f"Type: {spin_data.direction}")

        return frame

    def _draw_trajectory_annotations(self, frame, frame_idx, spin_data, crop_offset=(0, 0)):
        """Draw trajectory-related annotations on frame."""
        if not self.trajectory_analyzer:
            return frame

        offset_x, offset_y = crop_offset

        # Draw trajectory path
        trajectory_path = self.trajectory_analyzer.get_trajectory_path()
        if len(trajectory_path) > 1:
            for i in range(len(trajectory_path) - 1):
                # Adjust coordinates for crop offset
                pt1 = (trajectory_path[i][0] - offset_x, trajectory_path[i][1] - offset_y)
                pt2 = (trajectory_path[i+1][0] - offset_x, trajectory_path[i+1][1] - offset_y)
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Draw current ball position
        if spin_data and spin_data.ball_position:
            x, y = spin_data.ball_position
            # Adjust coordinates for crop offset
            x -= offset_x
            y -= offset_y
            r = spin_data.ball_radius if spin_data.ball_radius else 20

            # Draw ball circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Find trajectory point for this frame
            traj_point = None
            if frame_idx < len(self.trajectory_analyzer.trajectory_points):
                traj_point = self.trajectory_analyzer.trajectory_points[frame_idx]

            if traj_point:
                # Draw velocity vector
                if traj_point.velocity_2d:
                    vx, vy = traj_point.velocity_2d
                    # Scale down for visualization
                    scale = 0.1
                    end_x = int(x + vx * scale)
                    end_y = int(y + vy * scale)
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

                # Display trajectory info
                info_y = y - r - 20

                if self.trajectory_analyzer.pixels_per_mm:
                    # Use real-world units
                    speed_ms = traj_point.speed / self.trajectory_analyzer.pixels_per_mm / 1000
                    cv2.putText(
                        frame,
                        f"Speed: {speed_ms:.2f} m/s",
                        (x + r + 10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                else:
                    # Fallback to pixel units
                    cv2.putText(
                        frame,
                        f"Speed: {traj_point.speed:.1f} px/s",
                        (x + r + 10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                if traj_point.acceleration_2d and self.trajectory_analyzer.pixels_per_mm:
                    ax, ay = traj_point.acceleration_2d
                    accel_mag_px = np.sqrt(ax**2 + ay**2)
                    accel_mag_ms2 = accel_mag_px / self.trajectory_analyzer.pixels_per_mm / 1000
                    info_y += 20
                    cv2.putText(
                        frame,
                        f"Accel: {accel_mag_ms2:.2f} m/s²",
                        (x + r + 10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2
                    )
                elif traj_point.acceleration_2d:
                    ax, ay = traj_point.acceleration_2d
                    accel_mag = np.sqrt(ax**2 + ay**2)
                    info_y += 20
                    cv2.putText(
                        frame,
                        f"Accel: {accel_mag:.1f} px/s²",
                        (x + r + 10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2
                    )

                # Update current frame info labels
                self.current_position_label.config(text=f"Position: ({x}, {y})")
                if traj_point.velocity_3d and self.trajectory_analyzer.pixels_per_mm:
                    self.current_speed_label.config(text=f"Speed: {speed_ms:.2f} m/s")
                elif traj_point.speed > 0:
                    self.current_speed_label.config(text=f"Speed: {traj_point.speed:.1f} px/s")

                if traj_point.velocity_2d:
                    if self.trajectory_analyzer.pixels_per_mm:
                        vx_ms = traj_point.velocity_2d[0] / self.trajectory_analyzer.pixels_per_mm / 1000
                        vy_ms = traj_point.velocity_2d[1] / self.trajectory_analyzer.pixels_per_mm / 1000
                        self.current_velocity_label.config(text=f"Velocity: ({vx_ms:.2f}, {vy_ms:.2f}) m/s")
                    else:
                        self.current_velocity_label.config(text=f"Velocity: ({traj_point.velocity_2d[0]:.1f}, {traj_point.velocity_2d[1]:.1f}) px/s")

                if traj_point.acceleration_2d:
                    if self.trajectory_analyzer.pixels_per_mm:
                        self.current_acceleration_label.config(text=f"Acceleration: {accel_mag_ms2:.2f} m/s²")
                    else:
                        self.current_acceleration_label.config(text=f"Acceleration: {accel_mag:.1f} px/s²")

        return frame

    def display_frame(self, frame):
        """Display a frame in the video label."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit display
        display_height = 600
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_width = int(display_height * aspect_ratio)

        frame_resized = cv2.resize(frame_rgb, (display_width, display_height))

        # Convert to PhotoImage
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)

        self.video_label.config(image=photo, text="")
        self.video_label.image = photo

    def toggle_play(self):
        """Toggle video playback."""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_btn.config(text="Pause")
            self.play_video()
        else:
            self.play_btn.config(text="Play")

    def play_video(self):
        """Play video frames."""
        if not self.is_playing or not self.tracker:
            return

        if self.current_frame_idx < self.tracker.frame_count - 1:
            self.current_frame_idx += 1
            self.frame_slider.set(self.current_frame_idx)
            self.show_frame(self.current_frame_idx)

            # Schedule next frame
            delay = int(1000 / self.tracker.fps)
            self.root.after(delay, self.play_video)
        else:
            self.is_playing = False
            self.play_btn.config(text="Play")

    def on_slider_change(self, value):
        """Handle slider movement."""
        if not self.is_playing:
            frame_idx = int(float(value))
            self.current_frame_idx = frame_idx
            self.show_frame(frame_idx)

    def export_csv(self):
        """Export results to CSV."""
        if not self.processed_data:
            messagebox.showwarning("Warning", "No data to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save CSV File",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                f.write("frame,rotation_angle,total_rps,total_rpm,topspin_rps,sidespin_rps,direction,ball_x,ball_y\n")
                for data in self.processed_data:
                    ball_x = data.ball_position[0] if data.ball_position else ''
                    ball_y = data.ball_position[1] if data.ball_position else ''
                    f.write(
                        f"{data.frame_number},{data.rotation_angle:.2f},"
                        f"{data.rps:.4f},{data.rps*60:.4f},"
                        f"{data.topspin_rps:.4f},{data.sidespin_rps:.4f},"
                        f"{data.direction},{ball_x},{ball_y}\n"
                    )

            messagebox.showinfo("Success", f"Data exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")

    def launch_frame_labeler(self):
        """Launch the frame labeling tool."""
        import subprocess
        import sys

        # Launch frame_labeler.py as a separate process
        labeler_path = Path(__file__).parent / "frame_labeler.py"

        if not labeler_path.exists():
            messagebox.showerror("Error", f"Frame labeler not found at {labeler_path}")
            return

        try:
            subprocess.Popen([sys.executable, str(labeler_path)])
            messagebox.showinfo("Info", "Frame labeler launched in separate window")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch frame labeler: {e}")


def main():
    root = tk.Tk()
    app = SpinTrackerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
