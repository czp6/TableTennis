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
from professional_video_loader import ProfessionalVideoLoader
from pose_comparison import PoseComparison
from gemini_coach import GeminiCoach
from session_manager import SessionManager


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
        self.analysis_mode = "spin"  # "spin", "trajectory", or "player"
        self.needs_rotation = True  # Rotate 90 degrees clockwise
        self.zoom_enabled = False  # Zoom to ball region
        self.zoom_margin = 3.0  # Multiplier for ball radius to determine crop size

        # Professional comparison
        self.pro_video_loader = ProfessionalVideoLoader()
        self.pose_comparison = PoseComparison()
        self.professional_data = None
        self.comparison_results = None

        # AI coaching
        self.gemini_coach = GeminiCoach()
        self.coaching_feedback = None

        # Session management
        self.session_manager = SessionManager()

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
            values=["spin", "trajectory", "player"],
            state="readonly",
            width=12
        )
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Process only current mode checkbox
        self.process_mode_only_var = tk.BooleanVar(value=False)
        mode_only_check = ttk.Checkbutton(
            control_frame,
            text="Process Mode Only",
            variable=self.process_mode_only_var
        )
        mode_only_check.pack(side=tk.LEFT, padx=5)

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

        # Player tracking tab
        player_tab = ttk.Frame(self.notebook)
        self.notebook.add(player_tab, text="Player Tracking")

        # Professional comparison tab
        comparison_tab = ttk.Frame(self.notebook)
        self.notebook.add(comparison_tab, text="Pro Comparison")

        # AI coaching tab
        coaching_tab = ttk.Frame(self.notebook)
        self.notebook.add(coaching_tab, text="AI Coaching")

        # Pose visualization controls
        pose_viz_frame = ttk.LabelFrame(player_tab, text="Visualization", padding="10")
        pose_viz_frame.pack(fill=tk.X, pady=5)

        self.show_skeleton_var = tk.BooleanVar(value=True)
        skeleton_check = ttk.Checkbutton(
            pose_viz_frame,
            text="Show Skeleton Overlay",
            variable=self.show_skeleton_var
        )
        skeleton_check.pack(anchor=tk.W)

        self.show_keypoint_conf_var = tk.BooleanVar(value=False)
        keypoint_conf_check = ttk.Checkbutton(
            pose_viz_frame,
            text="Show Keypoint Confidence",
            variable=self.show_keypoint_conf_var
        )
        keypoint_conf_check.pack(anchor=tk.W)

        self.track_multiple_people_var = tk.BooleanVar(value=False)
        multi_people_check = ttk.Checkbutton(
            pose_viz_frame,
            text="Track Multiple People",
            variable=self.track_multiple_people_var
        )
        multi_people_check.pack(anchor=tk.W)

        self.highlight_dominant_hand_var = tk.BooleanVar(value=True)
        dominant_hand_check = ttk.Checkbutton(
            pose_viz_frame,
            text="Highlight Right Hand (Dominant)",
            variable=self.highlight_dominant_hand_var
        )
        dominant_hand_check.pack(anchor=tk.W)

        # Detection statistics frame
        stats_frame = ttk.LabelFrame(player_tab, text="Detection Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)

        self.detection_rate_label = ttk.Label(stats_frame, text="Detection Rate: --")
        self.detection_rate_label.pack(anchor=tk.W)

        self.active_model_label = ttk.Label(stats_frame, text="Active Model: --")
        self.active_model_label.pack(anchor=tk.W)

        # Update model name if tracker is available
        if hasattr(self, 'tracker') and self.tracker and hasattr(self.tracker, 'pose_detector'):
            if hasattr(self.tracker.pose_detector, 'get_active_model_name'):
                model_name = self.tracker.pose_detector.get_active_model_name()
                self.active_model_label.config(text=f"Active Model: {model_name}")
            else:
                self.active_model_label.config(text="Active Model: YOLOv8m-pose")

        self.avg_confidence_label = ttk.Label(stats_frame, text="Avg Confidence: --")
        self.avg_confidence_label.pack(anchor=tk.W)

        ttk.Label(pose_viz_frame, text="Pose Confidence Threshold:").pack(anchor=tk.W, pady=(10, 0))
        self.pose_conf_var = tk.DoubleVar(value=0.2)
        pose_conf_slider = ttk.Scale(
            pose_viz_frame,
            from_=0.1,
            to=0.5,
            orient=tk.HORIZONTAL,
            variable=self.pose_conf_var,
            command=self.on_pose_param_change
        )
        pose_conf_slider.pack(fill=tk.X, pady=2)
        self.pose_conf_label = ttk.Label(pose_viz_frame, text="0.20")
        self.pose_conf_label.pack(anchor=tk.W)

        # Current pose metrics
        pose_metrics_frame = ttk.LabelFrame(player_tab, text="Current Pose Metrics", padding="10")
        pose_metrics_frame.pack(fill=tk.X, pady=5)

        self.stance_width_label = ttk.Label(pose_metrics_frame, text="Stance Width: --")
        self.stance_width_label.pack(anchor=tk.W)

        self.arm_extension_label = ttk.Label(pose_metrics_frame, text="Arm Extension: --")
        self.arm_extension_label.pack(anchor=tk.W)

        self.body_rotation_label = ttk.Label(pose_metrics_frame, text="Body Rotation: --")
        self.body_rotation_label.pack(anchor=tk.W)

        self.knee_bend_label = ttk.Label(pose_metrics_frame, text="Knee Bend: --")
        self.knee_bend_label.pack(anchor=tk.W)

        self.stroke_phase_label = ttk.Label(pose_metrics_frame, text="Stroke Phase: --", font=("Arial", 12, "bold"))
        self.stroke_phase_label.pack(anchor=tk.W, pady=(10, 0))

        # Pose data export
        pose_export_frame = ttk.Frame(player_tab)
        pose_export_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            pose_export_frame,
            text="Export Pose Data",
            command=self.export_pose_data
        ).pack(side=tk.LEFT, padx=5)

        # Professional comparison controls
        pro_select_frame = ttk.LabelFrame(comparison_tab, text="Professional Player", padding="10")
        pro_select_frame.pack(fill=tk.X, pady=5)

        ttk.Label(pro_select_frame, text="Select Professional:").pack(anchor=tk.W)
        self.pro_player_var = tk.StringVar(value="Ma Long")
        pro_combo = ttk.Combobox(
            pro_select_frame,
            textvariable=self.pro_player_var,
            values=["Ma Long", "Fan Zhendong", "Custom..."],
            state="readonly",
            width=20
        )
        pro_combo.pack(fill=tk.X, pady=5)

        ttk.Button(
            pro_select_frame,
            text="Load Professional Video",
            command=self.load_professional_video
        ).pack(fill=tk.X, pady=5)

        ttk.Button(
            pro_select_frame,
            text="Compare with Professional",
            command=self.compare_with_professional
        ).pack(fill=tk.X, pady=5)

        # Comparison results
        comparison_results_frame = ttk.LabelFrame(comparison_tab, text="Comparison Results", padding="10")
        comparison_results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.comparison_text = tk.Text(comparison_results_frame, height=15, wrap=tk.WORD)
        self.comparison_text.pack(fill=tk.BOTH, expand=True)

        # AI coaching controls
        coaching_controls_frame = ttk.LabelFrame(coaching_tab, text="Generate Feedback", padding="10")
        coaching_controls_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            coaching_controls_frame,
            text="Generate AI Coaching Feedback",
            command=self.generate_coaching_feedback
        ).pack(fill=tk.X, pady=5)

        ttk.Label(coaching_controls_frame, text="Focus on top N areas:").pack(anchor=tk.W, pady=(10, 0))
        self.coaching_focus_var = tk.IntVar(value=3)
        coaching_focus_spin = ttk.Spinbox(
            coaching_controls_frame,
            from_=1,
            to=5,
            textvariable=self.coaching_focus_var,
            width=10
        )
        coaching_focus_spin.pack(anchor=tk.W, pady=5)

        # Coaching feedback display
        coaching_feedback_frame = ttk.LabelFrame(coaching_tab, text="Coaching Feedback", padding="10")
        coaching_feedback_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.coaching_text = tk.Text(coaching_feedback_frame, height=20, wrap=tk.WORD)
        self.coaching_text.pack(fill=tk.BOTH, expand=True)

        coaching_export_frame = ttk.Frame(coaching_tab)
        coaching_export_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            coaching_export_frame,
            text="Export Coaching Report",
            command=self.export_coaching_report
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            coaching_export_frame,
            text="Save Session",
            command=self.save_session
        ).pack(side=tk.LEFT, padx=5)

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
        """Handle mode change between spin, trajectory, and player tracking."""
        self.analysis_mode = self.mode_var.get()

        # Switch to appropriate notebook tab
        if self.analysis_mode == "spin":
            self.notebook.select(0)
        elif self.analysis_mode == "trajectory":
            self.notebook.select(1)
        elif self.analysis_mode == "player":
            self.notebook.select(2)

        # Refresh display if video is processed
        if self.processed_data and self.tracker:
            self.show_frame(self.current_frame_idx)

    def on_zoom_toggle(self):
        """Handle zoom toggle."""
        self.zoom_enabled = self.zoom_var.get()

    def on_pose_param_change(self, event=None):
        """Handle pose parameter changes."""
        self.pose_conf_label.config(text=f"{self.pose_conf_var.get():.2f}")

        # Update tracker parameters if tracker exists
        if self.tracker:
            self.tracker.pose_conf_threshold = self.pose_conf_var.get()

            # Refresh current frame if video is loaded
            if self.processed_data:
                self.show_frame(self.current_frame_idx)

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
            # Determine processing mode
            process_mode = "all"
            if self.process_mode_only_var.get():
                process_mode = self.mode_var.get()
                if process_mode == "spin":
                    process_mode = "ball"  # Spin mode needs ball detection
                elif process_mode == "trajectory":
                    process_mode = "ball"  # Trajectory mode needs ball detection

            self.tracker = SpinTracker(
                self.video_path,
                fps=fps,
                rotate_90_cw=self.rotation_var.get(),
                process_mode=process_mode
            )

            # Set multi-person tracking flag
            if hasattr(self.tracker, 'track_multiple_people'):
                self.tracker.track_multiple_people = self.track_multiple_people_var.get()
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

        # Update active model label
        if self.tracker and hasattr(self.tracker, 'pose_detector'):
            if hasattr(self.tracker.pose_detector, 'get_active_model_name'):
                model_name = self.tracker.pose_detector.get_active_model_name()
                self.active_model_label.config(text=f"Active Model: {model_name}")
            else:
                self.active_model_label.config(text="Active Model: YOLOv8m-pose")

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
        elif self.analysis_mode == "trajectory":
            frame = self._draw_trajectory_annotations(frame, frame_idx, spin_data, crop_offset)
        elif self.analysis_mode == "player":
            frame = self._draw_player_annotations(frame, spin_data, crop_offset)

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

    def _draw_player_annotations(self, frame, spin_data, crop_offset=(0, 0)):
        """Draw player pose annotations on frame."""
        offset_x, offset_y = crop_offset

        # Draw multiple people if enabled
        if self.track_multiple_people_var.get() and self.tracker and hasattr(self.tracker, 'multi_pose_data_history'):
            frame_idx = spin_data.frame_number if spin_data else 0
            if frame_idx < len(self.tracker.multi_pose_data_history):
                poses = self.tracker.multi_pose_data_history[frame_idx]
                if self.show_skeleton_var.get():
                    for i, pose in enumerate(poses):
                        # Use different colors for different people
                        person_color_offset = i * 60  # Hue shift
                        frame = self._draw_skeleton(frame, pose, offset_x, offset_y, person_id=i)

        # Draw primary person
        if not spin_data or not spin_data.pose_keypoints:
            return frame

        pose = spin_data.pose_keypoints

        # Draw skeleton if enabled (and not already drawn in multi-person mode)
        if self.show_skeleton_var.get() and not self.track_multiple_people_var.get():
            frame = self._draw_skeleton(frame, pose, offset_x, offset_y)

        # Draw keypoint confidence if enabled
        if self.show_keypoint_conf_var.get():
            frame = self._draw_keypoint_confidence(frame, pose, offset_x, offset_y)

        # Update pose metrics labels
        if spin_data.pose_metrics:
            metrics = spin_data.pose_metrics
            self.stance_width_label.config(text=f"Stance Width: {metrics.stance_width:.1f} px")
            self.arm_extension_label.config(text=f"Arm Extension: {metrics.arm_extension:.1f} px")
            self.body_rotation_label.config(text=f"Body Rotation: {metrics.body_rotation:.1f}°")
            self.knee_bend_label.config(text=f"Knee Bend: {metrics.knee_bend:.1f}°")
            self.stroke_phase_label.config(text=f"Stroke Phase: {metrics.stroke_phase.value}")

        # Update detection statistics
        if self.tracker and hasattr(self.tracker, 'pose_detection_attempts'):
            if self.tracker.pose_detection_attempts > 0:
                raw_rate = self.tracker.pose_detection_successes / self.tracker.pose_detection_attempts
                total_poses = self.tracker.pose_detection_successes + getattr(self.tracker, 'pose_interpolated_frames', 0)
                effective_rate = total_poses / self.tracker.pose_detection_attempts

                self.detection_rate_label.config(
                    text=f"Detection: {raw_rate:.1%} raw, {effective_rate:.1%} effective"
                )

                # Calculate average confidence from detected poses
                if spin_data.pose_keypoints:
                    valid_keypoints = spin_data.pose_keypoints.keypoints[spin_data.pose_keypoints.keypoints[:, 2] > 0.25]
                    if len(valid_keypoints) > 0:
                        avg_conf = valid_keypoints[:, 2].mean()

                        # Show tracking quality if available
                        if hasattr(self.tracker, 'pose_tracker') and self.tracker.pose_tracker:
                            quality = self.tracker.pose_tracker.get_tracking_quality()
                            self.avg_confidence_label.config(
                                text=f"Avg Conf: {avg_conf:.2f} | Quality: {quality:.1%}"
                            )
                        else:
                            self.avg_confidence_label.config(text=f"Avg Confidence: {avg_conf:.2f}")

        return frame

    def _draw_skeleton(self, frame, pose, offset_x, offset_y, person_id=0):
        """Draw skeleton connections on frame with right hand highlighting."""
        # COCO skeleton connections
        skeleton = [
            (5, 6),   # shoulders
            (5, 7),   # left shoulder to elbow
            (7, 9),   # left elbow to wrist
            (6, 8),   # right shoulder to elbow
            (8, 10),  # right elbow to wrist
            (5, 11),  # left shoulder to hip
            (6, 12),  # right shoulder to hip
            (11, 12), # hips
            (11, 13), # left hip to knee
            (13, 15), # left knee to ankle
            (12, 14), # right hip to knee
            (14, 16)  # right knee to ankle
        ]

        # Right arm connections (for highlighting)
        right_arm_connections = [(6, 8), (8, 10)]

        # Color offset for multi-person tracking
        hue_shift = person_id * 60

        # Draw connections
        for start_idx, end_idx in skeleton:
            start_kp = pose.get_keypoint(start_idx)
            end_kp = pose.get_keypoint(end_idx)

            if start_kp and end_kp:
                # Check confidence - lowered threshold for smoother display
                conf = min(start_kp[2], end_kp[2])
                if conf < 0.15:  # Even lower threshold for maximum continuity
                    continue

                # Adjust for crop offset
                start_pt = (int(start_kp[0] - offset_x), int(start_kp[1] - offset_y))
                end_pt = (int(end_kp[0] - offset_x), int(end_kp[1] - offset_y))

                # Check if this is right arm (dominant hand)
                is_right_arm = (start_idx, end_idx) in right_arm_connections
                highlight_right = self.highlight_dominant_hand_var.get() if hasattr(self, 'highlight_dominant_hand_var') else True

                # Color based on confidence and right arm highlighting
                if is_right_arm and highlight_right:
                    # Right arm gets special highlighting
                    if conf > 0.7:
                        color = (255, 0, 255)  # Magenta - high confidence right arm
                    elif conf > 0.5:
                        color = (255, 100, 255)  # Light magenta
                    elif conf > 0.3:
                        color = (200, 0, 200)  # Dark magenta
                    else:
                        color = (150, 0, 150)  # Very dark magenta
                    thickness = 4  # Thicker for right arm
                else:
                    # Normal coloring for other connections
                    if conf > 0.7:
                        color = (0, 255, 0)  # Green - high confidence
                    elif conf > 0.5:
                        color = (0, 255, 255)  # Yellow - medium confidence
                    elif conf > 0.3:
                        color = (0, 165, 255)  # Orange - low confidence
                    elif conf > 0.15:
                        color = (128, 128, 255)  # Light red - interpolated/low confidence
                    else:
                        color = (0, 0, 255)  # Red - very low confidence
                    thickness = 3 if conf > 0.5 else 2

                # Apply hue shift for multi-person
                if person_id > 0:
                    color = self._shift_color_hue(color, hue_shift)

                cv2.line(frame, start_pt, end_pt, color, thickness)

        # Draw keypoints
        for i in range(17):
            kp = pose.get_keypoint(i)
            if kp and kp[2] > 0.15:  # Lowered threshold
                pt = (int(kp[0] - offset_x), int(kp[1] - offset_y))

                # Check if this is right wrist (keypoint 10)
                is_right_wrist = (i == 10)
                highlight_right = self.highlight_dominant_hand_var.get() if hasattr(self, 'highlight_dominant_hand_var') else True

                # Color and size based on confidence and right wrist
                if is_right_wrist and highlight_right:
                    # Right wrist gets special highlighting
                    color = (255, 0, 255)  # Magenta
                    radius = 8  # Larger for right wrist
                else:
                    # Normal coloring
                    if kp[2] > 0.7:
                        color = (0, 255, 0)  # Green
                    elif kp[2] > 0.5:
                        color = (0, 255, 255)  # Yellow
                    elif kp[2] > 0.3:
                        color = (0, 165, 255)  # Orange
                    elif kp[2] > 0.15:
                        color = (128, 128, 255)  # Light red - interpolated
                    else:
                        color = (0, 0, 255)  # Red
                    radius = 5 if kp[2] > 0.5 else 4

                # Apply hue shift for multi-person
                if person_id > 0:
                    color = self._shift_color_hue(color, hue_shift)

                cv2.circle(frame, pt, radius, color, -1)

                # Show confidence value if debug mode enabled
                if self.show_keypoint_conf_var.get():
                    cv2.putText(frame, f"{kp[2]:.2f}",
                               (pt[0]+5, pt[1]-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        return frame

    def _shift_color_hue(self, bgr_color, hue_shift):
        """Shift BGR color by hue for multi-person visualization."""
        # Convert BGR to HSV
        bgr_array = np.uint8([[bgr_color]])
        hsv = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)

        # Shift hue
        hsv[0][0][0] = (hsv[0][0][0] + hue_shift) % 180

        # Convert back to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(map(int, bgr[0][0]))

    def _draw_keypoint_confidence(self, frame, pose, offset_x, offset_y):
        """Draw keypoint confidence values on frame."""
        keypoint_names = [
            "nose", "l_eye", "r_eye", "l_ear", "r_ear",
            "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
            "l_wrist", "r_wrist", "l_hip", "r_hip",
            "l_knee", "r_knee", "l_ankle", "r_ankle"
        ]

        for i in range(17):
            kp = pose.get_keypoint(i)
            if kp and kp[2] > 0.3:
                pt = (int(kp[0] - offset_x), int(kp[1] - offset_y))
                text = f"{keypoint_names[i]}: {kp[2]:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1
                )

        return frame

    def export_pose_data(self):
        """Export pose data to JSON."""
        if not self.processed_data:
            messagebox.showwarning("Warning", "No data to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Pose Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            import json

            pose_data = {}
            for data in self.processed_data:
                if data.pose_keypoints and data.pose_metrics:
                    pose_data[data.frame_number] = {
                        'keypoints': data.pose_keypoints.keypoints.tolist(),
                        'metrics': data.pose_metrics.to_dict()
                    }

            with open(file_path, 'w') as f:
                json.dump(pose_data, f, indent=2)

            messagebox.showinfo("Success", f"Pose data exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")

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

    def load_professional_video(self):
        """Load and process professional player video."""
        player_name = self.pro_player_var.get()

        if player_name == "Custom...":
            # Let user select custom video
            file_path = filedialog.askopenfilename(
                title="Select Professional Player Video",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                return

            # Ask for player name
            from tkinter import simpledialog
            player_name = simpledialog.askstring("Player Name", "Enter professional player name:")
            if not player_name:
                return
        else:
            # Check if cached
            cached = self.pro_video_loader.get_cached_professional(player_name)
            if cached:
                self.professional_data = cached
                messagebox.showinfo("Success", f"Loaded cached data for {player_name}")
                return

            # Need to load video
            file_path = filedialog.askopenfilename(
                title=f"Select {player_name} Video",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                return

        # Process video
        self.status_label.config(text=f"Processing {player_name} video...")
        self.root.update()

        try:
            fps = float(self.fps_var.get())
            self.professional_data = self.pro_video_loader.load_and_process(file_path, player_name, fps)
            messagebox.showinfo("Success", f"Processed {len(self.professional_data['poses'])} poses for {player_name}")
            self.status_label.config(text=f"Loaded {player_name} data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process video: {e}")
            self.status_label.config(text="Ready")

    def compare_with_professional(self):
        """Compare amateur poses with professional."""
        if not self.processed_data:
            messagebox.showwarning("Warning", "Process your video first")
            return

        if not self.professional_data:
            messagebox.showwarning("Warning", "Load professional video first")
            return

        self.status_label.config(text="Comparing poses...")
        self.root.update()

        try:
            # Extract amateur poses with metrics
            amateur_poses = []
            for data in self.processed_data:
                if data.pose_metrics:
                    amateur_poses.append({
                        'frame_idx': data.frame_number,
                        'metrics': data.pose_metrics.to_dict()
                    })

            if not amateur_poses:
                messagebox.showwarning("Warning", "No pose data found in your video")
                return

            # Get professional poses
            pro_poses = self.professional_data['poses']
            pro_metrics = self.professional_data['metrics']

            # Combine poses with metrics
            professional_poses = []
            for i, pose in enumerate(pro_poses):
                if i < len(pro_metrics):
                    professional_poses.append({
                        'frame_idx': pose['frame_idx'],
                        'metrics': pro_metrics[i]
                    })

            # Calculate differences
            avg_differences = self.pose_comparison.calculate_average_differences(amateur_poses, professional_poses)
            improvement_areas = self.pose_comparison.identify_improvement_areas(avg_differences, top_n=5)

            # Store results
            self.comparison_results = {
                'professional_player': self.professional_data['player_name'],
                'avg_differences': avg_differences,
                'improvement_areas': improvement_areas
            }

            # Display results
            self.comparison_text.delete(1.0, tk.END)
            self.comparison_text.insert(tk.END, f"Comparison with {self.professional_data['player_name']}\n")
            self.comparison_text.insert(tk.END, "=" * 50 + "\n\n")

            self.comparison_text.insert(tk.END, "Top Areas for Improvement:\n\n")
            for i, area in enumerate(improvement_areas, 1):
                self.comparison_text.insert(tk.END, f"{i}. {area['description']}\n\n")

            self.comparison_text.insert(tk.END, "\nDetailed Metrics:\n")
            self.comparison_text.insert(tk.END, "-" * 50 + "\n")
            for key, value in avg_differences.items():
                if 'avg_' in key:
                    metric_name = key.replace('avg_', '').replace('_', ' ').title()
                    self.comparison_text.insert(tk.END, f"{metric_name}: {value:.2f}\n")

            self.status_label.config(text="Comparison complete")
            messagebox.showinfo("Success", "Comparison complete! Check the Pro Comparison tab.")

        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {e}")
            self.status_label.config(text="Ready")

    def generate_coaching_feedback(self):
        """Generate AI coaching feedback."""
        if not self.comparison_results:
            messagebox.showwarning("Warning", "Run professional comparison first")
            return

        self.status_label.config(text="Generating coaching feedback...")
        self.root.update()

        try:
            focus_areas = self.coaching_focus_var.get()
            improvement_areas = self.comparison_results['improvement_areas']

            feedback = self.gemini_coach.generate_feedback(
                self.comparison_results['avg_differences'],
                improvement_areas,
                focus_areas
            )

            self.coaching_feedback = feedback

            # Display feedback
            self.coaching_text.delete(1.0, tk.END)
            self.coaching_text.insert(tk.END, feedback)

            self.status_label.config(text="Coaching feedback generated")
            messagebox.showinfo("Success", "Coaching feedback generated! Check the AI Coaching tab.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate feedback: {e}")
            self.status_label.config(text="Ready")

    def export_coaching_report(self):
        """Export coaching report to file."""
        if not self.coaching_feedback:
            messagebox.showwarning("Warning", "Generate coaching feedback first")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Coaching Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                f.write("Table Tennis Coaching Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Date: {Path(file_path).stem}\n")
                f.write(f"Video: {self.video_path}\n\n")

                if self.comparison_results:
                    f.write(f"Compared with: {self.comparison_results['professional_player']}\n\n")

                f.write(self.coaching_feedback)

            messagebox.showinfo("Success", f"Report exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")

    def save_session(self):
        """Save current training session."""
        if not self.processed_data:
            messagebox.showwarning("Warning", "No data to save")
            return

        try:
            # Build session data
            session_data = {
                'video_path': self.video_path,
                'ball_tracking': {
                    'frames_analyzed': len(self.processed_data),
                    'avg_spin': sum(d.rps for d in self.processed_data if d.rps) / len(self.processed_data) if self.processed_data else 0
                },
                'pose_tracking': {
                    'poses_detected': sum(1 for d in self.processed_data if d.pose_keypoints)
                }
            }

            if self.comparison_results:
                session_data['comparison'] = self.comparison_results

            if self.coaching_feedback:
                session_data['coaching_feedback'] = self.coaching_feedback

            # Save session
            session_id = self.session_manager.save_session(session_data)
            messagebox.showinfo("Success", f"Session saved: {session_id}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save session: {e}")

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
