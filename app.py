import sys
import numpy as np
import cv2
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QComboBox, QLabel, 
                               QSlider, QFileDialog, QStatusBar, QCheckBox, QGroupBox)
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QImage, QPainter, QColor, QFont

class FluidSolver:
    def __init__(self, N=128):
        self.N = N
        self.size = (N + 2, N + 2)
        
        self.u = np.zeros(self.size, dtype=np.float32)
        self.v = np.zeros(self.size, dtype=np.float32)
        self.u_prev = np.zeros(self.size, dtype=np.float32)
        self.v_prev = np.zeros(self.size, dtype=np.float32)
        
        self.dens = np.zeros((self.size[0], self.size[1], 3), dtype=np.float32)
        self.dens_prev = np.zeros((self.size[0], self.size[1], 3), dtype=np.float32)
        self.curl = np.zeros(self.size, dtype=np.float32)
        
        self.viscosity = 0.0001
        self.diffusion = 0.0001
        self.dt = 0.1
        self.fade = 0.985
        self.vorticity_strength = 0.5 

    def set_bnd(self, b, x):
        if x.ndim == 2:
            x[0, 1:-1] = -x[1, 1:-1] if b == 1 else x[1, 1:-1]
            x[-1, 1:-1] = -x[-2, 1:-1] if b == 1 else x[-2, 1:-1]
            x[1:-1, 0] = -x[1:-1, 1] if b == 2 else x[1:-1, 1]
            x[1:-1, -1] = -x[1:-1, -2] if b == 2 else x[1:-1, -2]
        else:
            x[0, 1:-1, :] = -x[1, 1:-1, :] if b == 1 else x[1, 1:-1, :]
            x[-1, 1:-1, :] = -x[-2, 1:-1, :] if b == 1 else x[-2, 1:-1, :]
            x[1:-1, 0, :] = -x[1:-1, 1, :] if b == 2 else x[1:-1, 1, :]
            x[1:-1, -1, :] = -x[1:-1, -2, :] if b == 2 else x[1:-1, -2, :]

    def diffuse(self, b, x, x0, diff):
        a = self.dt * diff * self.N * self.N
        for _ in range(4):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])) / (1 + 4 * a)
            self.set_bnd(b, x)

    def advect(self, b, d, d0, u, v):
        dt0 = self.dt * self.N
        i, j = np.meshgrid(np.arange(1, self.N + 1), np.arange(1, self.N + 1), indexing='ij')
        x = np.clip(i - dt0 * u[1:-1, 1:-1], 0.5, self.N + 0.5)
        y = np.clip(j - dt0 * v[1:-1, 1:-1], 0.5, self.N + 0.5)
        
        i0 = x.astype(np.int32); i1 = i0 + 1
        j0 = y.astype(np.int32); j1 = j0 + 1
        
        s1 = x - i0; s0 = 1 - s1
        t1 = y - j0; t0 = 1 - t1
        
        d[1:-1, 1:-1] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + \
                        s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
        self.set_bnd(b, d)

    def project(self, u, v, p, div):
        h = 1.0 / self.N
        div[1:-1, 1:-1] = -0.5 * h * (u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2])
        p[1:-1, 1:-1] = 0
        self.set_bnd(0, div)
        self.set_bnd(0, p)
        
        for _ in range(6):
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
            self.set_bnd(0, p)
            
        u[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        self.set_bnd(1, u)
        self.set_bnd(2, v)

    def vorticity_confinement(self):
        self.curl[1:-1, 1:-1] = 0.5 * (self.v[2:, 1:-1] - self.v[:-2, 1:-1] - self.u[1:-1, 2:] + self.u[1:-1, :-2])
        if self.vorticity_strength == 0: return
        
        grad_w_x = np.zeros(self.size, dtype=np.float32)
        grad_w_y = np.zeros(self.size, dtype=np.float32)
        grad_w_x[1:-1, 1:-1] = 0.5 * (np.abs(self.curl[2:, 1:-1]) - np.abs(self.curl[:-2, 1:-1]))
        grad_w_y[1:-1, 1:-1] = 0.5 * (np.abs(self.curl[1:-1, 2:]) - np.abs(self.curl[1:-1, :-2]))
        
        mag = np.sqrt(grad_w_x**2 + grad_w_y**2) + 1e-5
        N_x = grad_w_x / mag
        N_y = grad_w_y / mag
        
        force_x = self.vorticity_strength * (N_y * self.curl)
        force_y = self.vorticity_strength * (-N_x * self.curl)
        
        self.u += self.dt * force_x
        self.v += self.dt * force_y

    def step(self):
        self.u += self.u_prev * self.dt
        self.v += self.v_prev * self.dt
        
        self.u_prev, self.u = self.u, self.u_prev
        self.diffuse(1, self.u, self.u_prev, self.viscosity)
        self.v_prev, self.v = self.v, self.v_prev
        self.diffuse(2, self.v, self.v_prev, self.viscosity)
        
        self.project(self.u, self.v, self.u_prev, self.v_prev)
        
        self.u_prev, self.u = self.u, self.u_prev
        self.v_prev, self.v = self.v, self.v_prev
        self.advect(1, self.u, self.u_prev, self.u_prev, self.v_prev)
        self.advect(2, self.v, self.v_prev, self.u_prev, self.v_prev)
        
        self.vorticity_confinement()
        self.project(self.u, self.v, self.u_prev, self.v_prev)
        
        self.dens += self.dens_prev * self.dt
        self.dens_prev, self.dens = self.dens, self.dens_prev
        self.diffuse(0, self.dens, self.dens_prev, self.diffusion)
        self.dens_prev, self.dens = self.dens, self.dens_prev
        self.advect(0, self.dens, self.dens_prev, self.u, self.v)
        
        self.dens *= self.fade
        self.u_prev.fill(0)
        self.v_prev.fill(0)
        self.dens_prev.fill(0)


class FluidCanvas(QWidget):
    def __init__(self, width=1280, height=720, grid_size=128):
        super().__init__()
        self.sim_width = width
        self.sim_height = height
        self.setFixedSize(width, height)
        self.setStyleSheet("background-color: black;")

        self.set_solver(grid_size)
        self.mouse_down = False
        self.mouse_pos = QPoint(0, 0)
        self.last_mouse_pos = QPoint(0, 0)
        self.setMouseTracking(True)

        self.recording = False
        self.video_writer = None
        
        self.render_mode = 0
        self.bloom_strength = 0.6
        self.vignette_enabled = True
        self.chromatic_aberration = True
        self.inject_color = [255, 50, 50]
        self.brush_size = 3
        
        self.emitter_active = False
        self.emitter_x = 64
        self.emitter_dir = 1
        
        # FPS calculation
        self.last_frame_time = time.time()
        self.fps = 0.0

    def set_solver(self, N):
        self.solver = FluidSolver(N=N)
        self.emitter_x = N // 2

    def add_dye(self, gx, gy, vx, vy, color, radius=3, strength=150.0):
        x0, x1 = max(1, gx - radius), min(self.solver.N + 1, gx + radius)
        y0, y1 = max(1, gy - radius), min(self.solver.N + 1, gy + radius)
        
        self.solver.u_prev[y0:y1, x0:x1] += vx
        self.solver.v_prev[y0:y1, x0:x1] += vy
        self.solver.dens_prev[y0:y1, x0:x1, 0] += color[0] * (strength/255.0)
        self.solver.dens_prev[y0:y1, x0:x1, 1] += color[1] * (strength/255.0)
        self.solver.dens_prev[y0:y1, x0:x1, 2] += color[2] * (strength/255.0)

    def trigger_impulse(self):
        # 360 degree explosion
        cx, cy = self.solver.N // 2, self.solver.N // 2
        for angle in range(0, 360, 15):
            rad = np.radians(angle)
            vx, vy = np.cos(rad) * 15.0, np.sin(rad) * 15.0
            self.add_dye(cx, cy, vx, vy, self.inject_color, radius=5, strength=300.0)

    def update_physics(self):
        if self.emitter_active:
            self.emitter_x += self.emitter_dir * 2
            if self.emitter_x <= 5 or self.emitter_x >= self.solver.N - 5: self.emitter_dir *= -1
            self.add_dye(self.emitter_x, 20, 0, 8, self.inject_color, radius=4, strength=200.0)

        if self.mouse_down:
            gx = int(self.mouse_pos.x() / self.sim_width * self.solver.N)
            gy = int(self.mouse_pos.y() / self.sim_height * self.solver.N)
            vx = (self.mouse_pos.x() - self.last_mouse_pos.x()) * 2.0
            vy = (self.mouse_pos.y() - self.last_mouse_pos.y()) * 2.0
            self.add_dye(gx, gy, vx, vy, self.inject_color, radius=self.brush_size)

        self.last_mouse_pos = self.mouse_pos
        self.solver.step()

    def render_frame(self):
        if self.render_mode == 0:
            frame = np.clip(self.solver.dens, 0, 255).astype(np.uint8)
        elif self.render_mode == 1:
            speed = np.sqrt(self.solver.u**2 + self.solver.v**2)
            norm = np.clip(speed * 20, 0, 255).astype(np.uint8)
            frame = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
        elif self.render_mode == 2:
            norm = np.clip(np.abs(self.solver.curl) * 50, 0, 255).astype(np.uint8)
            frame = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)
            
        # Scale up to window resolution
        frame = cv2.resize(frame, (self.sim_width, self.sim_height), interpolation=cv2.INTER_CUBIC)
        
        # 1. Chromatic Aberration (Split RGB channels at edges)
        if self.chromatic_aberration and self.render_mode == 0:
            b, g, r = cv2.split(frame)
            shift = 5
            r_shifted = np.roll(r, shift, axis=1)
            b_shifted = np.roll(b, -shift, axis=1)
            frame = cv2.merge([b_shifted, g, r_shifted])
            # Clean up wrap-around artifacts
            frame[:, :shift] = 0
            frame[:, -shift:] = 0
        
        # 2. Bloom / Glow Effect
        if self.bloom_strength > 0:
            bright_mask = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)[1]
            glow = cv2.GaussianBlur(bright_mask, (21, 21), 0)
            frame = cv2.addWeighted(frame, 1.0, glow, self.bloom_strength, 0)
            
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 3. Vignette (Darken corners)
        if self.vignette_enabled:
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (w//2, h//2), int(min(w,h)*0.75), (1,1,1), thickness=-1)
            mask = cv2.GaussianBlur(mask, (w//2|1, h//2|1), 0) # Must be odd
            frame = (frame * mask[:, :, np.newaxis]).astype(np.uint8)
            
        return frame

    def paintEvent(self, event):
        frame_bgr = self.render_frame()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        
        painter = QPainter(self)
        painter.drawImage(0, 0, q_img)
        
        # FPS Counter Overlay
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        current_fps = 1.0 / dt if dt > 0 else 0
        self.fps = self.fps * 0.9 + current_fps * 0.1 # Smooth FPS
        
        font = QFont("Consolas", 10)
        painter.setFont(font)
        painter.setPen(QColor(0, 255, 0, 180))
        painter.drawText(10, 20, f"FPS: {self.fps:.1f}")
        painter.setPen(QColor(255, 255, 255, 120))
        painter.drawText(10, 40, f"Grid: {self.solver.N}x{self.solver.N}")

        if self.recording:
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(self.width() - 30, 10, 15, 15)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_down = True
            self.mouse_pos = event.position().toPoint()
            self.last_mouse_pos = self.mouse_pos

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_down = False

    def mouseMoveEvent(self, event):
        self.last_mouse_pos = self.mouse_pos
        self.mouse_pos = event.position().toPoint()

    def toggle_recording(self, filename, fps=30):
        if not self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (self.sim_width, self.sim_height))
            self.recording = True
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False

    def write_frame(self):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(self.render_frame())

    def clear(self):
        self.solver.u.fill(0)
        self.solver.v.fill(0)
        self.solver.dens.fill(0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cinematic Fluid Studio - YouTube & Shorts Creator")
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { color: #cdd6f4; font-family: 'Segoe UI', Arial; }
            QPushButton { background-color: #313244; border: 1px solid #45475a; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #45475a; }
            QComboBox { background-color: #313244; border: 1px solid #45475a; padding: 5px; border-radius: 4px; }
            QSlider::groove:horizontal { border: 1px solid #45475a; height: 4px; background: #181825; }
            QSlider::handle:horizontal { background: #89b4fa; width: 14px; margin: -6px 0; border-radius: 7px; }
            QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; padding-top: 10px; font-weight: bold; color: #89b4fa; }
            QCheckBox { spacing: 8px; }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)

        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignTop)
        
        # Setup Group
        grp_setup = QGroupBox("Studio Setup")
        l1 = QVBoxLayout(); grp_setup.setLayout(l1)
        
        l1.addWidget(QLabel("Aspect Ratio:"))
        self.combo_aspect = QComboBox()
        self.combo_aspect.addItems(["16:9 (1280x720) YouTube", "9:16 (720x1280) Shorts", "1:1 (1080x1080)"])
        self.combo_aspect.currentIndexChanged.connect(self.change_aspect_ratio)
        l1.addWidget(self.combo_aspect)

        l1.addWidget(QLabel("Grid Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(["128x128 (Fast)", "256x256 (Sharp)", "384x384 (Ultra)"])
        self.combo_res.currentIndexChanged.connect(self.change_resolution)
        l1.addWidget(self.combo_res)
        controls_layout.addWidget(grp_setup)

        # Visuals Group
        grp_vis = QGroupBox("Cinematic Visuals")
        l2 = QVBoxLayout(); grp_vis.setLayout(l2)
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Render Mode: Dye (Color)", "Render Mode: Speed", "Render Mode: Vorticity"])
        self.combo_mode.currentIndexChanged.connect(lambda i: setattr(self.canvas, 'render_mode', i))
        l2.addWidget(self.combo_mode)
        
        self.chk_bloom = QCheckBox("Enable Bloom (Glow)")
        self.chk_bloom.setChecked(True)
        self.chk_bloom.stateChanged.connect(lambda s: setattr(self.canvas, 'bloom_strength', 0.6 if s else 0.0))
        l2.addWidget(self.chk_bloom)
        
        self.lbl_bloom = QLabel("Bloom Strength: 0.6")
        l2.addWidget(self.lbl_bloom)
        self.slider_bloom = QSlider(Qt.Horizontal)
        self.slider_bloom.setMinimum(0); self.slider_bloom.setMaximum(100)
        self.slider_bloom.setValue(60)
        self.slider_bloom.valueChanged.connect(self.update_bloom)
        l2.addWidget(self.slider_bloom)
        
        self.chk_vignette = QCheckBox("Enable Vignette")
        self.chk_vignette.setChecked(True)
        self.chk_vignette.stateChanged.connect(lambda s: setattr(self.canvas, 'vignette_enabled', bool(s)))
        l2.addWidget(self.chk_vignette)
        
        self.chk_ca = QCheckBox("Enable Chromatic Aberration")
        self.chk_ca.setChecked(True)
        self.chk_ca.stateChanged.connect(lambda s: setattr(self.canvas, 'chromatic_aberration', bool(s)))
        l2.addWidget(self.chk_ca)
        controls_layout.addWidget(grp_vis)

        # Brushes & Color
        grp_brush = QGroupBox("Brushes & Color")
        l3 = QVBoxLayout(); grp_brush.setLayout(l3)
        
        self.lbl_color = QLabel("Inject Color: Red")
        l3.addWidget(self.lbl_color)
        self.slider_color = QSlider(Qt.Horizontal)
        self.slider_color.setMinimum(0); self.slider_color.setMaximum(5)
        self.slider_color.valueChanged.connect(self.change_color)
        l3.addWidget(self.slider_color)
        
        self.lbl_brush = QLabel("Brush Size: 3")
        l3.addWidget(self.lbl_brush)
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setMinimum(1); self.slider_brush.setMaximum(10)
        self.slider_brush.setValue(3)
        self.slider_brush.valueChanged.connect(self.update_brush)
        l3.addWidget(self.slider_brush)
        
        self.btn_impulse = QPushButton("Trigger Impulse (Explosion)")
        self.btn_impulse.setStyleSheet("background-color: #f9e2af; color: #1e1e2e; font-weight: bold;")
        self.btn_impulse.clicked.connect(self.canvas.trigger_impulse)
        l3.addWidget(self.btn_impulse)
        controls_layout.addWidget(grp_brush)

        # Physics Group
        grp_phys = QGroupBox("Fluid Physics")
        l4 = QVBoxLayout(); grp_phys.setLayout(l4)
        self.lbl_vort = QLabel("Vorticity Swirl: 0.5")
        l4.addWidget(self.lbl_vort)
        self.slider_vort = QSlider(Qt.Horizontal)
        self.slider_vort.setMinimum(0); self.slider_vort.setMaximum(100)
        self.slider_vort.setValue(50)
        self.slider_vort.valueChanged.connect(self.update_vorticity)
        l4.addWidget(self.slider_vort)

        self.lbl_fade = QLabel("Fade Speed: 0.985")
        l4.addWidget(self.lbl_fade)
        self.slider_fade = QSlider(Qt.Horizontal)
        self.slider_fade.setMinimum(80); self.slider_fade.setMaximum(100)
        self.slider_fade.setValue(98)
        self.slider_fade.valueChanged.connect(self.update_fade)
        l4.addWidget(self.slider_fade)
        controls_layout.addWidget(grp_phys)

        # Recording Group
        grp_rec = QGroupBox("Automation & Recording")
        l5 = QVBoxLayout(); grp_rec.setLayout(l5)
        self.chk_emitter = QCheckBox("Auto Emitter (Faucet)")
        self.chk_emitter.stateChanged.connect(lambda s: setattr(self.canvas, 'emitter_active', bool(s)))
        l5.addWidget(self.chk_emitter)

        self.btn_clear = QPushButton("Clear Canvas")
        self.btn_clear.clicked.connect(self.canvas.clear)
        l5.addWidget(self.btn_clear)

        self.btn_record = QPushButton("Start Recording (MP4)")
        self.btn_record.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold; padding: 12px;")
        self.btn_record.clicked.connect(self.toggle_record)
        l5.addWidget(self.btn_record)
        controls_layout.addWidget(grp_rec)

        main_layout.addLayout(controls_layout, 1)

        # Canvas
        self.canvas = FluidCanvas(1280, 720, grid_size=128)
        main_layout.addWidget(self.canvas, 3)

        # Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("background-color: #181825; color: #a6adc8;")
        self.status.showMessage("Ready. Drag on canvas to create fluid!")

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_sim)
        self.timer.start(16)

        self.recording_filename = ""
        self.record_start_time = 0

    def update_sim(self):
        self.canvas.update_physics()
        self.canvas.update()
        if self.canvas.recording:
            self.canvas.write_frame()
            elapsed = time.time() - self.record_start_time
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            self.status.showMessage(f"● REC {self.recording_filename}  |  Time: {h:02d}:{m:02d}:{s:02d}")

    def update_bloom(self, val):
        v = val / 100.0
        self.canvas.bloom_strength = v
        self.lbl_bloom.setText(f"Bloom Strength: {v:.2f}")

    def update_brush(self, val):
        self.canvas.brush_size = val
        self.lbl_brush.setText(f"Brush Size: {val}")

    def change_color(self, idx):
        colors = {"Red": [255, 50, 50], "Green": [50, 255, 50], "Blue": [50, 50, 255], "Cyan": [50, 255, 255], "Magenta": [255, 50, 255], "Yellow": [255, 255, 50]}
        color_names = list(colors.keys())
        self.canvas.inject_color = colors[color_names[idx]]
        self.lbl_color.setText(f"Inject Color: {color_names[idx]}")

    def update_vorticity(self, val):
        v = val / 100.0
        self.canvas.solver.vorticity_strength = v
        self.lbl_vort.setText(f"Vorticity Swirl: {v:.2f}")

    def update_fade(self, val):
        f = val / 100.0
        self.canvas.solver.fade = f
        self.lbl_fade.setText(f"Fade Speed: {f:.3f}")

    def change_resolution(self, idx):
        N = [128, 256, 384][idx]
        self.canvas.set_solver(N)

    def change_aspect_ratio(self, index):
        self.canvas.clear()
        if index == 0: self.canvas.sim_width, self.canvas.sim_height = 1280, 720
        elif index == 1: self.canvas.sim_width, self.canvas.sim_height = 720, 1280
        elif index == 2: self.canvas.sim_width, self.canvas.sim_height = 1080, 1080
        self.canvas.setFixedSize(self.canvas.sim_width, self.canvas.sim_height)
        self.adjustSize()

    def toggle_record(self):
        if not self.canvas.recording:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Video", "cinematic_fluid.mp4", "MP4 Files (*.mp4)")
            if filename:
                if not filename.endswith('.mp4'): filename += '.mp4'
                self.recording_filename = filename
                self.canvas.toggle_recording(filename, fps=30)
                self.record_start_time = time.time()
                self.btn_record.setText("Stop Recording")
                self.btn_record.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; padding: 12px;")
        else:
            self.canvas.toggle_recording("", fps=30)
            self.btn_record.setText("Start Recording (MP4)")
            self.btn_record.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold; padding: 12px;")
            self.status.showMessage(f"Saved to {self.recording_filename}")

    def closeEvent(self, event):
        if self.canvas.recording: self.canvas.toggle_recording("", fps=30)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())