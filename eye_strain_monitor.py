import csv
import os
import platform
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

# Keep runtime console cleaner during demo runs; does not affect detection logic.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass


# Left and right eye landmark indexes for MediaPipe Face Mesh.
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


@dataclass
class Config:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 360
    process_every_n_frames: int = 2
    calibration_seconds: int = 10
    min_blink_frames: int = 2
    max_blink_duration_sec: float = 0.8
    long_closure_warn_sec: float = 1.1
    blink_window_sec: int = 60
    log_interval_sec: float = 2.0
    break_reminder_minutes: int = 20
    target_blink_rate_min: int = 10
    target_blink_rate_max: int = 22
    ideal_face_area_min: float = 0.06
    ideal_face_area_max: float = 0.22


class EyeStrainMonitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session_start = time.time()
        self.last_log_time = 0.0
        self.last_beep_time = 0.0
        self.calibration_start = time.time()

        self.calibrating = True
        self.calibration_ears = []
        self.baseline_ear = 0.26
        self.ear_threshold = 0.21
        self.personal_target_blink_rate = 14

        self.blink_count = 0
        self.blink_timestamps = deque()
        self.eye_closure_durations = deque(maxlen=120)

        self.eyes_closed = False
        self.closed_start = None
        self.closed_frame_count = 0

        self.break_alert_sent = False

        self._init_face_mesh()
        self._init_logger()

    def _init_face_mesh(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _init_logger(self):
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("logs", f"session_{ts}.csv")
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "session_seconds",
                    "ear",
                    "blink_count",
                    "blink_rate_per_min",
                    "closure_duration_sec",
                    "fatigue_score",
                    "eye_health_score",
                    "distance_label",
                    "face_area_ratio",
                    "alert",
                ]
            )

    @staticmethod
    def _distance(p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def compute_ear(self, landmarks, eye_idx, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx]
        p1, p2, p3, p4, p5, p6 = pts

        vert_1 = self._distance(p2, p6)
        vert_2 = self._distance(p3, p5)
        horiz = self._distance(p1, p4)

        if horiz < 1e-6:
            return 0.0
        return (vert_1 + vert_2) / (2.0 * horiz)

    def compute_face_area_ratio(self, landmarks, frame_w, frame_h):
        xs = [lm.x * frame_w for lm in landmarks]
        ys = [lm.y * frame_h for lm in landmarks]

        x_min, x_max = max(0, min(xs)), min(frame_w, max(xs))
        y_min, y_max = max(0, min(ys)), min(frame_h, max(ys))

        face_w = max(1.0, x_max - x_min)
        face_h = max(1.0, y_max - y_min)
        area_ratio = (face_w * face_h) / float(frame_w * frame_h)

        return area_ratio, (int(x_min), int(y_min), int(x_max), int(y_max))

    def classify_distance(self, area_ratio):
        if area_ratio < self.cfg.ideal_face_area_min:
            return "Far"
        if area_ratio > self.cfg.ideal_face_area_max:
            return "Too Close"
        return "Ideal"

    def update_calibration(self, ear):
        elapsed = time.time() - self.calibration_start
        if elapsed <= self.cfg.calibration_seconds:
            self.calibration_ears.append(ear)
            return

        if self.calibrating:
            arr = np.array(self.calibration_ears, dtype=np.float32)
            if arr.size > 20:
                mean_ear = float(np.mean(arr))
                std_ear = float(np.std(arr))
                self.baseline_ear = float(np.clip(mean_ear, 0.2, 0.4))
                adaptive_drop = max(0.035, std_ear * 0.8)
                self.ear_threshold = float(np.clip(self.baseline_ear - adaptive_drop, 0.16, 0.3))
            else:
                self.baseline_ear = 0.26
                self.ear_threshold = 0.21

            # Estimate personal blink target from calibration; fall back to default range.
            estimated_rate = max(0, len(self.blink_timestamps)) * (60.0 / max(1.0, self.cfg.calibration_seconds))
            estimated_rate = int(round(estimated_rate))
            if estimated_rate <= 0:
                estimated_rate = 14
            self.personal_target_blink_rate = int(np.clip(estimated_rate, self.cfg.target_blink_rate_min, self.cfg.target_blink_rate_max))
            self.calibrating = False

    def update_blinks(self, ear):
        now = time.time()
        closure_duration = 0.0

        if ear < self.ear_threshold:
            self.closed_frame_count += 1
            if not self.eyes_closed and self.closed_frame_count >= self.cfg.min_blink_frames:
                self.eyes_closed = True
                self.closed_start = now
        else:
            if self.eyes_closed and self.closed_start is not None:
                closure_duration = now - self.closed_start
                if 0.05 <= closure_duration <= self.cfg.max_blink_duration_sec:
                    self.blink_count += 1
                    self.blink_timestamps.append(now)
                    self.eye_closure_durations.append(closure_duration)
                elif closure_duration > self.cfg.max_blink_duration_sec:
                    self.eye_closure_durations.append(closure_duration)

            self.eyes_closed = False
            self.closed_start = None
            self.closed_frame_count = 0

        # Keep only last minute blinks for rate.
        while self.blink_timestamps and (now - self.blink_timestamps[0] > self.cfg.blink_window_sec):
            self.blink_timestamps.popleft()

        current_closure = 0.0
        if self.eyes_closed and self.closed_start is not None:
            current_closure = now - self.closed_start

        return closure_duration, current_closure

    def blink_rate_per_min(self):
        return int(round(len(self.blink_timestamps) * 60.0 / self.cfg.blink_window_sec))

    def compute_fatigue_score(self, blink_rate, current_closure, session_seconds, distance_label):
        # 1) Low blink rate penalty (0-40)
        target = self.personal_target_blink_rate
        low_blink_penalty = 0.0
        if blink_rate < target:
            low_blink_penalty = min(40.0, (target - blink_rate) / max(1.0, target) * 40.0)

        # 2) Long closure penalty (0-35)
        avg_closure = float(np.mean(self.eye_closure_durations)) if self.eye_closure_durations else 0.0
        long_closure_value = max(current_closure, avg_closure)
        long_closure_penalty = 0.0
        if long_closure_value > 0.18:
            long_closure_penalty = min(35.0, (long_closure_value - 0.18) / 1.2 * 35.0)

        # 3) Continuous screen-time penalty (0-25), starts after 10 minutes.
        mins = session_seconds / 60.0
        screen_penalty = 0.0
        if mins > 10:
            screen_penalty = min(25.0, (mins - 10.0) / 35.0 * 25.0)

        # 4) Distance penalty (0-8) for too close/far posture.
        distance_penalty = 0.0
        if distance_label != "Ideal":
            distance_penalty = 8.0

        fatigue = low_blink_penalty + long_closure_penalty + screen_penalty + distance_penalty
        fatigue = int(np.clip(round(fatigue), 0, 100))
        eye_health = int(np.clip(100 - fatigue, 0, 100))
        return fatigue, eye_health

    def alert_text(self, blink_rate, current_closure, session_seconds, distance_label):
        reminders = []

        if current_closure > self.cfg.long_closure_warn_sec:
            reminders.append("Eyes closed too long!")

        if session_seconds > 60 and blink_rate < max(8, self.personal_target_blink_rate - 5):
            reminders.append("Blink more!")

        break_secs = self.cfg.break_reminder_minutes * 60
        if session_seconds >= break_secs and not self.break_alert_sent:
            reminders.append("Take a break!")
            self.break_alert_sent = True

        if distance_label == "Too Close":
            reminders.append("Move farther from screen")
        elif distance_label == "Far":
            reminders.append("Move a bit closer")

        if not reminders and self.calibrating:
            return "Calibrating baseline... keep natural gaze"
        if not reminders:
            return "Good posture. Keep going."
        return " | ".join(reminders)

    def maybe_beep(self, should_beep):
        if not should_beep:
            return
        now = time.time()
        if now - self.last_beep_time < 2.0:
            return

        self.last_beep_time = now
        if platform.system().lower().startswith("win"):
            try:
                import winsound

                winsound.Beep(1300, 180)
            except Exception:
                pass

    def log_metrics(self, ear, blink_rate, closure_duration, fatigue, eye_health, distance_label, area_ratio, alert):
        now = time.time()
        if now - self.last_log_time < self.cfg.log_interval_sec:
            return

        self.last_log_time = now
        session_seconds = now - self.session_start
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    f"{session_seconds:.1f}",
                    f"{ear:.4f}",
                    self.blink_count,
                    blink_rate,
                    f"{closure_duration:.3f}",
                    fatigue,
                    eye_health,
                    distance_label,
                    f"{area_ratio:.4f}",
                    alert,
                ]
            )

    @staticmethod
    def format_time(total_seconds):
        total_seconds = int(total_seconds)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def draw_dashboard(self, frame, metrics, eye_points, face_box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_box

        # Face box and eye landmarks.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 90), 2)
        for p in eye_points:
            cv2.circle(frame, p, 2, (80, 200, 255), -1)

        # Semi-transparent dashboard panel.
        panel_w = 360
        panel_h = 250
        panel_x = 12
        panel_y = 12

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 20, 20),
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        title_color = (0, 220, 255)
        text_color = (235, 235, 235)
        warn_color = (0, 90, 255)

        cv2.putText(frame, "Eye Strain Monitor", (panel_x + 14, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, title_color, 2)

        y = panel_y + 62
        step = 28
        cv2.putText(frame, f"Blinks: {metrics['blink_count']}", (panel_x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)
        y += step
        cv2.putText(frame, f"Blink Rate: {metrics['blink_rate']} /min", (panel_x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)
        y += step
        cv2.putText(frame, f"Fatigue Score: {metrics['fatigue']} /100", (panel_x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)
        y += step
        cv2.putText(frame, f"Eye Health: {metrics['eye_health']} /100", (panel_x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)
        y += step
        cv2.putText(frame, f"Timer: {metrics['timer']}", (panel_x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)
        y += step
        cv2.putText(frame, f"Distance: {metrics['distance']}", (panel_x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 2)

        # Fatigue progress bar.
        bar_x, bar_y = panel_x + 14, panel_y + panel_h - 52
        bar_w, bar_h = panel_w - 28, 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        filled = int(bar_w * metrics["fatigue"] / 100.0)
        bar_color = (70, 210, 80) if metrics["fatigue"] < 40 else ((0, 190, 255) if metrics["fatigue"] < 70 else (0, 80, 255))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)

        # Alert strip.
        alert_box_y = h - 46
        cv2.rectangle(frame, (0, alert_box_y), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, metrics["alert"], (18, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.62, warn_color if "!" in metrics["alert"] else text_color, 2)

        return frame

    def _open_camera(self):
        preferred = [self.cfg.camera_index, 1, 2]
        camera_indices = []
        for idx in preferred:
            if idx not in camera_indices:
                camera_indices.append(idx)

        dshow = getattr(cv2, "CAP_DSHOW", None)
        msmf = getattr(cv2, "CAP_MSMF", None)
        backends = [dshow, msmf, None]

        for cam_idx in camera_indices:
            for backend in backends:
                if backend is None:
                    cap = cv2.VideoCapture(cam_idx)
                    backend_name = "DEFAULT"
                else:
                    cap = cv2.VideoCapture(cam_idx, backend)
                    backend_name = "DSHOW" if backend == dshow else "MSMF"

                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
                    cap.set(cv2.CAP_PROP_FPS, 24)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap, cam_idx, backend_name

                cap.release()

        return None, -1, "NONE"

    def run(self):
        cap, used_camera_idx, used_backend = self._open_camera()

        if cap is None:
            print("ERROR: Could not access webcam.")
            print("Fix: close Zoom/Meet/Camera app, then retry.")
            print("Fix: change camera_index in Config from 0 to 1 or 2.")
            return

        print("Press 'q' to quit.")
        print(f"Session log: {self.log_path}")
        print(f"Camera: index={used_camera_idx}, backend={used_backend}")

        frame_idx = 0
        cached_results = None
        consecutive_read_failures = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_read_failures += 1
                if consecutive_read_failures > 60:
                    print("ERROR: Webcam read failed repeatedly. Restarting app is recommended.")
                    break
                continue
            consecutive_read_failures = 0

            frame = cv2.flip(frame, 1)
            frame_idx += 1
            if frame_idx % max(1, self.cfg.process_every_n_frames) == 0 or cached_results is None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cached_results = self.face_mesh.process(rgb)
            results = cached_results

            now = time.time()
            session_seconds = now - self.session_start

            ear_avg = 0.0
            blink_rate = self.blink_rate_per_min()
            closure_duration = 0.0
            current_closure = 0.0
            fatigue = 0
            eye_health = 100
            distance_label = "No Face"
            area_ratio = 0.0
            alert = "No face detected. Align your face with camera."
            eye_points = []
            face_box = (20, 20, 120, 120)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]

                left_ear = self.compute_ear(face_landmarks, LEFT_EYE_IDX, w, h)
                right_ear = self.compute_ear(face_landmarks, RIGHT_EYE_IDX, w, h)
                ear_avg = (left_ear + right_ear) / 2.0

                self.update_calibration(ear_avg)
                closure_duration, current_closure = self.update_blinks(ear_avg)
                blink_rate = self.blink_rate_per_min()

                area_ratio, face_box = self.compute_face_area_ratio(face_landmarks, w, h)
                distance_label = self.classify_distance(area_ratio)

                fatigue, eye_health = self.compute_fatigue_score(blink_rate, current_closure, session_seconds, distance_label)
                alert = self.alert_text(blink_rate, current_closure, session_seconds, distance_label)

                should_beep = (
                    "Take a break!" in alert
                    or "Blink more!" in alert
                    or "Eyes closed too long!" in alert
                )
                self.maybe_beep(should_beep)

                eye_points = [
                    (int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in (LEFT_EYE_IDX + RIGHT_EYE_IDX)
                ]

            metrics = {
                "blink_count": self.blink_count,
                "blink_rate": blink_rate,
                "fatigue": fatigue,
                "eye_health": eye_health,
                "timer": self.format_time(session_seconds),
                "distance": distance_label,
                "alert": alert,
            }

            frame = self.draw_dashboard(frame, metrics, eye_points, face_box)

            # Top-right calibration and threshold hints.
            top_right_x = frame.shape[1] - 290
            cv2.putText(
                frame,
                f"EAR: {ear_avg:.3f}  TH: {self.ear_threshold:.3f}",
                (top_right_x, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (240, 240, 240),
                2,
            )
            if self.calibrating:
                remain = max(0, self.cfg.calibration_seconds - int(time.time() - self.calibration_start))
                cv2.putText(
                    frame,
                    f"Calibrating... {remain}s",
                    (top_right_x, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (0, 215, 255),
                    2,
                )

            self.log_metrics(
                ear=ear_avg,
                blink_rate=blink_rate,
                closure_duration=max(closure_duration, current_closure),
                fatigue=fatigue,
                eye_health=eye_health,
                distance_label=distance_label,
                area_ratio=area_ratio,
                alert=alert,
            )

            try:
                cv2.imshow("Eye Strain & Blink Rate Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as ex:
                print(f"ERROR: OpenCV display error: {ex}")
                print("Tip: run outside debugger using .venv Python in terminal.")
                break
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    cfg = Config()
    app = EyeStrainMonitor(cfg)
    app.run()


if __name__ == "__main__":
    main()
