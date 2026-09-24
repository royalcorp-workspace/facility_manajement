"""
Modul Pipeline & Camera Orchestrator untuk Facility Management Vision Engine.
Menghubungkan Preprocessor -> MotionGate -> Detector (BoundedSemaphore) -> Tracker -> EventDispatcher -> AlertDispatcher -> Buffer O(1) -> SQLite.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

from engine.config_loader import CameraConfig, ROIZonesConfig
from engine.detector_interface import DetectorBase
from engine.detector_impl import YOLO11nDetector, SyntheticMockDetector
from engine.event_dispatcher import EventDispatcher
from engine.geometry import scale_points
from engine.motion_gate import MotionGate
from engine.preprocessor import ProcessedFrame
from engine.tracker_impl import CentroidTracker
from engine.smart_parking import SmartParkingTracker
from engine.logger import get_logger
from storage.database import get_writer_conn, get_writer_lock
from storage.models import CameraRegistry

logger = get_logger(__name__)

if TYPE_CHECKING:
    from notification.dispatcher import AlertDispatcher, AlertPacket
    from web.buffer import MultiCameraBuffer


class CameraOrchestrator(threading.Thread):
    """
    Thread pengelola pipeline pengolahan visual untuk satu kamera.
    Mendukung proteksi CPU melalui BoundedSemaphore dan off-lock buffer live streaming.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        roi_config: ROIZonesConfig,
        proc_queue: queue.Queue[ProcessedFrame],
        db_path: str | Path = "storage/facility.db",
        detector: Optional[DetectorBase] = None,
        motion_gate: Optional[MotionGate] = None,
        tracker: Optional[CentroidTracker] = None,
        event_dispatcher: Optional[EventDispatcher] = None,
        ai_semaphore: Optional[threading.BoundedSemaphore] = None,
        frame_buffer: Optional[MultiCameraBuffer] = None,
        alert_dispatcher: Optional[AlertDispatcher] = None,
        parking_tracker: Optional[SmartParkingTracker] = None,
        live_jpeg_quality: int = 75,
        camera_index: int = 0,
        total_cameras: int = 1,
        initial_warmup_frames: int = 30,
        quiescent_scan_interval: float = 15.0,
    ) -> None:
        super().__init__(name=f"Orchestrator-{camera_config.camera_id}", daemon=True)
        self.camera_config = camera_config
        self.roi_config = roi_config
        self.proc_queue = proc_queue
        self.db_path = Path(db_path)
        self.ai_semaphore = ai_semaphore
        self.frame_buffer = frame_buffer
        self.alert_dispatcher = alert_dispatcher
        self.live_jpeg_quality = live_jpeg_quality
        self.camera_index = camera_index
        self.total_cameras = max(1, total_cameras)

        self._stop_event = threading.Event()
        self.processed_count: int = 0
        self.motion_detected_count: int = 0
        self.initial_warmup_frames: int = initial_warmup_frames
        self.last_motion_detected_time: float = 0.0
        self._quiescent_scan_interval: float = quiescent_scan_interval
        self.fps: float = 20.0
        self._last_fps_time: float = 0.0
        self._fps_frame_count: int = 0

        # Smart Parking Tracker (Kapasitas Dinamis Berdasarkan Poligon Aktif)
        self.parking_tracker = parking_tracker or SmartParkingTracker(
            dwell_threshold_sec=10.0,
            vehicle_classes={"car", "truck", "bus"},
        )

        # Inisialisasi komponen default jika tidak di-inject
        self.motion_gate = motion_gate or MotionGate(
            pixel_threshold=25,
            min_changed_pixels_pct=0.005,
            heartbeat_interval_sec=self._quiescent_scan_interval,
        )

        self.detector = detector or YOLO11nDetector(
            confidence_threshold=0.25,
            iou_threshold=0.45,
            target_classes=["person", "car", "motorcycle", "bus", "truck", "backpack", "handbag"],
            model_path="weights/yolo11n.onnx",
        )

        self.tracker = tracker or CentroidTracker(
            max_disappeared_frames=30,
            max_distance_px=60.0,
            anchor_radius_px=15.0,
            enter_min_frames=3,
            spatial_memory_ttl_sec=3.0,
        )

        self.event_dispatcher = event_dispatcher or EventDispatcher(
            snapshot_quality=85,
            exit_timeout_sec=3.0,
        )

    @property
    def frame_count(self) -> int:
        """Alias counter frame untuk kompatibilitas."""
        return self.processed_count

    def create_annotated_frame(
        self,
        frame: np.ndarray,
        tracks: List[TrackResult],
        timestamp: float,
        is_warmup: bool = False,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Pembuatan annotated frame:
        1. Auto-scaling koordinat dari 1080p native ke resolusi canvas frame aktif saat ini:
           scale_x = frame_width / 1920.0
           scale_y = frame_height / 1080.0
        2. Evaluasi status okupansi parkir dinamis (Smart Parking)
        3. Render garis petak poligon, tripwire, dan status okupansi (Enterprise IVA Style)
        """
        annotated = frame.copy()
        frame_h, frame_w = annotated.shape[:2]

        ref_w = 1920.0
        ref_h = 1080.0
        if self.camera_config and self.camera_config.resolution:
            ref_w = float(self.camera_config.resolution.capture_width or 1920.0)
            ref_h = float(self.camera_config.resolution.capture_height or 1080.0)

        scale_x = frame_w / ref_w
        scale_y = frame_h / ref_h

        # Update status okupansi parkir dinamis
        parking_stats = self.parking_tracker.update(
            tracks=tracks,
            polygons=self.roi_config.polygons,
            scale_x=scale_x,
            scale_y=scale_y,
            current_time=timestamp,
            is_warmup=is_warmup,
        )

        display_fps = fps if fps is not None else getattr(self, "fps", 20.0)
        cam_id = self.camera_config.camera_id if self.camera_config else "cam_01"
        cam_name = self.camera_config.display_name if self.camera_config else "Koridor Utama"

        # Render seluruh visual kontur interaktif (Modern IVA Look)
        self.parking_tracker.render_overlay(
            canvas=annotated,
            polygons=self.roi_config.polygons,
            tripwires=self.roi_config.tripwires,
            tracks=tracks,
            scale_x=scale_x,
            scale_y=scale_y,
            parking_stats=parking_stats,
            current_time=timestamp,
            fps=display_fps,
            camera_id=cam_id,
            camera_name=cam_name,
        )

        return annotated

    def stop(self) -> None:
        """Sinyal penghentian orchestrator thread."""
        self._stop_event.set()

    def run(self) -> None:
        """Loop utama pemrosesan frame dari proc_queue."""
        db_conn: Optional[sqlite3.Connection] = None

        try:
            db_conn = get_writer_conn()
            writer_lock = get_writer_lock()
            # Update registry status ke 'active'
            registry = CameraRegistry(
                camera_id=self.camera_config.camera_id,
                display_name=self.camera_config.display_name,
            )
            with writer_lock:
                registry.upsert(db_conn)
                registry.update_status(db_conn, "active")

            # Warmup detector
            self.detector.warmup(
                (self.camera_config.resolution.ai_height, self.camera_config.resolution.ai_width, 3)
            )

            while not self._stop_event.is_set():
                try:
                    processed_frame = self.proc_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                self.processed_count += 1
                now_wall = time.time()
                if self._last_fps_time == 0.0:
                    self._last_fps_time = now_wall
                else:
                    self._fps_frame_count += 1
                    fps_dt = now_wall - self._last_fps_time
                    if fps_dt >= 1.0:
                        self.fps = round(self._fps_frame_count / fps_dt, 1)
                        self._fps_frame_count = 0
                        self._last_fps_time = now_wall

                # 1. Motion Gate Check
                should_run_dnn, reason = self.motion_gate.should_process(
                    processed_frame.ai_frame,
                    processed_frame.timestamp,
                )

                # Evaluasi Initial Cold-Start Scan & Periodic Quiescent Sanity Scan
                is_warmup = getattr(processed_frame, "is_warmup", False) or (self.processed_count <= self.initial_warmup_frames)
                force_full_inference = False

                if is_warmup:
                    # Cold-start warmup: bypass motion gating secara mutlak dan paksa inferensi YOLO penuh
                    force_full_inference = True
                    self.last_motion_detected_time = processed_frame.timestamp
                else:
                    # Evaluasi gerakan normal
                    if should_run_dnn and reason == "motion":
                        self.last_motion_detected_time = processed_frame.timestamp
                    elif self.last_motion_detected_time > 0:
                        # Cek quiescent sanity scan (jika tidak ada gerakan selama 15 detik berturut-turut)
                        if (processed_frame.timestamp - self.last_motion_detected_time) >= self._quiescent_scan_interval:
                            force_full_inference = True
                            self.last_motion_detected_time = processed_frame.timestamp

                # Evaluasi giliran inferensi (Staggered AI Cadence saat multi-kamera)
                is_turn = (self.total_cameras <= 1) or (
                    (self.processed_count + self.camera_index) % self.total_cameras == 0
                )

                effective_should_run = should_run_dnn or force_full_inference
                detections = []
                is_coasting = False

                if effective_should_run:
                    if should_run_dnn:
                        self.motion_detected_count += 1

                    if is_turn or force_full_inference:
                        # PROTEKSI CPU: Eksekusi inferensi DNN dibatasi oleh Global BoundedSemaphore
                        if self.ai_semaphore is not None:
                            with self.ai_semaphore:
                                detections = self.detector.detect(processed_frame.ai_frame)
                        else:
                            detections = self.detector.detect(processed_frame.ai_frame)
                    else:
                        # Frame coasting: lewati DNN untuk memberi giliran kamera lain
                        is_coasting = True
                else:
                    # Frame tanpa gerakan & bukan sanity scan: masuk mode coasting agar track aktif tidak terhapus
                    is_coasting = True

                # 2. Update Tracker (Deteksi baru vs Coasting)
                if is_coasting and hasattr(self.tracker, "coast"):
                    tracks = self.tracker.coast(
                        frame_number=processed_frame.frame_number,
                        timestamp=processed_frame.timestamp,
                    )
                else:
                    tracks = self.tracker.update(
                        detections=detections,
                        frame_number=processed_frame.frame_number,
                        timestamp=processed_frame.timestamp,
                    )

                # 3. Process Events & Snapshots
                with writer_lock:
                    events = self.event_dispatcher.process_tracks(
                        processed_frame=processed_frame,
                        tracks=tracks,
                        roi_config=self.roi_config,
                        db_conn=db_conn,
                    )

                # Update telemetri gate tripwire crossing & flash trigger
                if events:
                    for ev in events:
                        if ev.event_type == "LINE_CROSSING":
                            tw_id = ev.zone_id or ""
                            tw_suffix = tw_id.split("_")[-1]
                            paired_slot_id = f"zone_{tw_suffix}"
                            direction = "A_TO_B"
                            if ev.notes:
                                notes_up = str(ev.notes).upper()
                                if "B_TO_A" in notes_up or "B TO A" in notes_up or "OUT" in notes_up:
                                    direction = "B_TO_A"
                            self.parking_tracker.apply_tripwire_signal(
                                slot_id=paired_slot_id,
                                direction=direction,
                                track_id=ev.track_id or -1,
                                current_time=processed_frame.timestamp,
                            )
                            self.parking_tracker.record_gate_crossing(
                                ev.notes or "",
                                tripwire_id=tw_id,
                                timestamp=processed_frame.timestamp,
                            )

                # 4. Dispatch Alerts non-blocking
                if self.alert_dispatcher and events:
                    from notification.dispatcher import AlertPacket
                    for ev in events:
                        self.alert_dispatcher.dispatch(
                            AlertPacket(
                                camera_id=ev.camera_id,
                                zone_id=ev.zone_id,
                                event_type=ev.event_type,
                                track_id=ev.track_id,
                                class_label=ev.class_label,
                                message=ev.notes or f"Event {ev.event_type}",
                                timestamp=processed_frame.timestamp,
                                snapshot_path=ev.snapshot_path,
                            )
                        )

                # 5. Off-Lock Live Video Preview Encoding & Buffer Update
                if self.frame_buffer is not None:
                    # Anotasi visual live feed dengan auto-scaling ROI (1080p -> Canvas)
                    live_vis = self.create_annotated_frame(
                        frame=processed_frame.ai_frame,
                        tracks=tracks,
                        timestamp=processed_frame.timestamp,
                        is_warmup=is_warmup,
                        fps=self.fps,
                    )

                    # OFF-LOCK ENCODING: Kompresi JPEG di luar penahanan lock buffer
                    ret, jpeg_bytes = cv2.imencode(
                        ".jpg",
                        live_vis,
                        [cv2.IMWRITE_JPEG_QUALITY, self.live_jpeg_quality],
                    )
                    if ret:
                        self.frame_buffer.set_frame(
                            camera_id=self.camera_config.camera_id,
                            jpeg_bytes=jpeg_bytes.tobytes(),
                            timestamp=processed_frame.timestamp,
                            sequence_id=processed_frame.frame_number,
                        )

        except Exception as e:
            if db_conn:
                try:
                    registry = CameraRegistry(
                        camera_id=self.camera_config.camera_id,
                        display_name=self.camera_config.display_name,
                    )
                    with writer_lock:
                        registry.update_status(db_conn, "error")
                except Exception:
                    pass
            raise e
        finally:
            if db_conn:
                try:
                    registry = CameraRegistry(
                        camera_id=self.camera_config.camera_id,
                        display_name=self.camera_config.display_name,
                    )
                    with writer_lock:
                        registry.update_status(db_conn, "inactive")
                except Exception:
                    pass
