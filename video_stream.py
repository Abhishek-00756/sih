"""Latest-frame video grabber for live IP cameras.

OpenCV's default capture queue accumulates stale frames when inference
is slower than the camera FPS. This reader always exposes the newest
decoded frame and reconnects dropped RTSP sockets.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

LOGGER = logging.getLogger("perception.stream")

Source = Union[str, int]


def _looks_like_live_source(src: Source) -> bool:
    if isinstance(src, int):
        return True
    text = str(src).lower()
    return text.startswith(("rtsp://", "rtsps://", "http://", "https://", "tcp://"))


class ThreadedCamera:
    """Background capture that drops queued frames so tracking stays real-time."""

    def __init__(
        self,
        src: Source = 0,
        buffer_size: int = 1,
        reconnect_delay: float = 2.0,
        read_sleep: float = 0.0,
        name: str = "",
        freeze_ttl: float = 8.0,
        ffmpeg_tcp: bool = True,
    ) -> None:
        self.src = src
        self.name = name or str(src)
        self.buffer_size = max(1, int(buffer_size))
        self.reconnect_delay = float(reconnect_delay)
        self.read_sleep = float(read_sleep)
        self.freeze_ttl = float(freeze_ttl)
        self.ffmpeg_tcp = bool(ffmpeg_tcp)
        self.live = _looks_like_live_source(src)

        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._need_frame = threading.Event()
        self._need_frame.set()
        self.capture: Optional[cv2.VideoCapture] = None
        self.ret = False
        self.frame: Optional[np.ndarray] = None
        self.frame_id = 0
        self.last_ok = 0.0
        self.opened = False

        self._open()
        self.thread = threading.Thread(target=self.update, name=f"cam-{self.name}", daemon=True)
        self.thread.start()

    def _open(self) -> bool:
        if self.ffmpeg_tcp and isinstance(self.src, str) and str(self.src).lower().startswith("rtsp"):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        cap = cv2.VideoCapture(self.src)
        if self.live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        ok = cap.isOpened()
        if not ok:
            cap.release()
            LOGGER.warning("Failed to open stream %s (%s)", self.name, self.src)
            self.capture = None
            self.opened = False
            return False
        self.capture = cap
        self.opened = True
        LOGGER.info("Opened stream %s <- %s", self.name, self.src)
        return True

    def _drain(self, cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab queued frames; decode only the newest one on live sources."""
        if not self.live:
            return cap.read()
        grabbed = False
        for _ in range(4):
            if not cap.grab():
                if not grabbed:
                    return False, None
                break
            grabbed = True
        if not grabbed:
            return False, None
        return cap.retrieve()

    def update(self) -> None:
        while not self._stopped.is_set():
            if not self.live:
                if not self._need_frame.wait(timeout=0.25):
                    continue
                if self._stopped.is_set():
                    break
            cap = self.capture
            if cap is None or not cap.isOpened():
                if self._stopped.wait(self.reconnect_delay):
                    break
                self._open()
                continue

            ret, frame = self._drain(cap)
            now = time.time()
            if ret and frame is not None:
                with self._lock:
                    self.ret = True
                    self.frame = frame
                    self.frame_id += 1
                    self.last_ok = now
                if not self.live:
                    self._need_frame.clear()
            else:
                stale = (now - self.last_ok) > self.freeze_ttl if self.last_ok else True
                if self.live and stale:
                    LOGGER.warning("Reconnecting stream %s", self.name)
                    cap.release()
                    self.capture = None
                    self.opened = False
                    with self._lock:
                        self.ret = False
                    if self._stopped.wait(self.reconnect_delay):
                        break
                    continue
                if not self.live:
                    with self._lock:
                        self.ret = False
                    self._stopped.set()
                    break

            if self.read_sleep > 0:
                time.sleep(self.read_sleep)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self.ret or self.frame is None:
                return False, None
            frame = self.frame.copy()
            self.ret = False if not self.live else self.ret
        if not self.live:
            self._need_frame.set()
        return True, frame

    def isOpened(self) -> bool:
        return self.opened and not self._stopped.is_set()

    def release(self) -> None:
        self._stopped.set()
        self._need_frame.set()
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.opened = False
