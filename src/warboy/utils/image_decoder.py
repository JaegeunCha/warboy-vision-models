from typing import Callable, List

import cv2

from ..yolo.preprocess import YoloPreProcessor
from .queue import PipeLineQueue, QueueClosedError, StopSig

import time

class ImageListDecoder:
    def __init__(
        self,
        image_list: List,
        stream_mux: PipeLineQueue,
        frame_mux: PipeLineQueue,
        preprocess_function: Callable = YoloPreProcessor(),
        timings = None,
    ):
        self.image_paths = PipeLineQueue()
        for image in image_list:
            self.image_paths.put(image.image_info)
        self.image_paths.put(StopSig)

        self.preprocessor = preprocess_function
        self.stream_mux = stream_mux
        self.frame_mux = frame_mux
        self.timings = timings

    def run(self):
        img_idx = 0
        while True:
            try:
                image_path = self.image_paths.get()
                img = cv2.imread(image_path)
                
                t0 = time.perf_counter()
                input_, context = self.preprocessor(img)
                t1 = time.perf_counter()
                pre_ms = (t1 - t0) * 1000.0

                if self.timings is not None:
                    d = dict(self.timings.get(img_idx, {}))  # 기존 값 복사
                    d["pre"] = pre_ms
                    d["t0"] = t0
                    self.timings[img_idx] = d               # 새 dict로 교체

                    if img_idx < 5:
                        print(f"[Decoder] {img_idx} pre={pre_ms:.3f} ms")
                    
                self.stream_mux.put((input_, img_idx))
                self.frame_mux.put((img, context, img_idx))
                img_idx += 1

            except QueueClosedError:
                break

        self.stream_mux.put(StopSig)
        self.frame_mux.put(StopSig)
        return
