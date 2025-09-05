import time
from typing import Callable

from .queue import PipeLineQueue, QueueClosedError, StopSig

import numpy as np
class ImageEncoder:
    def __init__(
        self,
        frame_mux: PipeLineQueue,
        output_mux: PipeLineQueue,
        result_mux: PipeLineQueue,
        postprocess_function: Callable,
    ):
        self.frame_mux = frame_mux
        self.output_mux = output_mux
        self.result_mux = result_mux
        self.postprocessor = postprocess_function
        pass

    def run(self):
        FPS = 0.0
        curr_idx = 0
        num_comp = 0
        start_time = time.time()
        while True:
            try:
                frame, context, img_idx = self.frame_mux.get()
                output = self.output_mux.get()
                annotated_img = self.postprocessor(output, context, frame)
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    FPS = ((curr_idx - num_comp)) / elapsed_time
                    start_time = time.time()
                    num_comp = curr_idx

                if not self.result_mux is None:
                    self.result_mux.put((annotated_img, FPS, img_idx))
                    curr_idx += 1
            except QueueClosedError:
                if not self.result_mux is None:
                    self.result_mux.put(StopSig)
                break
            except Exception as e:
                print(f"Error ImageEncoder: {e}")
                break


class PredictionEncoder:
    def __init__(
        self,
        frame_mux: PipeLineQueue,
        output_mux: PipeLineQueue,
        result_mux: PipeLineQueue,
        postprocess_function: Callable,
        timings = None,
    ):
        self.frame_mux = frame_mux
        self.output_mux = output_mux
        self.result_mux = result_mux
        self.postprocessor = postprocess_function
        self.timings = timings

    def run(self):
        while True:
            try:
                frame, context, img_idx = self.frame_mux.get()
                output = self.output_mux.get()

                t4 = time.perf_counter()
                preds = self.postprocessor(output, context, frame.shape[:2])
                t5 = time.perf_counter()
                post_ms = (t5 - t4) * 1000.0

                if self.timings is not None:
                    d = dict(self.timings.get(img_idx, {}))
                    d["post"] = post_ms
                    pre_ms = d.get("pre", 0.0)
                    infer_ms = d.get("infer", 0.0)
                    e2e_active = pre_ms + infer_ms + post_ms
                    d["e2e_active"] = e2e_active

                    # e2e_wall
                    t0 = d.get("t0")
                    if t0 is not None:
                        e2e_wall_val = (time.perf_counter() - t0) * 1000.0
                        d["e2e"] = e2e_wall_val
                    else:
                        e2e_wall_val = None

                    self.timings[img_idx] = d

                    if img_idx < 5:
                        e2e_wall_str = f"{e2e_wall_val:.3f} ms" if e2e_wall_val else "NA"
                        print(f"[Encoder] {img_idx} post={post_ms:.3f} ms "
                            f"e2e_active={e2e_active:.3f} ms "
                            f"e2e_wall={e2e_wall_str}")
                    
                if not self.result_mux is None:
                    self.result_mux.put((preds, 0.0, img_idx))
            except QueueClosedError:
                if not self.result_mux is None:
                    self.result_mux.put(StopSig)
                break
            except Exception as e:
                print(f"Error PredictionEncoder: {e}")
                break
