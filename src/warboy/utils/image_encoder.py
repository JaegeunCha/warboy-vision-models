import time
from typing import Callable, List, Optional

from .queue import PipeLineQueue, QueueClosedError, StopSig

from pathlib import Path
import cv2, numpy as np

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
        class_names: Optional[List[str]] = None,
        save_samples: int = 0,
        sample_start: int = 1001,   # COCO stem ID 시작 기본값(요청대로 1001)
        save_dir: str = "outputs",
        sample_by: str = "stem",    # "stem" | "index" (기본 stem)       
        batch_size: int = 1, 
    ):
        self.frame_mux = frame_mux
        self.output_mux = output_mux
        self.result_mux = result_mux
        self.postprocessor = postprocess_function
        self.timings = timings
        self.class_names = class_names or []
        self.save_samples = int(save_samples or 0)
        self.sample_start = int(sample_start or 0)
        self.save_dir = Path(save_dir)
        self.sample_by = sample_by
        self.save_dir_root = Path(save_dir)
        self.batch_size = batch_size

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

                    ## e2e_wall
                    #t0 = d.get("t0")
                    #if t0 is not None:
                    #    wall_ms = (time.perf_counter() - t0) * 1000.0
                    #    B = d.get("batch_size", 1)                        
                    #    d["e2e"] = wall_ms / max(B, 1)   
                    #else:
                    #    d["e2e"] = None

                    self.timings[img_idx] = d

                    if img_idx < 5:
                        #e2e_wall_str = f"{e2e_wall_val:.3f} ms" if e2e_wall_val else "NA"
                        #print(f"[Encoder] {img_idx} post={post_ms:.3f} ms "
                        #    f"e2e_active={e2e_active:.3f} ms "
                        #    f"e2e_wall={e2e_wall_str}")
                        e2e_val = d.get("e2e")
                        e2e_wall_str = f"{e2e_val:.3f} ms" if e2e_val is not None else "NA"
                        print(f"[Encoder] {img_idx} post={post_ms:.3f} ms "
                              f"e2e_active={e2e_active:.3f} ms "
                              f"e2e_wall={e2e_wall_str}")     
                        
                ## --- 저장: sample_start ≤ idx < sample_start + save_samples ---
                #if self.save_samples and self.sample_start <= img_idx < self.sample_start + self.save_samples:
                #    import os, cv2, numpy as np
                #    os.makedirs(self.save_dir, exist_ok=True)
                #    draw = frame.copy()

                #    # preds: list/tuple → 첫 요소 (배치=1)
                #    det = preds[0] if isinstance(preds, (list, tuple)) else preds
                #    if det is not None and len(det):
                #        for xyxyc in det:
                #            x1, y1, x2, y2, conf, cls = xyxyc[:6]
                #            p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
                #            cv2.rectangle(draw, p1, p2, (0,255,0), 2)
                #            cv2.putText(draw, f"{int(cls)} {float(conf):.2f}",
                #                        (p1[0], max(p1[1]-4, 0)),
                #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                #    out_path = os.path.join(self.save_dir, f"{img_idx:06d}_pred.jpg")
                #    try:
                #        cv2.imwrite(out_path, draw)
                #        if img_idx < self.sample_start + 3:
                #            print(f"[PredictionEncoder] saved: {out_path}")
                #    except Exception as e:
                #        print(f"[PredictionEncoder] save error: {e}")      

                # -------------------------------
                # 샘플 저장 (bbox + label)
                # -------------------------------
                if self.save_samples > 0:
                    # 저장 대상인지 판정
                    should_save = False

                    # 컨텍스트/인덱스 확보
                    img_path = None
                    if isinstance(context, dict):
                        img_path = context.get("image_path")

                    if self.sample_by == "stem" and img_path:
                        # COCO 파일명 → 정수 stem ID
                        try:
                            stem_id = int(Path(img_path).stem)
                        except Exception:
                            stem_id = None
                        if stem_id is not None:
                            if self.sample_start <= stem_id < self.sample_start + self.save_samples:
                                should_save = True
                    else:
                        # index 방식 (전/후방 호환)
                        if self.sample_start <= img_idx + 1 < self.sample_start + self.save_samples:
                            should_save = True

                    if should_save:
                        draw = frame.copy()
                        # preds 형상 표준화: (N, >=6) [x1,y1,x2,y2,conf,cls]
                        dets = preds[0] if isinstance(preds, (list, tuple)) else preds
                        if dets is not None:
                            nd = np.asarray(dets)
                            if nd.ndim == 2 and nd.shape[1] >= 6 and nd.shape[0] > 0:
                                for row in nd:
                                    x1, y1, x2, y2, conf, cls = row[:6]
                                    p1 = (int(x1), int(y1))
                                    p2 = (int(x2), int(y2))
                                    cv2.rectangle(draw, p1, p2, (0, 255, 0), 2)
                                    
                                    # 클래스 이름 매핑
                                    name = (
                                        self.class_names[int(cls)]
                                        if self.class_names and 0 <= int(cls) < len(self.class_names)
                                        else str(int(cls))
                                    )
                                    label = f"{name} {float(conf):.2f}"
                                    cv2.putText(draw, label, (p1[0], max(p1[1]-4, 0)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                        # 저장 경로: outputs/{model명}_{batch_size}/
                        stem = Path(img_path).stem if img_path else f"{img_idx:06d}"
                        model_bs_dir = self.save_dir_root.parent / f"{self.save_dir_root.name}_{self.batch_size}"
                        model_bs_dir.mkdir(parents=True, exist_ok=True)
                        out_path = model_bs_dir / f"{stem}_pred.jpg"
                        cv2.imwrite(str(out_path), draw)
                        #print(f"[PredictionEncoder] saved: {out_path}")

                if not self.result_mux is None:
                    self.result_mux.put((preds, 0.0, img_idx))
            except QueueClosedError:
                if not self.result_mux is None:
                    self.result_mux.put(StopSig)
                break
            except Exception as e:
                print(f"Error PredictionEncoder: {e}")
                break
