import asyncio
import time
from collections import defaultdict
from typing import List

from furiosa.runtime import create_queue
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig

from ..utils.queue import PipeLineQueue, QueueClosedError, StopSig

import numpy as np

class WarboyApplication:
    """
    An inference engine using FuriosaAI Runtime, based on Queue

    Args:
        model(str): a path to quantized onnx file
        worker_num(int): the number of npu workers
        device(str): a set of NPU devices by a textual string (e.g. "warboy(2)*1", "npu0pe0", etc)
        stream_mux_list(list):
        output_mux_list(list):
    """

    def __init__(
        self,
        model: str,
        worker_num: str,
        device: str,
        stream_mux_list: List[PipeLineQueue],
        output_mux_list: List[PipeLineQueue],
        batch_size: int = 1,
        timings = None,
    ):
        #self.config = {"model": model, "worker_num": worker_num, "npu_device": device}
        #self.model = FuriosaRTModel(
        #    FuriosaRTModelConfig(name="YOLO", batch_size=1, **self.config)
        #)
        
        if model.endswith(".enf"):
            print(f"[WarboyApplication] Loading precompiled ENF: {model}")
            self.config = {
                "model": model,
                "worker_num": worker_num,
                "npu_device": device,
            }
            self.model = FuriosaRTModel(
                FuriosaRTModelConfig(
                    name="YOLO",                  
                    **self.config
                )
            )
        else:
            print(f"[WarboyApplication] Loading ONNX (will compile): {model}")
            self.config = {
                "model": model,
                "worker_num": worker_num,
                "npu_device": device,
            }
            self.model = FuriosaRTModel(
                FuriosaRTModelConfig(
                    name="YOLO",
                    batch_size=batch_size,
                    **self.config
                )
            )        
        
        self.batch_size = batch_size
        self.stream_mux_list = stream_mux_list
        self.output_mux_list = output_mux_list

        self.timings = timings
        print("WarboyApplication - init")

    def run(self):
        asyncio.run(self.task())

    async def task(self):
        await self.load()
        await asyncio.gather(
            *(
                self.inference(video_channel, stream_mux, output_mux)
                for video_channel, (stream_mux, output_mux) in enumerate(
                    zip(self.stream_mux_list, self.output_mux_list)
                )
            )
        )
        return

    async def inference(
        self, video_channel: int, stream_mux: PipeLineQueue, output_mux: PipeLineQueue
    ):
        # ✅ batch_size == 1 → 절대 변경하지 않음
        if self.batch_size == 1:
            while True:
                try:
                    input_, img_idx = stream_mux.get()
                except QueueClosedError:
                    break

                t2 = time.perf_counter()
                output = await self.model.predict(input_)  # 원래대로

                if img_idx < 2:  # 처음 몇 장만
                    print("[DEBUG bs=1]", type(output), getattr(output, "shape", None))
                    if isinstance(output, (list, tuple)):
                        for j, head in enumerate(output):
                            print(f"  head{j}:", type(head), getattr(head, "shape", None))

                t3 = time.perf_counter()
                infer_ms = (t3 - t2) * 1000.0

                if self.timings is not None:
                    d = dict(self.timings.get(img_idx, {}))
                    d["infer"] = infer_ms
                    d["t2"] = t2
                    self.timings[img_idx] = d
                    if img_idx < 5:
                        print(f"[Runtime] {img_idx} infer={infer_ms:.3f} ms")

                output_mux.put(output)  # 원래대로

            output_mux.put(StopSig)
            return

        # ✅ batch_size > 1 → 배치 모드
        batch_inputs, batch_indices = [], []

        def _norm_head_per_image(head: np.ndarray, i: int):
            """
            배치 ENF가 평탄화해서 내보내는 head를
            postprocessor가 기대하는 (1, C, H, W)로 되돌린다.
            """
            if not isinstance(head, np.ndarray):
                return head

            arr = head

            # 1) 배치축이 있는 경우 per-image로 슬라이스
            if arr.ndim == 4 and arr.shape[0] > i:
                # (B, C, H, W) → (1, C, H, W)
                return arr[i:i+1]
            if arr.ndim == 3 and arr.shape[0] == self.batch_size and arr.shape[0] > i:
                # (B, C*H, W) 같은 형태 → i 슬라이스 후 아래에서 복원
                arr = arr[i]

            # 2) per-image 상태에서 모양 복원
            if arr.ndim == 3:
                # (C, H, W) → (1, C, H, W)
                return arr[None, ...]
            if arr.ndim == 2:
                # (C*H, W) → (1, C, H, W), 여기서 H=W 가정
                W = arr.shape[1]
                H = W
                C = arr.shape[0] // H if H > 0 else 0
                if C * H == arr.shape[0] and C > 0:
                    return arr.reshape(1, C, H, W)
                return arr

            return arr

        def _per_image(outputs, i):
            if isinstance(outputs, (list, tuple)):
                fixed = [ _norm_head_per_image(h, i) for h in outputs ]
                return type(outputs)(fixed)
            if isinstance(outputs, np.ndarray):
                return _norm_head_per_image(outputs, i)
            return outputs

        def _emit(outputs, infer_ms, t2, B_effective=None):
            """
            B_effective: 추론 시간 분배에 사용할 배치 크기
            - 정규 배치: len(batch_indices)
            - 잔여 패딩 배치: self.batch_size (HW 관점 분배)
            """
            idxs = list(batch_indices)
            if not idxs:
                return

            B = B_effective if B_effective is not None else len(idxs)

            for i, idx in enumerate(idxs):
                out_i = _per_image(outputs, i)  # ← 먼저 생성

                # 🔍 필요하면 구조만 확인
                if idx < 2:
                    def _peek(x):
                        if isinstance(x, (list, tuple)):
                            return [getattr(a, 'shape', None) for a in x]
                        return getattr(x, 'shape', None)
                    print(f"[DEBUG bs>1] out_i structure: {_peek(out_i)}")

                if self.timings is not None:
                    d = dict(self.timings.get(idx, {}))
                    d["infer"] = infer_ms / B
                    d["t2"] = t2
                    d["batch_size"] = B
                    self.timings[idx] = d

                output_mux.put(out_i)

        while True:
            try:
                input_, img_idx = stream_mux.get()
                batch_inputs.append(input_)
                batch_indices.append(img_idx)

                if len(batch_inputs) < self.batch_size:
                    continue

            except QueueClosedError:
                if batch_inputs:
                    actual = len(batch_inputs)
                    if actual < self.batch_size:
                        pad_needed = self.batch_size - actual
                        batch_inputs_padded = batch_inputs + [batch_inputs[-1]] * pad_needed
                    else:
                        batch_inputs_padded = batch_inputs

                    batched_input = np.concatenate(batch_inputs_padded, axis=0)  # (B,C,H,W)
                    t2 = time.perf_counter()
                    outputs = await self.model.predict(batched_input)
                    t3 = time.perf_counter()
                    infer_ms = (t3 - t2) * 1000.0

                    # 패딩 실행 → 분배는 B=self.batch_size 기준
                    _emit(outputs, infer_ms, t2, B_effective=self.batch_size)
                break

            # 정규 배치 실행
            batch_input = np.concatenate(batch_inputs, axis=0)  # (B,C,H,W)
            t2 = time.perf_counter()
            outputs = await self.model.predict(batch_input)
            t3 = time.perf_counter()
            infer_ms = (t3 - t2) * 1000.0
            _emit(outputs, infer_ms, t2)

            batch_inputs, batch_indices = [], []

        output_mux.put(StopSig)
        return
    
    async def load(self):
        await self.model.load()


class WarboyQueueRuntime:
    def __init__(
        self,
        model: str,
        worker_num: int,
        device: str,
        stream_mux_list: List[PipeLineQueue],
        output_mux_list: List[PipeLineQueue],
    ):
        self.config = {"model": model, "worker_num": worker_num, "device": device}
        self.submitter = None
        self.receiver = None
        self.stream_mux_list = stream_mux_list
        self.output_mux_list = output_mux_list
        self.pending_tasks = 0
        self.stop_count = 0
        self.total_submitters = len(self.stream_mux_list)

        self.pending_lock = None
        self.done_event = None

        print("WarboyQueueRuntime - init")

    def run(self):
        asyncio.run(self.run_())

    async def run_(self):
        self.submitter, self.receiver = await create_queue(**self.config)
        self.pending_lock = asyncio.Lock()
        self.done_event = asyncio.Event()

        task = [self.recv_with()] + [
            self.submit_with(video_channel)
            for video_channel in range(len(self.stream_mux_list))
        ]
        await asyncio.gather(*task)

    async def submit_with(self, video_channel: int):

        while True:
            try:
                input_, img_idx = self.stream_mux_list[video_channel].get()

                async with self.pending_lock:
                    self.pending_tasks += 1

                await self.submitter.submit(input_, context=(video_channel, img_idx))
            except QueueClosedError:
                async with self.pending_lock:
                    self.stop_count += 1
                    if (
                        self.stop_count == self.total_submitters
                        and self.pending_tasks == 0
                    ):
                        self.done_event.set()
                        break
                print(f"Channel - {video_channel} End!")
                break
            except Exception as e:
                print("[WarboyQueueRuntime] submit_with:", e)
                break
        return

    async def recv_with(self):

        # buffer: channel → {img_idx → output}
        buffer = defaultdict(dict)
        # expected next index: channel → int
        expected_idx = defaultdict(lambda: 0)

        while True:
            try:
                t1 = time.time()
                (video_channel, img_idx), output = await self.receiver.recv()

                buffer[video_channel][img_idx] = output
                while expected_idx[video_channel] in buffer[video_channel]:
                    output = buffer[video_channel].pop(expected_idx[video_channel])
                    self.output_mux_list[video_channel].put(output)
                    expected_idx[video_channel] += 1

                async with self.pending_lock:
                    self.pending_tasks -= 1
                    if (
                        self.stop_count == self.total_submitters
                        and self.pending_tasks == 0
                    ):
                        self.done_event.set()
                        break
            except asyncio.TimeoutError:
                print("TimeOut Receiver")
                break
            except Exception as e:
                print("[WarboyQueueRuntime] recv_with:", e)
                break

        for output_mux in self.output_mux_list:
            output_mux.put(StopSig)
        return
