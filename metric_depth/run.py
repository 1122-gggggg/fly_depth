import tensorrt as trt
import numpy as np
import cv2
import time
import argparse
import os
import ctypes
import cupy as cp
from cupyx.scipy.ndimage import zoom  # 用於 GPU 端高效縮放
from pathlib import Path

# --- [1] CUDA 底層驅動與 DLL 管理 ---
cuda_bin_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
cuda_dll_name = "cudart64_13.dll"
cuda_dll_path = os.path.join(cuda_bin_dir, cuda_dll_name)

if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(cuda_bin_dir)

libcudart = ctypes.WinDLL(cuda_dll_path)
libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

class DepthTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.bindings, self.tensors = [], []
        
        print(f"🔍 偵測 Engine 配置 (Batch Size: 1)...")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            if shape[0] == -1: # 動態尺寸處理
                new_shape = list(shape)
                new_shape[0] = 1
                self.context.set_input_shape(name, tuple(new_shape))
                shape = new_shape

            size = int(np.prod(shape) * np.dtype(dtype).itemsize)
            device_mem = ctypes.c_void_p()
            libcudart.cudaMalloc(ctypes.byref(device_mem), size)
            
            self.bindings.append(device_mem.value)
            self.tensors.append({
                'name': name, 'mem': device_mem, 'size': size, 
                'shape': shape, 'dtype': dtype, 'idx': i
            })
            print(f"   [{'Input' if i==0 else 'Output'}] {name}: {shape}")

        self.input_h, self.input_w = self.tensors[0]['shape'][2], self.tensors[0]['shape'][3]

        # 預先分配輸出端的 CuPy 陣列
        res_shape = (self.tensors[1]['shape'][-2], self.tensors[1]['shape'][-1])
        self.output_gpu = cp.empty(res_shape, dtype=cp.float32) 
        self.bindings[1] = self.output_gpu.data.ptr

    def infer_gpu(self, gpu_input_ptr):
        """專為 Batch Size 1 優化的純 Device-to-Device 推論"""
        libcudart.cudaMemcpy(self.tensors[0]['mem'], gpu_input_ptr, self.tensors[0]['size'], 3)
        self.context.execute_v2(self.bindings)
        return self.output_gpu


def phase1_inference(img_path, engine_path, output_dir):
    """階段 1: 讀取影片 -> 推論 -> 儲存深度數據 (.npy)"""
    print("\n--- [Phase 1: 推論與數據儲存] ---")
    model = DepthTRT(engine_path)
    cap = cv2.VideoCapture(img_path)
    
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_factors = (model.input_h / h, model.input_w / w, 1)

    os.makedirs(output_dir, exist_ok=True)
    depth_data_list = []
    
    frame_count = 0
    total_infer_time = 0.0

    print(f"🚀 開始純推論測試...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        t_start = time.perf_counter()
        
        gpu_frame = cp.asarray(frame)
        gpu_rgb = gpu_frame[:, :, ::-1]
        gpu_resized = zoom(gpu_rgb, scale_factors, order=1)
        gpu_input = gpu_resized.transpose(2, 0, 1).astype(cp.float32) / 255.0
        
        depth_gpu = model.infer_gpu(gpu_input.data.ptr)
        depth = cp.asnumpy(depth_gpu) # 拉回 CPU
        
        t_end = time.perf_counter()
        
        infer_time = t_end - t_start
        total_infer_time += infer_time
        fps = 1.0 / infer_time if infer_time > 0 else 0
        
        depth_data_list.append(depth)
        frame_count += 1
        
        print(f"\r推論進度: 幀 {frame_count} | 當前 FPS: {fps:6.1f}", end="")

    cap.release()
    
    avg_fps = frame_count / total_infer_time if total_infer_time > 0 else 0
    print(f"\n✅ Phase 1 完成 | 總幀數: {frame_count} | 平均推論 FPS: {avg_fps:.1f}")
    
    # 儲存深度資料供 Phase 2 使用
    np.save(os.path.join(output_dir, "depth_data.npy"), np.array(depth_data_list))
    return frame_count, w, h


def phase2_visualization(img_path, output_dir, w, h):
    """階段 2: 讀取深度數據 -> 合併原圖 -> 繪製避障邏輯 -> 輸出影片"""
    print("\n--- [Phase 2: 深度視覺化與影片生成] ---")
    cap = cv2.VideoCapture(img_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    out_video = cv2.VideoWriter(os.path.join(output_dir, 'RTX5090_ANAFI_Avoidance.mp4'), 
                               cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (w * 2, h))

    # 讀取 Phase 1 儲存的數據
    depth_data_array = np.load(os.path.join(output_dir, "depth_data.npy"))
    
    LATENCY_SEC = 0.28  # ANAFI 延遲
    CURRENT_SPEED = 10.0 
    LATENCY_OFFSET = CURRENT_SPEED * LATENCY_SEC 

    frame_idx = 0
    total_viz_time = 0.0

    print(f"🚀 開始視覺化處理...")
    while cap.isOpened() and frame_idx < len(depth_data_array):
        ret, frame = cap.read()
        if not ret: break
        
        t_start = time.perf_counter()
        
        depth = depth_data_array[frame_idx]
        
        # 避障邏輯視覺化
        h_unit = depth.shape[0] // 3
        zones = [("TOP", depth[0:h_unit, :]), ("MID", depth[h_unit:2*h_unit, :]), ("BOT", depth[2*h_unit:, :])]
        
        for idx, (name, data) in enumerate(zones):
            raw_min_d = data.min()
            real_min_d = raw_min_d - LATENCY_OFFSET
            
            is_danger = real_min_d < 3.0 
            color = (0, 0, 255) if is_danger else (0, 255, 0)
            y_s = idx * (h // 3)
            
            cv2.rectangle(frame, (20, y_s + 20), (w - 20, y_s + (h//3) - 20), color, 8 if is_danger else 2)
            label = f"{name}: {real_min_d:.1f}m (Compensated)"
            cv2.putText(frame, label, (50, y_s + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

        # 深度圖上色與大小調整
        d_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        d_side = cv2.resize(d_color, (w, h))

        # 合併影像
        sbs_frame = np.hstack((frame, d_side))
        
        t_end = time.perf_counter()
        
        viz_time = t_end - t_start
        total_viz_time += viz_time
        fps = 1.0 / viz_time if viz_time > 0 else 0
        
        cv2.putText(sbs_frame, f"VIZ FPS: {fps:.1f}", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        out_video.write(sbs_frame)
        
        frame_idx += 1
        print(f"\r視覺化進度: 幀 {frame_idx} | 當前 FPS: {fps:6.1f}", end="")

    cap.release()
    out_video.release()
    
    avg_fps = frame_idx / total_viz_time if total_viz_time > 0 else 0
    print(f"\n✅ Phase 2 完成 | 平均視覺化 FPS: {avg_fps:.1f}")
    print(f"🎉 影片已儲存於 {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--engine-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output_final')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 執行階段 1
    total_frames, w, h = phase1_inference(args.img_path, args.engine_path, output_dir)
    
    # 執行階段 2
    if total_frames > 0:
        phase2_visualization(args.img_path, output_dir, w, h)