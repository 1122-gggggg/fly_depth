import olympe
import os, time, numpy as np, cv2, tensorrt as trt
import cupy as cp
import threading

# --- [1] TensorRT 穩定版類別 ---
class DepthInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # 預先配置輸出記憶體於 GPU 上
        output_shape = self.engine.get_tensor_shape(self.engine.get_tensor_name(1))
        self.d_output = cp.empty(output_shape, dtype=cp.float32)
        
        print(f"✅ RTX 5090 深度引擎已就緒")

    def infer_gpu(self, d_input):
        # 輸入與輸出都直接使用 GPU 指標
        self.context.execute_v2(bindings=[d_input.data.ptr, self.d_output.data.ptr])
        # 此處範例傳回 numpy array 以供 OpenCV 顯示
        return self.d_output.get()

# --- [2] RTSP 讀取加速線程 (防止畫面凍結) ---
class VideoCaptureThread:
    def __init__(self, url):
        # 優化 FFMPEG 參數以降低 ANAFI 的 280ms 原生延遲
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 最小緩衝區
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame # 永遠覆蓋為最新影格
            else:
                time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# --- [3] 主程序 ---
DRONE_IP = "192.168.42.1"
RTSP_URL = f"rtsp://{DRONE_IP}/live"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.path.join(CURRENT_DIR, "depth_728p_fp16.engine")

def main():
    print(f"嘗試連線無人機: {DRONE_IP}...")
    drone = olympe.Drone(DRONE_IP)
    if not drone.connect():
        print("❌ 無人機連線失敗")
        return
    print("✅ 無人機連線成功")

    print(f"載入 TensorRT 模型: {ENGINE_PATH}...")
    model = DepthInference(ENGINE_PATH)
    
    print(f"啟動 RTSP 接收線程: {RTSP_URL}...")
    vcap = VideoCaptureThread(RTSP_URL)

    print("等待第一張 RTSP 影像...")
    wait_time = 0
    while vcap.read() is None:
        time.sleep(0.1)
        wait_time += 0.1
        if wait_time > 5.0:  
            print("❌ 無法取得 RTSP 影像串流")
            vcap.stop()
            drone.disconnect()
            return
            
    print("✅ 取得首張影像，開始即時推論")
    cv2.namedWindow("ANAFI Real-time AI Monitor", cv2.WINDOW_NORMAL)
    frame_idx = 0

    try:
        fps_start_time = time.perf_counter()
        fps_frame_count = 0
        current_fps = 0.0

        while True:
            frame = vcap.read()
            if frame is None:
                continue
            
            # --- 核心運算開始時間 ---
            t_start = time.perf_counter()

            # 1. 前處理 (CUDA HtoD)
            d_frame = cp.asarray(frame) 
            # Padding (1280x720 -> 1288x728)
            d_padded = cp.pad(d_frame, ((0, 8), (0, 8), (0, 0)), mode='constant')
            # BGR 轉 RGB, HWC 轉 CHW, 正規化
            d_input_data = (d_padded[:, :, ::-1].transpose(2, 0, 1).astype(cp.float32) / 255.0)
            
            # 擴充 batch 維度 (CHW -> 1CHW) - TensorRT 通常需要 Batch 維度
            d_input_data = cp.expand_dims(d_input_data, axis=0)
            
            # 確保記憶體連續
            d_input_data = cp.ascontiguousarray(d_input_data)
            
            # 2. 推論 (全程 GPU 記憶體指針交換，並轉回 numpy)
            depth_out = model.infer_gpu(d_input_data)
            
            # --- 核心運算結束時間 ---
            t_infer_end = time.perf_counter()
            infer_ms = (t_infer_end - t_start) * 1000
            
            frame_idx += 1
            fps_frame_count += 1
            
            if time.perf_counter() - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (time.perf_counter() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.perf_counter()
                print(f"純推論 FPS: {current_fps:.1f} | 單張延遲: {infer_ms:.1f} ms | Frame: {frame_idx}")

            # 3. 後處理 (裁切回 720p 並正規化)
            # 假設輸出的 shape 為 (1, 728, 1288) 或類似結構，這裡針對前兩個維度處理
            # 如果你的模型輸出帶有 channel 維度 (例如 1, 1, 728, 1288)，請調整 index，例如 depth_out[0, 0, 0:720, 0:1280]
            if depth_out.ndim == 3:
                depth_crop = depth_out[0, 0:720, 0:1280]
            elif depth_out.ndim == 4:
                depth_crop = depth_out[0, 0, 0:720, 0:1280]
            else:
                depth_crop = depth_out[0:720, 0:1280]

            d_norm = ((depth_crop - depth_crop.min()) / (depth_crop.max() - depth_crop.min() + 1e-6) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
            
            # 4. 並列拼接 (左: 原始畫面 | 右: 深度圖)
            combined = np.hstack((frame, depth_color))
            
            cv2.putText(combined, f"Core FPS: {current_fps:.1f} | Infer Latency: {infer_ms:.1f}ms", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("ANAFI Real-time AI Monitor", combined)
            
            key = cv2.waitKey(1)
            if key == 27 or key == 113:
                print("收到結束指令，準備關閉...")
                break
            
            if cv2.getWindowProperty("ANAFI Real-time AI Monitor", cv2.WND_PROP_VISIBLE) < 1:
                print("視窗被關閉，準備結束...")
                break
    finally:
        vcap.stop()
        time.sleep(1)
        drone.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()