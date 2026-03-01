import torch
from depth_anything_v2.dpt import DepthAnythingV2
import os

# 1. ViT-B 的結構配置 (Encoder: vitb)
model_configs = {
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
}

# 2. 初始化模型
model = DepthAnythingV2(**model_configs['vitb'])

# 3. 載入權重 (Base 版本 Metric 權重)
ckpt_path = r"/home/allen/fly_depth/checkpoints/depth_anything_v2_metric_hypersim_vitb.pth"
if not os.path.exists(ckpt_path):
    print(f"❌ 找不到模型檔案: {ckpt_path}")
    exit()

# 載入權重
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()

# 4. 準備虛擬輸入 
# ANAFI 串流為 720p (1280x720)，導出為 1288x728 以符合 14 的倍數要求
res_w = 1288
res_h = 728
dummy_input = torch.randn(1, 3, res_h, res_w)

# 5. 導出 ONNX
onnx_path = f"depth_anything_v2_vitb_{res_w}x{res_h}.onnx"
print(f"📦 正在導出 ViT-B 到 ONNX (解析度: {res_w}x{res_h})...")

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    opset_version=17, 
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

print(f"✅ ViT-B ONNX 導出成功: {onnx_path}")