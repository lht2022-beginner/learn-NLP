import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("当前 GPU 名称:", torch.cuda.get_device_name(0))
    
    # 测试 GPU 操作
    x = torch.rand(5, 5).cuda()
    print("GPU 张量设备:", x.device)
    print("测试成功！CUDA 已正常运行")
else:
    print("错误：CUDA 未启用，请检查以下事项：")
    print("1. 是否安装了 GPU 版本的 PyTorch（如 torch==2.0.1+cu118）")
    print("2. 系统是否安装了兼容的 NVIDIA 驱动（nvidia-smi 可查看驱动版本）")