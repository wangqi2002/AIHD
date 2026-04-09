from modelscope import snapshot_download

# 下载 Qwen3-4B 模型，自动续传
model_dir = snapshot_download(
    'Qwen/Qwen3-4B',
    cache_dir='/home/win/.cache/modelscope/hub',  # 可省略，默认就是这个路径
)
print(f"模型已下载至：{model_dir}")