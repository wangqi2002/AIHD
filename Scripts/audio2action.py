import os
import time
import json

from funasr import AutoModel

from Scripts.utils_llm import *
from Scripts.utils_ur5e import *

import Scripts.share_vars 

class audio2action():
    def __init__(self):

        self.asr_model = AutoModel(
        model="/home/win/.cache/modelscope/hub/FunAudioLLM/Fun-ASR-Nano-2512",
        trust_remote_code=True,
        remote_code="/home/win/Project/P03/Fun-ASR//model.py",
        device="cuda:0",
    )   
        self.chat_model = QwenModel() 

        print("模型加载完毕")

    def audio_to_text(self, audio_path):
        start_time = time.time()
        res = self.asr_model.generate(
        input=[audio_path],
        cache={},
        batch_size=1,
        hotwords=["拿一杯", "可乐", "芬达", "雪碧",],
        language="中文",
        itn=True, # or False
    )
        text = res[0]["text"]
        end_time = time.time()
        print(f"语音识别用时: {end_time - start_time} 秒")
        return text
    def action(self, text):
        start_time = time.time()
        print(text)
        thinking_content, content, inference_time = self.chat_model.generate_response(prompt = text)
        end_time = time.time()
        print(f"模型识别用时: {end_time - start_time} 秒")
        # ur_robot = UR_Robot()
        content = json.loads(content)
        # share_vars.global_drink_type = content['type']

        # for each in content['function']: # 运行智能体规划编排的每个函数
        #     print('\n开始执行动作', each)
        #     eval(each)
        return content

if __name__ == "__main__":
    audio_path = "/home/win/Project/P02/Fun-ASR/test.wav"
    a2a = audio2action()
    a2a.action(audio_path)