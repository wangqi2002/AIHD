import os
import time
import json
from threading import Thread

from funasr import AutoModel

from Scripts.utils_llm import *
from Scripts.utils_ur5e import *

import Scripts.share_vars as share_vars

class audio2action():

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        print("__init__")
        print(f"第一次初始化: {self._initialized}")
        if not self._initialized:
            print("正在加载模型...")
            self.asr_model = AutoModel(
                model="/home/win/.cache/modelscope/hub/FunAudioLLM/Fun-ASR-Nano-2512",
                trust_remote_code=True,
                remote_code="/home/win/Project/P04/AIHD/Fun-ASR//model.py",
                device="cuda:0",
            )   
            self.chat_model = QwenModel() 
            self.ur_robot = UR_Robot()
            self.__class__._initialized = True
            print("模型加载完毕")

    # 为线程定义一个函数
    def robot_execute(self, threadName, content):
        for each in content['function']: # 运行智能体规划编排的每个函数
            print('\n开始执行动作', each)
            eval(each)
        print(f"{threadName}机器人动作执行完毕，共执行了 {len(content['function'])} 个动作")
        
            
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
        content = json.loads(content)
        print(f"智能体规划编排的动作：{content}")
        # share_vars.global_drink_type = content['type']
        # print(f"抓取饮料类型：{share_vars.global_drink_type}")
        
        # 创建并启动线程来执行机器人动作
        robot_thread = Thread(target=self.robot_execute, args=("RobotThread", content))
        robot_thread.start()
        
        return content

if __name__ == "__main__":
    audio_path = "/home/win/Project/P02/Fun-ASR/test.wav"
    a2a = audio2action()
    a2a.action(audio_path)