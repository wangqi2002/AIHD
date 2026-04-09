import time
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# python_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(python_dir)

class QwenModel:
    def __init__(self, model_name="/home/win/.cache/modelscope/hub/Qwen/Qwen3-4B"):
        """
        初始化Qwen模型
        """
        self.model_name = model_name
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        # 初始化聊天历史
        self.chat_history = []

        # # 加载总提示词
        # dir_path = os.path.dirname(os.path.dirname(__file__)) 
        # dir_path = os.path.join(dir_path,"Prompts")

        # f1 = open(os.path.join(dir_path, "task.txt"))
        # system_prompt = f1.read()
        # self.system_prompt = system_prompt 
        # # self.system_prompt = system_prompt + '\n' + api_prompt
        # print(self.system_prompt)

        # 加载总提示词
        dir_path = os.path.dirname(os.path.dirname(__file__)) 
        dir_path = os.path.join(dir_path,"Prompts")

        f1 = open(os.path.join(dir_path, "system.txt"))
        system_prompt = f1.read()
        f3 = open(os.path.join(dir_path, "task.txt"))
        task_prompt = f3.read()
        self.system_prompt = system_prompt + '\n' + task_prompt
        # self.system_prompt = system_prompt + '\n' + api_prompt
        # print(self.system_prompt)

        # instruction = """
        # 你是一个工业机器人智能助手，工业机器人内置了一些函数。
        # 你的任务是根据我的指令，以json形式输出要运行的对应函数。
        # 【以下是所有内置函数的介绍】
        # 1. 快速回到原点：ur_robot.go_home()
        # 2. 笛卡尔位置运动：ur_robot.move_direction(x, y, z),其中x,y,z分别表示沿x轴,y轴,z轴移动的距离,单位是毫米,可取正负.向前为正，向后为负，向左为正，向右为负，向上为正，向下为负.
        # 3. 关节位置运动：ur_robot.move_j(a, b, c, d, e, f),其中a,b,c,d,e,f代表第一关节、第二关节、第三关节、第四关节、第五关节、第六关节的旋转角度,单位是度数,取值范围是-360度到360度.
        # 4. 爪手控制：ur_robot.gripper(b),'b' 代表爪手的开合程度,其取值为True表示完全关闭,False表示完全打开.

        # 如果我输入的指令不完整，请提示让我补充完整。
        # """

        # # 输出格式
        # output_format = """
        # 以 JSON 格式输出
        # 你直接输出json即可，从{开始，不要输出包含```json的开头或结尾
        # 输出json中的全角字符被正确转换为半角字符，特别是冒号
        # 在"function"键中，输出函数名列表，列表中每个元素都是字符串，代表要运行的函数名称和参数。每个函数既可以单独运行，也可以和其他函数先后运行。列表元素的先后顺序，表示执行函数的先后顺序
        # 在"response"键中，根据我的指令和你编排的动作，以第一人称输出你回复我的话，不要超过20个字，用中文回复。
        # """

        # # 加入例子
        # examples = """
        # 我的指令是：帮我把机器人移回原点。你只需要输出：{"function":["ur_robot.go_home()"], "response":"好的，我正在返回原点"}
        # 我的指令是：机器人向前移动120毫米。你只需要输出：{"function":["ur_robot.move_direction(0,120,0)"], "response":"好的，我朝着前方移动120mm"}
        # 我的指令是：机器人向后移动200毫米。你只需要输出：{"function":["ur_robot.move_direction(0,-200,0)"], "response":"好的，我朝着后方移动120mm"}
        # 我的指令是：向左移动100毫米。你只需要输出：{"function":["ur_robot.move_direction(100,0,0)"], "response":"好的，我正在向着左边移动100mm"}
        # 我的指令是：向右移动80毫米。你只需要输出：{"function":["ur_robot.move_direction(-80,0,0)"], "response":"好的，我正在向着右边移动80mm"}
        # 我的指令是：向上移动55毫米。你只需要输出：{"function":["ur_robot.move_direction(0,0,55)"], "response":"好的，我正在向着上方移动55mm"}
        # 我的指令是：向下移动70毫米。你只需要输出：{"function":["ur_robot.move_direction(0,0,-80)"], "response":"好的，我正在向着下方移动70mm"}
        # 我的指令是：第一关节反向旋转45度。你只需要输出：{"function":["ur_robot.move_j(-45,0,0,0,0,0)"], "response":"好的，关节1反向旋转45度"}
        # 我的指令是：第二关节正向旋转35度。你只需要输出：{"function":["ur_robot.move_j(0,35,0,0,0,0)"], "response":"好的，关节2正向旋转35度"}
        # 我的指令是：第三关节正向旋转15度。你只需要输出：{"function":["ur_robot.move_j(0,0,15,0,0,0)"], "response":"好的，关节3正向旋转15度"}
        # 我的指令是：第四关节反向旋转45度。你只需要输出：{"function":["ur_robot.move_j(0,0,0,-45,0,0)"], "response":"好的，关节3反向旋转45度"}
        # 我的指令是：第五关节反向旋转20度。你只需要输出：{"function":["ur_robot.move_j(0,0,0,0,-20,0)"], "response":"好的，关节5反向旋转20度"}
        # 我的指令是：第六关节正向旋转60度。你只需要输出：{"function":["ur_robot.move_j(0,0,0,0,0,60)"], "response":"好的，关节6正向旋转60度"}
        # 我的指令是：抓取物体。你只需要输出：{"function":["ur_robot.gripper(True)"], "response":"正在抓取物体"}
        # 我的指令是：手腕旋转。你只需要输出：{"function":[""], "response":"旋转角度数值缺失，请补充旋转的角度值"}
        # 任务一：饮品需求场景

        # 我的指令是：帮我把绿色方块放在红色方块上面。你输出：{"function":[vlm_move("帮我把绿色方块放在红色方块上面")], "response":"好的，正在执行"}
        # 我的指令是：帮我把红色方块放在绿色方块上面。你输出：{"function":[vlm_move("帮我把红色方块放在绿色方块上面")], "response":"收到，正在执行命令"}
        # """

        # # prompt 模版
        # self.system_prompt = f"""
        # {instruction}

        # {output_format}

        # 加入例子：
        # {examples}
        # """
    
    def add_message_to_history(self, role, content):
        """
        将消息添加到聊天历史中
        """
        self.chat_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """
        清空聊天历史
        """
        self.chat_history = []
    
    def generate_response(self, prompt, max_new_tokens=100, system_prompt=None):
        """
        根据输入的prompt生成回复
        """
        system_prompt = self.system_prompt
        # 准备模型输入
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加历史对话记录
        messages.extend(self.chat_history)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 在思考模式和非思考模式之间切换，默认为True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        start_time = time.time()

        # 执行文本生成
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # 解析思考内容
        try:
            # rindex查找151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668) - 1
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        end_time = time.time()
        inference_time = end_time - start_time

        # 将用户输入和模型回复添加到聊天历史
        # self.add_message_to_history("user", prompt)
        # self.add_message_to_history("assistant", content)

        return thinking_content, content, inference_time
    
    def chat_loop(self):
        """
        启动交互式对话循环，使用当前 QwenModel 实例。
        
        Args:
            system_prompt: 系统提示词（可选）
        """
        print("开始对话，输入 'quit' 或 'exit' 退出程序，输入 'clear' 清空历史记录")
        

        while True:
            user_input = input("\n请输入您的问题: ")
            
            if user_input.lower().strip() in ['quit', 'exit', '退出']:
                print("程序已退出")
                break
            
            if user_input.lower().strip() == 'clear':
                self.clear_history()
                print("历史记录已清空")
                continue
            
            if not user_input.strip():
                print("输入不能为空，请重新输入")
                continue
            
            thinking_content, content, inference_time = self.generate_response(
                user_input, system_prompt=self.system_prompt
            )
            print("thinking content:", thinking_content)
            print("content:", content)
            print("inference time:", inference_time)
            
            if "好的，马上为您提供" in content:
                self.clear_history()
                print("已为您完成任务，聊天历史已清空")

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    qwen_model = QwenModel()
    qwen_model.chat_loop()