import time
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

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

        # 加载总提示词
        dir_path = os.path.dirname(os.path.dirname(__file__)) 
        dir_path = os.path.join(dir_path,"Prompts")

        f1 = open(os.path.join(dir_path, "system.txt"))
        system_prompt = f1.read()
        self.system_prompt = system_prompt 
    
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