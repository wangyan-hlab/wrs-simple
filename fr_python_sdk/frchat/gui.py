import re
import copy
import numpy as np
import tkinter as tk
from tkinter import filedialog
import datetime
from fr_python_sdk.frchat.bot import FRChatBot
from fr_python_sdk.frchat.init_prompt_rbtcmd import MSG_RBTCMD_INTRO
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv("dev.env"))  # read local .env file


class FRChatGUI(object):
    """
        A GUI to use the FR ChatBot

        Author: wangyan, shujian, chatgpt
        Date: 2023/05/23
    """

    def __init__(self, title, width=1024, height=512, font=('Times New Roman', 10), 
                 robot_connect=False, init_prompt=MSG_RBTCMD_INTRO):
        self.bot = FRChatBot(init_prompt, temperature=0.1, history_num_to_del=0)
        self.init_prompt =  copy.deepcopy(init_prompt)
        self.title = title
        self.width = width
        self.height = height
        self.font = font
        self.root = tk.Tk()
        self.input_content = None
        self.output_content = None
        self.text_input_history = None
        self.text_input = None
        self.text_output = None
        self.robot_connect = robot_connect


    def create_gui(self):
        """
            Create the GUI
        """
        ## 创建主窗口
        root = self.root
        root.title(self.title)
        root.geometry(f"{self.width}x{self.height}")
        font = self.font

        ## 创建输入历史框和用户输入框
        frame_left = tk.Frame(root)
        frame_left.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        ### 创建输入历史框
        label_input = tk.Label(frame_left, text="输入历史", font=font)
        label_input.grid(row=0, column=0, sticky="w")
        scrollbar_input_history = tk.Scrollbar(frame_left)
        text_input_history = tk.Text(frame_left, 
                                    width=10, 
                                    height=10, 
                                    yscrollcommand=scrollbar_input_history.set,
                                    font=font)
        text_input_history.grid(row=1, column=0, sticky="nsew")
        scrollbar_input_history.config(command=text_input_history.yview)
        scrollbar_input_history.grid(row=1, column=1, sticky="ns")
        ### 创建用户输入框
        label_input = tk.Label(frame_left, text="请输入(回车换行,Ctrl+s发送)", font=font)
        label_input.grid(row=2, column=0, sticky="w")
        scrollbar_input = tk.Scrollbar(frame_left)
        text_input = tk.Text(frame_left, 
                            width=10, 
                            height=10, 
                            yscrollcommand=scrollbar_input.set,
                            font=font)
        text_input.grid(row=3, column=0, sticky="nsew")
        scrollbar_input.config(command=text_input.yview)
        scrollbar_input.grid(row=3, column=1, sticky="ns")
        ## 创建输出框
        frame_right = tk.Frame(root)
        frame_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        label_output = tk.Label(frame_right, text="小助手", font=font)
        label_output.grid(row=0, column=0, sticky="w")
        scrollbar_output = tk.Scrollbar(frame_right)
        text_output = tk.Text(frame_right, 
                            width=10, 
                            height=20, 
                            yscrollcommand=scrollbar_output.set,
                            font=font)
        text_output.grid(row=1, column=0, sticky="nsew")
        scrollbar_output.config(command=text_output.yview)
        scrollbar_output.grid(row=1, column=1, sticky="ns")

        ## 让文本框适应窗口大小
        frame_left.rowconfigure(1, weight=1)
        frame_left.rowconfigure(3, weight=1)
        frame_left.columnconfigure(0, weight=1)
        frame_left.columnconfigure(1, weight=0)
        frame_right.rowconfigure(1, weight=1)
        frame_right.columnconfigure(0, weight=1)
        frame_right.columnconfigure(1, weight=0)

        ## 禁止输入历史框和输出框编辑
        text_input_history.config(state="disabled")
        text_output.config(state="disabled")

        ## 菜单栏用于导入yaml配置文件
        ### 创建一个空菜单，用于添加文件选项
        menubar = tk.Menu(root) 
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="配置文件", menu=filemenu)
        ### 将 “打开” 选项添加到菜单中
        filemenu.add_command(label="打开", command=self.open_file)
        ### 将单栏添加到主窗口中
        root.config(menu=menubar)

        ### 让输入框在每次输入后自动滚动到底部
        def on_input_change(*_):
            text_input.yview_moveto(1.0)
        text_input.bind("<KeyPress>", on_input_change)

        return text_input_history, text_input, text_output


    def open_file(self):
        """
            定义打开文件方法
        """
        filename = filedialog.askopenfilename(filetypes=[("yaml files", "*.yaml")])
        if filename:
            print(f"选择的配置文件: {filename}")
            # 读取配置文件内容并添加到输入框
            output = self.bot.read_config(filename)
            self.text_input.insert("end", str(output))


    def start_gui(self):
        self.text_input_history, self.text_input, self.text_output = self.create_gui()
        self.root.bind("<Control-Key-s>", self.process_message)
        self.root.bind("<Control-Key-r>", self.reinit_prompt)
        ## 开始事件循环
        self.root.mainloop()


    def reinit_prompt(self, *args):
        """
            重新初始化prompt
        """
        self.bot.messages.clear()
        self.bot.messages = copy.deepcopy(self.init_prompt)
        print("[Reinit] bot_messages", self.bot.messages)


    def save_input_history(self, message):
        """
            保存输入历史并添加时间戳
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.text_input_history.configure(state="normal")
        self.text_input_history.insert("end", f"\n====={now}=====\n{message}")
        self.text_input_history.configure(state="disabled")
        self.text_input_history.see('end')


    def process_message(self, *args):
        """
            处理用户输入并返回消息
        """
        self.input_content = self.text_input.get("1.0", "end")
        self.match_prompt_pattern()
        if self.input_content:            
            self.text_input.delete("1.0", "end")
            self.save_input_history(self.input_content)
            self.text_input.see('end')
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.output_content = self.bot.chat(self.input_content)

            self.match_response_pattern()
            self.text_output.configure(state="normal")
            self.text_output.insert("end", f"\n-----{now}-----\n{self.output_content}\n")
            self.text_output.configure(state="disabled")
            self.text_output.see('end')


    def match_prompt_pattern(self):
        pass
    

    def match_response_pattern(self):
        """
            Matching certain patterns in the response, e.g.: yaml, python, etc.
        """
        prompt = self.input_content
        response = self.output_content
        pattern = re.compile(r"```python([\s\S]*?)```")
        matches = pattern.findall(response)

        # 将匹配到的内容保存到 test.py 文件中
        if matches:
            with open('test.py', 'w', encoding='utf-8') as f:
                for match in matches:
                    f.write(f"{match}\n")
            if self.robot_connect:
                try:
                    from test import main
                    main()
                    print('运行成功')
                except Exception as e:
                    print('所有的错误，我都在这里处理掉%s' % e)
                    # 按照错误类型重新修改,并重新提交问题
                    error_question_prompt=str(e)
                    response = self.bot.chat(prompt+error_question_prompt)
                    self.output_content = response
                    pattern = re.compile(r"```python([\s\S]*?)```")
                    matches = pattern.findall(self.output_content)
                    with open('test.py', 'w', encoding='utf-8') as f:
                        for match in matches:
                            f.write(f"{match}\n")
            else:
                print("[INFO] 未连接机器人,只保存程序文件,不进行纠错")
