import re
import os
import copy
import tkinter as tk
from frchat.gui import FRChatGUI
from frchat.bot_palletize import FRChatBotPalletize
from frchat.init_prompt_palletize import MSG_PALLETIZE_INTRO, WELCOME_TEXT


class FRChatGUIPalletize(FRChatGUI):
    """
        A GUI to use the FR ChatBot to generate palletization programs

        Author: wangyan
        Data: 2023/05/23
    """
    
    def __init__(self, title, width=1024, height=768, font=('Times New Roman', 10), 
                 robot_connect=False, init_prompt=MSG_PALLETIZE_INTRO):
        super().__init__(title, width, height, font)
        self.bot = FRChatBotPalletize(messages=init_prompt,temperature=0.0)
        self.init_prompt =  copy.deepcopy(init_prompt)
        # 文件保存相关
        self.yaml_name = None
        self.yaml_content = None
        self.python_name = None
        self.python_content = None
        # 图形绘制相关
        self.box_length = 10.0
        self.box_width = 8.0
        self.pallet_length = 22.0
        self.pallet_width = 18.0
        self.box_interval = 2.0
        self.nrow = 2
        self.ncol = 2
        self.first_corner = [0, 0]
        self.move_direction = None
        self.move_pattern = None
        self.canvas = None
        self.frame_canvas = None
        self.scale_factor = 300
        self.robot_connect = robot_connect
        if robot_connect:
            from frmovewrapper.frmove import FRCobot
            self.frrbt = FRCobot()


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
        ## 创建画布和输出框
        frame_right = tk.Frame(root)
        frame_right.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        ### 创建画布
        label_draw = tk.Label(frame_right, text="预览图", font=font)
        label_draw.grid(row=0, column=0, sticky="w")
        ### y scrollbar
        scrollbar_draw_y = tk.Scrollbar(frame_right, orient="vertical")
        ### x scrollbar
        scrollbar_draw_x = tk.Scrollbar(frame_right, orient="horizontal")
        self.canvas = tk.Canvas(frame_right, 
                                width=10, 
                                height=5, 
                                bg="white", 
                                xscrollcommand=scrollbar_draw_x.set,
                                yscrollcommand=scrollbar_draw_y.set)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar_draw_y.config(command=self.canvas.yview)
        scrollbar_draw_y.grid(row=1, column=1, sticky="ns")
        scrollbar_draw_x.config(command=self.canvas.xview)
        scrollbar_draw_x.grid(row=2, column=0, sticky="ew")

        self.frame_canvas = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame_canvas, anchor="nw")

        ### 创建输出框
        label_output = tk.Label(frame_right, text="小助手", font=font)
        label_output.grid(row=3, column=0, sticky="w")
        scrollbar_output = tk.Scrollbar(frame_right)
        text_output = tk.Text(frame_right, 
                            width=10, 
                            height=10, 
                            yscrollcommand=scrollbar_output.set,
                            font=font)
        text_output.grid(row=4, column=0, sticky="nsew")
        scrollbar_output.config(command=text_output.yview)
        scrollbar_output.grid(row=4, column=1, sticky="ns")
        text_output.insert("end", WELCOME_TEXT)
        
        ### 图形绘制比例尺
        frame_scalebar = tk.Frame(root)
        frame_scalebar.pack(side="right", fill="both", expand=False, padx=5, pady=5)
        scalebar = tk.Scale(root, from_=1, to=500, 
                         orient="vertical", 
                         command=self.update_scale_factor, 
                         sliderlength=10, length=100, width=10)
        scalebar.set(300)  # 默认初始值为
        scalebar.pack(side="top", padx=5, pady=5)

        ## 让文本框适应窗口大小
        frame_left.rowconfigure(1, weight=1)
        frame_left.rowconfigure(3, weight=1)
        frame_left.columnconfigure(0, weight=1)
        frame_left.columnconfigure(1, weight=0)
        frame_right.rowconfigure(1, weight=1)
        frame_right.rowconfigure(4, weight=3)
        frame_right.columnconfigure(0, weight=1)
        frame_right.columnconfigure(1, weight=0)

        ## 禁止输入历史框和输出框编辑
        text_input_history.config(state="disabled")
        text_output.config(state="disabled")

        ## 菜单栏用于导入yaml配置文件
        ### 创建一个空菜单，用于添加文件选项
        menubar = tk.Menu(root) 
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="配置文件", menu=filemenu, font=font)
        ### 将 “打开” 选项添加到菜单中
        filemenu.add_command(label="打开", command=self.open_file)
        ### 将单栏添加到主窗口中
        root.config(menu=menubar)

        ### 让输入框在每次输入后自动滚动到底部
        def on_input_change(*_):
            text_input.yview_moveto(1.0)
        text_input.bind("<KeyPress>", on_input_change)

        return text_input_history, text_input, text_output


    def match_prompt_pattern(self, *args):

        # TODO: 测试功能
        # 在用户输入“获取当前tcp位姿”后,调用GetTCPPose()函数获取末端位姿,并将结果粘贴到输入框中
        prompt = self.text_input.get("1.0", "end")
        get_pose_pattern = re.compile(r"获取当前([\s\S]*?)位姿")
        get_pose = get_pose_pattern.findall(prompt)

        if get_pose:
            if get_pose[0] == "tcp":
                if self.robot_connect:
                    tcp_pose = self.frrbt.GetTCPPose()
                    for index, data in enumerate(tcp_pose):
                        tcp_pose[index] = round(data, 3)
                    self.text_input.insert("end", f"\n{str(tcp_pose)}")
                else:
                    self.text_input.insert("end", f"\n[INFO] 请连接机器人以获取数据\n")
            else:
                raise ValueError("无效的获取目标")
        else:
            pass


    def match_response_pattern(self):

        response = self.output_content

        # 匹配yaml相关内容
        yaml_name_pattern = re.compile(r"param_([\s\S]*?)\.yaml")
        self.yaml_name = yaml_name_pattern.findall(response)
        yaml_content_pattern = re.compile(r"```yaml\n([\s\S]*?)\n```")
        self.yaml_content = yaml_content_pattern.findall(response)
        # 创建yaml文件
        if self.yaml_name and (self.yaml_name[-1] != "xxx"):
            dir = 'config/palletize/'
            if not os.path.exists(dir):
                os.mkdir(dir)
            yaml_filepath = os.path.join(dir, f'param_{self.yaml_name[-1]}.yaml')
            # 将参数配置保存到yaml文件中
            if self.yaml_content:
                with open(yaml_filepath, 'w', encoding='utf-8') as f:
                    for match in self.yaml_content:
                        f.write(f"{match}\n")
                        print("[INFO] yaml文件已输出!")
                # 为了防止token超限，使用initial prompt重新初始化
                self.reinit_prompt()
        
        # 获取参数配置中图形绘制的相关数据
        print("[INFO] 开始提取图形绘制数据") 
        box_length_match = re.compile(r"工件长度: ([\s\S]*?)\n")
        box_length = box_length_match.findall(response)
        if box_length:
            self.box_length = float(box_length[-1])
        box_width_match = re.compile(r"工件宽度: ([\s\S]*?)\n")
        box_width = box_width_match.findall(response)
        if box_width:
            self.box_width = float(box_width[-1])
        pallet_length_match = re.compile(r"前边长度: ([\s\S]*?)\n")
        pallet_length = pallet_length_match.findall(response)
        if pallet_length:
            self.pallet_length = float(pallet_length[-1])
        pallet_width_match = re.compile(r"侧边长度: ([\s\S]*?)\n")
        pallet_width = pallet_width_match.findall(response)
        if pallet_width:
            self.pallet_width = float(pallet_width[-1])
        box_interval_match = re.compile(r"工件间隔: ([\s\S]*?)\n")
        box_interval = box_interval_match.findall(response)
        if box_interval:
            self.box_interval = float(box_interval[-1])
        nrow_match = re.compile(r"每层行数: ([\s\S]*?)\n")
        nrow = nrow_match.findall(response)
        if nrow:
            self.nrow = int(nrow[-1])
        ncol_match = re.compile(r"每层列数: ([\s\S]*?)\n")
        ncol = ncol_match.findall(response)
        if ncol:
            self.ncol = int(ncol[-1])
        move_direction_match = re.compile(r"移动方向: ([\s\S]*?)\n")
        move_direction = move_direction_match.findall(response)
        if move_direction:
            self.move_direction = move_direction[-1]
        move_pattern_match = re.compile(r"运动路径: ([\s\S]*?)\n")
        move_pattern = move_pattern_match.findall(response)
        if move_pattern:
            self.move_pattern = move_pattern[-1]
        first_corner_match = re.compile(r"起始方位: ([\s\S]*?)\n")
        first_corner = first_corner_match.findall(response)
        if first_corner:
            self.first_corner = eval(first_corner[-1])
        print(f"[INFO] 图形绘制参数:\n工件长度:{self.box_length}, 工件宽度:{self.box_width}, 托盘前边长度:{self.pallet_length}, 托盘侧边长度:{self.pallet_width}, 工件间隔:{self.box_interval}, 每层行数:{self.nrow}, 每层列数:{self.ncol}, 起始方位:{self.first_corner}, 移动方向:{self.move_direction}, 运动路径:{self.move_pattern}")

        # 匹配python相关内容
        python_name_pattern = re.compile(r"palletize_([\s\S]*?)\.py")
        self.python_name = python_name_pattern.findall(response)
        python_content_pattern = re.compile(r"```python\n([\s\S]*?)\n```")
        self.python_content = python_content_pattern.findall(response)
        # 保存码垛python程序
        if self.python_name and (self.python_name[-1] != "xxx"):
            dir = 'palletize_program/'
            if not os.path.exists(dir):
                os.mkdir(dir)
            py_filepath = os.path.join(dir, f'palletize_{self.python_name[-1]}.py')
            if self.python_content:
                with open(py_filepath, 'w', encoding='utf-8') as f:
                    for match in self.python_content:
                        f.write(f"{match}\n")
                        print("[INFO] python程序已输出!")

    
    def start_gui(self, *args):
        self.text_input_history, self.text_input, self.text_output = self.create_gui()
        # 绘制示意图
        self.draw_rectangle()
        self.root.bind("<Control-Key-s>", self.process_message)
        self.root.bind("<Control-Key-s>", self.draw_rectangle, add="+")
        self.root.bind("<Control-Key-r>", self.reinit_prompt)
        self.root.bind("<Control-Key-e>", self.match_prompt_pattern)
        ## 开始事件循环
        self.root.mainloop()


    def reinit_prompt(self, *args):
        """
            重新初始化prompt
        """
        self.bot.messages.clear()
        self.bot.messages = copy.deepcopy(self.init_prompt)
        print("[Reinit] bot_messages", self.bot.messages)


    def update_scale_factor(self, *args):
        """
            图形绘制比例尺
        """
        self.scale_factor = int(*args)
        self.draw_rectangle()


    def draw_rectangle(self, *args):
        """
            根据参数配置绘制图形
        """
        # 缩放矩形的长度、宽度和摆放间隔
        scaled_box_length = self.box_length * self.scale_factor*0.01
        scaled_box_width = self.box_width * self.scale_factor*0.01
        scaled_box_interval = self.box_interval * self.scale_factor*0.01
        # print(f"[INFO] 当前缩放系数(%): {self.scale_factor}")
        # print(f"[INFO] 缩放后的图形绘制参数: {scaled_box_length}, {scaled_box_width}, {scaled_box_interval}")
        scaled_pallet_length = self.pallet_length * self.scale_factor*0.01
        scaled_pallet_width = self.pallet_width * self.scale_factor*0.01

        if any([scaled_box_length, scaled_box_width, scaled_box_interval, scaled_pallet_length, scaled_pallet_width]) == 0:
            print("[IGNORE] 存在无效的图形参数")
            return

        self.canvas.delete("rectangle")
        self.canvas.delete("pallet_origin")
        self.canvas.delete("pallet_xarrow")
        self.canvas.delete("pallet_yarrow")
        # 绘制托盘
        pallet_x1 = 10 
        pallet_y1 = 10
        pallet_x2 = pallet_x1 + scaled_pallet_length
        pallet_y2 = pallet_y1 + scaled_pallet_width
        self.canvas.create_rectangle(pallet_x1, pallet_y1, pallet_x2, pallet_y2, fill="grey", tags="rectangle")

        # 绘制工件
        first_corner_row = 0 if self.first_corner[0] == 0 else (self.nrow-1)
        first_corner_col = 0 if self.first_corner[1] == 0 else (self.ncol-1)

        for row in range(self.nrow):
            for col in range(self.ncol):
                if self.first_corner == [0,0]:
                    x1 = 10 + (scaled_box_length + scaled_box_interval) * col
                    y1 = 10 + (scaled_box_width + scaled_box_interval) * row
                    x2 = x1 + scaled_box_length
                    y2 = y1 + scaled_box_width
                elif self.first_corner == [0,1]:
                    x1 = 10+scaled_pallet_length-scaled_box_length - (scaled_box_length + scaled_box_interval) * col
                    y1 = 10 + (scaled_box_width + scaled_box_interval) * row
                    x2 = x1 + scaled_box_length
                    y2 = y1 + scaled_box_width
                elif self.first_corner == [1,0]:
                    x1 = 10 + (scaled_box_length + scaled_box_interval) * col
                    y1 = 10+scaled_pallet_width-scaled_box_width - (scaled_box_width + scaled_box_interval) * row
                    x2 = x1 + scaled_box_length
                    y2 = y1 + scaled_box_width
                elif self.first_corner == [1,1]:
                    x1 = 10+scaled_pallet_length-scaled_box_length - (scaled_box_length + scaled_box_interval) * col
                    y1 = 10+scaled_pallet_width-scaled_box_width - (scaled_box_width + scaled_box_interval) * row
                    x2 = x1 + scaled_box_length
                    y2 = y1 + scaled_box_width
                
                if [row, col] == [0, 0]:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", tags="rectangle")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="cyan", tags="rectangle")

        
        # 绘制其他元素
        ## 绘制原点
        if first_corner_col == 0:
            pallet_origin_x = scaled_box_length/2 + 10
        else:
            pallet_origin_x = scaled_box_length/2 + (10+scaled_pallet_length-scaled_box_length)
        
        if first_corner_row == 0:
            pallet_origin_y = scaled_box_width/2 + 10
        else:
            pallet_origin_y = scaled_box_width/2 + (10+scaled_pallet_width-scaled_box_width)

        circle_x1 = pallet_origin_x - 5
        circle_x2 = pallet_origin_x + 5
        circle_y1 = pallet_origin_y - 5
        circle_y2 = pallet_origin_y + 5
        self.canvas.create_oval(circle_x1, circle_y1, circle_x2, circle_y2, fill="yellow", tags="pallet_origin")
        
        ## 绘制移动方向
        if self.move_direction == 'Y':
            yarrow_x1, yarrow_y1 = pallet_origin_x, pallet_origin_y
            if first_corner_col == 0:
                yarrow_x2, yarrow_y2 = pallet_origin_x + \
                    (scaled_box_length+scaled_box_interval)*(self.ncol-1), pallet_origin_y
            else:
                yarrow_x2, yarrow_y2 = pallet_origin_x - \
                    (scaled_box_length+scaled_box_interval)*(self.ncol-1), pallet_origin_y
            
            for i in range(3):
                if first_corner_row == 0:
                    offset = i * (scaled_box_width + scaled_box_interval)
                else:
                    offset = -i * (scaled_box_width + scaled_box_interval)
                if self.move_pattern == 'headtail':
                    self.canvas.create_line(yarrow_x1, yarrow_y1+offset, yarrow_x2, yarrow_y2+offset, 
                                        arrow="last", width=3, fill="green",tags="pallet_yarrow")
                elif self.move_pattern == 'zigzag':
                    if i % 2 == 1:
                        self.canvas.create_line(yarrow_x2, yarrow_y2+offset, yarrow_x1, yarrow_y1+offset, 
                                        arrow="last", width=3, fill="green",tags="pallet_yarrow")
                    else:
                        self.canvas.create_line(yarrow_x1, yarrow_y1+offset, yarrow_x2, yarrow_y2+offset, 
                                        arrow="last", width=3, fill="green",tags="pallet_yarrow")
                else:
                    raise ValueError("无效的运动路径")
        
        elif self.move_direction == 'X':
            xarrow_x1, xarrow_y1 = pallet_origin_x, pallet_origin_y
            if first_corner_row == 0:
                xarrow_x2, xarrow_y2 = pallet_origin_x, pallet_origin_y + \
                    (scaled_box_width+scaled_box_interval)*(self.nrow-1)
            else:
                xarrow_x2, xarrow_y2 = pallet_origin_x, pallet_origin_y - \
                    (scaled_box_width+scaled_box_interval)*(self.nrow-1)
                
            for i in range(3):
                if first_corner_col == 0:
                    offset = i * (scaled_box_length + scaled_box_interval)
                else:
                    offset = -i * (scaled_box_length + scaled_box_interval)
                if self.move_pattern == 'headtail':
                    self.canvas.create_line(xarrow_x1+offset, xarrow_y1, xarrow_x2+offset, xarrow_y2, 
                                            arrow="last", width=3, fill="red",tags="pallet_xarrow")
                elif self.move_pattern == 'zigzag':
                    if i % 2 == 1:
                        self.canvas.create_line(xarrow_x2+offset, xarrow_y2, xarrow_x1+offset, xarrow_y1, 
                                            arrow="last", width=3, fill="red",tags="pallet_xarrow")
                    else:
                        self.canvas.create_line(xarrow_x1+offset, xarrow_y1, xarrow_x2+offset, xarrow_y2, 
                                            arrow="last", width=3, fill="red",tags="pallet_xarrow")
                else:
                    raise ValueError("无效的运动路径")

        elif not self.move_direction:
            yarrow_x1, yarrow_y1 = pallet_origin_x, pallet_origin_y
            yarrow_x2, yarrow_y2 = pallet_origin_x+20, pallet_origin_y
            self.canvas.create_line(yarrow_x1, yarrow_y1, yarrow_x2, yarrow_y2, arrow="last", width=3, fill="green",tags="pallet_yarrow")
            xarrow_x1, xarrow_y1 = pallet_origin_x, pallet_origin_y
            xarrow_x2, xarrow_y2 = pallet_origin_x, pallet_origin_y+20
            self.canvas.create_line(xarrow_x1, xarrow_y1, xarrow_x2, xarrow_y2, arrow="last", width=3, fill="red",tags="pallet_xarrow")

        self.frame_canvas.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
