import re
import numpy as np
from fr_python_sdk.frchat.gui import FRChatGUI
from fr_python_sdk.frchat.init_prompt_rbtcmd_test import MSG_RBTCMD_INTRO


class FRChatGUIDEMO(FRChatGUI):

    def __init__(self, title, width=1024, height=512, font=('Times New Roman', 10), 
                 robot_connect=False, init_prompt=MSG_RBTCMD_INTRO):
        super().__init__(title, width, height, font, robot_connect, init_prompt)

        self.jnt_values = np.zeros(6)
    
    def match_response_pattern(self):
        prompt = self.input_content
        response = self.output_content
        pattern = re.compile(r"```python([\s\S]*?)```")
        matches = pattern.findall(response)

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
