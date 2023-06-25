import time
import numpy as np
from fr_python_sdk.frrpc import RPC


def numpy_to_list(target):
    """
    功能: 将numpy.ndarray类型变量转换为SDK能够处理的list类型变量
    """

    target = list(target)
    for i in range(6):
        target[i] = float(target[i])
    
    return target


class FRCobot(object):
    """
        FR Cobot Movement Control Wrapper

        Author: wangyan
        Date: 2023/05/12
    """

    def __init__(self, robot_ip="192.168.58.2") -> None:
        
        self.robot = RPC(robot_ip)


    def ResetAllError(self):
        """
        功能: 尝试清除错误状态,只能清除可复位的错误
        """

        resetallerror_ret = self.robot.ResetAllError() 
        time.sleep(1.0)
        if resetallerror_ret != 0:
            print("[ERROR] 无法清除错误状态,错误不可复位,错误码:", resetallerror_ret)
        else:
            print("[INFO] 错误状态已清除,机器人成功复位")


    def GetJointPos(self, flag=0, unit="deg"):
        """
        功能: 获取机器人关节位置(角度deg或弧度rad)

        参数:
            flag ('int'):
                0: 阻塞
                1: 非阻塞
            unit ('str'):
                "deg": 角度
                "rad": 弧度
        
        返回: 关节位置[j1,j2,j3,j4,j5,j6], 单位: deg或rad
        """

        if unit == "deg":
            getjntpos_ret = self.robot.GetActualJointPosDegree(flag)
        elif unit == "rad":
            getjntpos_ret = self.robot.GetActualJointPosRadian(flag)
        else:
            raise ValueError("[WARNING] 无效的关键字")
        if getjntpos_ret[0] != 0:
            print("[ERROR] GetJointPos 失败,错误码:", getjntpos_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return
        else:
            # print(f"GetJointPos 成功,关节位置({unit}):", getjntpos_ret[1:])
            return getjntpos_ret[1:]

    
    def get_jnt_values(self):
        """
            Another name of GetJointPos(), but only in rad
        """

        self.GetJointPos(flag=0, unit="rad")


    def GetTCPPose(self, flag=0, tool="tool"):
        """
        功能: 获取机器人TCP位姿

        参数:
            flag ('int'):
                0: 阻塞
                1: 非阻塞
            tool ('str'):
                "tool": 获取工具TCP位姿
                "flange": 获取末端法兰TCP位姿

        返回: TCP位姿[x,y,z,rx,ry,rz], 单位: mm和deg
        """

        if tool == "tool":
            gettcppose_ret = self.robot.GetActualTCPPose(flag)
        elif tool == "flange":
            gettcppose_ret = self.robot.GetActualToolFlangePose(flag)
        else:
            raise ValueError("[WARNING] 无效的关键字")
        if gettcppose_ret[0] != 0:
            print("[ERROR] GetTCPPose 失败,错误码:", gettcppose_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return
        else:
            # print(f"GetTCPPose 成功,{tool} TCP位姿:", gettcppose_ret[1:])
            return gettcppose_ret[1:]


    def GetFrameNum(self, flag=0, frame="tcp"):
        """
        功能: 获取当前工具/工件坐标系编号

        参数:
            flag ('int'):
                0: 阻塞
                1: 非阻塞
            frame ('str'):
                "tcp": 工具坐标系
                “wobj”: 工件坐标系
        
        返回: 工具坐标系编号 tool_id / 工件坐标系编号 wobj_id
        """

        if frame == "tcp":
            getframenum_ret = self.robot.GetActualTCPNum(flag)
        elif frame == "wobj":
            getframenum_ret = self.robot.GetActualWObjNum(flag)
        else:
            raise ValueError("[WARNING] 无效的关键字")
        if getframenum_ret[0] != 0:
            print("[ERROR] GetFrameNum 失败,错误码:", getframenum_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return
        else:
            # print(f"GetFrameNum 成功,{frame}坐标系编号:", getframenum_ret[1])
            return getframenum_ret[1]


    def GetFrameOffset(self, flag=0, frame="tcp"):
        """
        功能: 获取当前工具/工件坐标系

        参数:
            flag ('int'):
                0: 阻塞
                1: 非阻塞
            frame ('str'):
                "tcp": 工具坐标系
                “wobj”: 工件坐标系
        
        返回: 工具/工件坐标系相对位姿[x,y,z,rx,ry,rz], 单位: mm和deg
        """

        if frame == "tcp":
            getframeoffset_ret = self.robot.GetTCPOffset(flag)
        elif frame == "wobj":
            getframeoffset_ret = self.robot.GetWObjOffset(flag)
        else:
            raise ValueError("[WARNING] 无效的关键字")
        if getframeoffset_ret[0] != 0:
            print("[ERROR] GetFrameOffset 失败,错误码:", getframeoffset_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return
        else:
            # print(f"GetFrameOffset 成功,{frame}坐标系相对位姿):", getframeoffset_ret[1:])
            return getframeoffset_ret[1:]


    def GetPayloadInfo(self, flag):
        """
        功能: 获取当前负载的质量和质心

        参数:
            flag ('int'):
                0: 阻塞
                1: 非阻塞
        
        返回: 负载质量weight, 单位: kg; 负载质心坐标[x,y,z], 单位: mm
        """

        gettgtpayload_ret = self.robot.GetTargetPayload(flag)
        if gettgtpayload_ret[0] != 0:
            print("[ERROR] GetPayloadWeight 失败,错误码:", gettgtpayload_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return
        else:
            print("[INFO] GetPayloadWeight 成功,负载质量(kg):", gettgtpayload_ret[1])
        gettgtpayloadcog_ret = self.robot.GetTargetPayloadCog(flag)
        if gettgtpayloadcog_ret[0] != 0:
            print("[ERROR] GetPayloadCOG 失败,错误码:", gettgtpayloadcog_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return
        else:
            print("[INFO] GetPayloadCOG 成功,负载质心坐标:", gettgtpayloadcog_ret[1:])
        return gettgtpayload_ret[1], gettgtpayloadcog_ret[1:]


    def FK(self, joint_pos):
        """
        功能: 正运动学求解

        参数:
            joint_pos('list[float]'): 
                关节位置[j1,j2,j3,j4,j5,j6], 单位: deg

        返回: 工具TCP位姿[x,y,z,rx,ry,rz], 单位: mm和deg
        """

        getfk_ret = self.robot.GetForwardKin(joint_pos)
        joint_pos = numpy_to_list(joint_pos)

        if getfk_ret[0] != 0:
            print("[ERROR] GetForwardKin 失败,错误码:", getfk_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return
        else:
            # print(f"GetForwardKin 成功,工具TCP位姿):", getfk_ret[1:])
            return getfk_ret[1:]


    def IK(self, flag, desc_pos, joint_pos_ref):
        """
        功能: 逆运动学求解

        参数: 
            flag ('int'):
                0: 绝对位姿(基坐标系)
                1: 相对位姿(基坐标系)
                2: 相对位姿(工具坐标系)
            desc_pos ('list[float]'):
                工具位姿[x,y,z,rx,ry,rz], 单位: mm和deg
            joint_pos_ref ('list[float]'):
                关节参考位置[j1,j2,j3,j4,j5,j6], 单位: deg
        
        返回: 关节位置[j1,j2,j3,j4,j5,j6], 单位: deg
        """

        desc_pos = numpy_to_list(desc_pos)
        joint_pos_ref = numpy_to_list(joint_pos_ref)

        ik_has_solution = self.robot.GetInverseKinHasSolution(flag, desc_pos, joint_pos_ref)
        if ik_has_solution[0] == 0:
            if ik_has_solution[1]:
                #TODO:这里需要纠错，文档中GetInverseKinRef()缺少第一个参数
                getikref_ret = self.robot.GetInverseKinRef(flag, desc_pos, joint_pos_ref)
                if getikref_ret[0] == 0:
                    target_joint_pos = getikref_ret[1:]
                    # print("GetInverseKinRef 成功,关节位置为:", target_joint_pos)
                    return target_joint_pos
                else:
                    print("[ERROR] GetInverseKinRef 失败,错误码:", getikref_ret[0])
                    self.ResetAllError() # 尝试清除错误状态
                    return # GetInverseKinHasSolution()失败则直接结束
            else:
                raise ValueError("[WARNING] 逆运动学无解")
        else:
            print("[ERROR] GetInverseKinHasSolution 失败,错误码:", ik_has_solution[0])
            self.ResetAllError() # 尝试清除错误状态
            return # GetInverseKinRef()失败则直接结束


    def JointJog(self, joint_num, joint_angle, 
                 vel=50.0, acc=50.0, max_dis=5.0, stop_mode="stopjog"):
        """
        功能: 控制机器人关节joint_num连续点动,旋转给定角度joint_angle

        参数:
            joint_num ('int'): 
                要旋转的关节编号, 可选[1,2,3,4,5,6]
            joint_angle ('float'): 
                旋转角度, 可指定正负, 单位: deg
            vel ('float'): 
                运动速度百分比, 0.0~100.0, 单位: %
            acc ('float'): 
                运动加速度百分比, 0.0~100.0, 单位: %
            max_dis ('float'): 
                单次点动最大角度, 单位: deg
            stop_mode ('str'): 
                “stopjog”: jog点动减速停止
                “immstopjog”: jog点动立即停止
        """

        ref = 0 # 关节点动
        dir = 1 if joint_angle > 0.0 else 0 # 旋转方向
        vel = float(vel)
        acc = float(acc)
        max_dis = float(max_dis)
        if max_dis <= 0.0:
            raise ValueError("max_dis must be greater than 0.0") # 单次点动最大角度必须为正
        loop_num = int(abs(joint_angle)//max_dis) if abs(joint_angle)%max_dis==0.0 \
            else int(abs(joint_angle)//max_dis)+1 # 计算点动次数
        
        for _ in range(loop_num):
            startjog_ret = self.robot.StartJOG(ref, joint_num, dir, vel, acc, max_dis)
            time.sleep(0.5*max_dis)
            if startjog_ret != 0:
                print("[ERROR] StartJOG 失败,错误码:", startjog_ret)
                self.ResetAllError() # 尝试清除错误状态
                return  # StartJOG()失败则直接结束
            if stop_mode == "stopjog":
                stopjog_ret = self.robot.StopJOG(1)
                time.sleep(1.0) 
                if stopjog_ret != 0:
                    print("[ERROR] StopJOG 失败,错误码:", stopjog_ret)
                    self.ResetAllError() # 尝试清除错误状态
                    return # StopJOG()失败则直接结束
            elif stop_mode == "immstopjog":
                immstopjog_ret = self.robot.ImmStopJOG()
                time.sleep(1.0)
                if immstopjog_ret != 0:
                    print("[ERROR] ImmStopJOG 失败,错误码:", immstopjog_ret)
                    self.ResetAllError() # 尝试清除错误状态
                    return # ImmStopJOG()失败则直接结束
        print("[INFO] JointJOG 运行成功")


    def CartJog(self, frame, dim, dis, 
                vel=50.0, acc=50.0, max_dis=5.0, stop_mode="stopjog"):
        """
        功能: 在frame坐标系下,控制机器人在给定运动自由度dim上连续点动,运动给定距离/旋转给定角度dis

        参数:
            frame ('str'): 
                "base": 基坐标系点动
                "tool": 工具坐标系点动
                "wobj": 工件坐标系点动
            dim ('int'):
                运动自由度: 1~6分别对应"x","y","z","rx","ry","rz" 
            dis ('float'): 
                运动距离/旋转角度, 可指定正负, 单位: mm或deg
            vel ('float'): 
                运动速度百分比, 0.0~100.0, 单位: %
            acc ('float'): 
                运动加速度百分比, 0.0~100.0, 单位: %
            max_dis ('float'): 
                单次点动最大距离/角度, 单位: mm或deg
            stop_mode ('str'): 
                “stopjog”: jog点动减速停止
                “immstopjog”: jog点动立即停止
        """
        
        dis = float(dis)
        vel = float(vel)
        acc = float(acc)
        max_dis = float(max_dis)
        if frame == "base":
            ref = 2 # 基坐标系点动
        elif frame == "tool":
            ref = 4 # 工具坐标系点动
        elif frame == "wobj":
            ref = 8 # 工件坐标系点动
        else:
            raise ValueError("[WARNING ]无效的关键字")
        dir = 1 if dis > 0.0 else 0 # 运动方向
        if max_dis <= 0.0:
            raise ValueError("[WARNING] max_dis must be greater than 0.0") # 单次点动最大角度必须为正
        loop_num = int(abs(dis)//max_dis) if abs(dis)%max_dis==0.0 \
            else int(abs(dis)//max_dis)+1 # 计算点动次数
        
        for _ in range(loop_num):
            startjog_ret = self.robot.StartJOG(ref, dim, dir, vel, acc, max_dis)
            time.sleep(0.5*max_dis)
            if startjog_ret != 0:
                print("[ERROR] StartJOG 失败,错误码:", startjog_ret)
                self.ResetAllError() # 尝试清除错误状态
                return  # StartJOG()失败则直接结束
            if stop_mode == "stopjog":
                stopjog_ret = self.robot.StopJOG(1)
                time.sleep(1.0) 
                if stopjog_ret != 0:
                    print("[ERROR] StopJOG 失败,错误码:", stopjog_ret)
                    self.ResetAllError() # 尝试清除错误状态
                    return # StopJOG()失败则直接结束
            elif stop_mode == "immstopjog":
                immstopjog_ret = self.robot.ImmStopJOG()
                time.sleep(1.0)
                if immstopjog_ret != 0:
                    print("[ERROR] ImmStopJOG 失败,错误码:", immstopjog_ret)
                    self.ResetAllError() # 尝试清除错误状态
                    return # ImmStopJOG()失败则直接结束
        print("[INFO] CartJOG 运行成功")


    def MoveJ(self, target_pos, target_flag="joint", 
              vel=50.0, ovl=100.0, exaxis_pos=[0.0, 0.0, 0.0, 0.0], blendT=-1.0, 
              offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        """
        功能: 控制机器人关节空间运动到目标位置

        参数:
            target_pos ('list[float]'): 
                target_flag="joint"时, 表示目标关节位置, [j1,j2,j3,j4,j5,j6], 单位: deg;
                target_flag="desc"时, 表示目标笛卡尔位姿, [x,y,z,rx,ry,rz], 单位: mm
            target_flag ('str'):
                "joint": 目标用关节位置表示
                "desc": 目标用笛卡尔位姿表示
            vel ('float'): 
                运动速度百分比, 0.0~100.0, 单位: %
            ovl ('float'): 
                速度缩放因子, 0.0~100.0, 单位: %
            exaxis_pos ('numpy.ndarray'): 
                外部轴1位置~外部轴4位置,单位mm
            blendT ('float'):
                -1.0: 运动到位(阻塞) 
                0~500: 平滑时间(非阻塞), 单位: ms
            offset_flag ('int'):
                0: 不偏移
                1: 工件/基坐标系下偏移
                2: 工具坐标系下偏移
            offset_pos ('numpy.ndarray'):
                位姿偏移量，单位:mm和°
        """

        vel = float(vel)
        ovl = float(ovl)
        blendT = float(blendT)
        acc = 0.0   # 加速度百分比，暂不开放，默认为0.0
        tool = self.GetFrameNum(frame="tcp")
        user = self.GetFrameNum(frame="wobj")
        
        if target_flag == "joint":
            target_joint_pos = target_pos
            target_joint_pos = numpy_to_list(target_joint_pos)
            target_desc_pos = self.FK(target_joint_pos)
        elif target_flag == "desc":
            target_desc_pos = target_pos
            target_desc_pos = numpy_to_list(target_desc_pos)
            joint_pos_ref = self.GetJointPos()
            target_joint_pos = self.IK(0, target_desc_pos, joint_pos_ref)
        else:
            raise ValueError("[WARNING] 无效的关键字")

        movej_ret = self.robot.MoveJ(target_joint_pos, target_desc_pos, tool, user, 
                         vel, acc, ovl, exaxis_pos, blendT, offset_flag, offset_pos)
        if movej_ret != 0:
            print("[ERROR] MoveJ 失败,错误码:", movej_ret)
            self.ResetAllError() # 尝试清除错误状态
            return # MoveJ()失败则直接结束
        
        getrbtmotiondone_ret = self.robot.GetRobotMotionDone()
        if getrbtmotiondone_ret[0] == 0:
            if getrbtmotiondone_ret[1] == 1:
                print("[INFO] MoveJ 运行成功")
            else:
                print("[INFO] MoveJ 运行未完成")
                self.ResetAllError() # 尝试清除错误状态
                return # MoveJ()未完成则直接结束
        else:
            print("[ERROR] MoveJ 失败,错误码:", getrbtmotiondone_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return # MoveJ()失败则直接结束
    

    def move_jnts(self, target_pos, target_flag="joint", 
                  vel=50.0, ovl=100.0, exaxis_pos=[0.0, 0.0, 0.0, 0.0], blendT=-1.0, 
                  offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        """
            Another name of MoveJ()
        """

        self.MoveJ(target_pos, target_flag, vel, ovl, 
                   exaxis_pos, blendT, offset_flag, offset_pos)


    def MoveL(self, target_pos, target_flag="joint", 
              vel=50.0, ovl=100.0, exaxis_pos=[0.0, 0.0, 0.0, 0.0], blendR=-1.0, 
              search=0, offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        """
        功能: 控制机器人笛卡尔空间直线运动到目标位置

        参数:
            target_pos ('list[float]'): 
                target_flag="joint"时, 表示目标关节位置, [j1,j2,j3,j4,j5,j6], 单位: deg;
                target_flag="desc"时, 表示目标笛卡尔位姿, [x,y,z,rx,ry,rz], 单位: mm
            target_flag ('str'):
                "joint": 目标用关节位置表示
                "desc": 目标用笛卡尔位姿表示
            vel ('float'): 
                运动速度百分比, 0.0~100.0, 单位: %
            ovl ('float'): 
                速度缩放因子, 0.0~100.0, 单位: %
            exaxis_pos ('numpy.ndarray'): 
                外部轴1位置~外部轴4位置,单位mm
            blendR ('double'):
                -1.0: 运动到位(阻塞)
                0~500: 平滑时间(非阻塞), 单位: ms
            search ('int'):
                0: 不焊丝寻位
                1: 焊丝寻位
            offset_flag ('int'):
                0: 不偏移
                1: 工件/基坐标系下偏移
                2: 工具坐标系下偏移
            offset_pos ('numpy.ndarray'):
                位姿偏移量，单位:mm和°
        """

        vel = float(vel)
        ovl = float(ovl)
        blendR = float(blendR)
        acc = 0.0   # 加速度百分比，暂不开放，默认为0.0
        tool = self.GetFrameNum(frame="tcp")
        user = self.GetFrameNum(frame="wobj")
        
        if target_flag == "joint":
            target_joint_pos = target_pos
            target_joint_pos = numpy_to_list(target_joint_pos)
            target_desc_pos = self.FK(target_joint_pos) # 计算目标位姿
        elif target_flag == "desc":
            target_desc_pos = target_pos
            target_desc_pos = numpy_to_list(target_desc_pos)
            joint_pos_ref = self.GetJointPos()
            target_joint_pos = self.IK(0, target_desc_pos, joint_pos_ref)
        else:
            raise ValueError("[WARNING] 无效的关键字")
        
        movel_ret = self.robot.MoveL(target_joint_pos, target_desc_pos, tool, user, 
                         vel, acc, ovl, blendR, exaxis_pos, search, offset_flag, offset_pos)
        if movel_ret != 0:
            print("[ERROR] MoveL 失败,错误码:", movel_ret)
            self.ResetAllError() # 尝试清除错误状态
            return # MoveL()失败则直接结束
        
        getrbtmotiondone_ret = self.robot.GetRobotMotionDone()
        if getrbtmotiondone_ret[0] == 0:
            if getrbtmotiondone_ret[1] == 1:
                print("[INFO] MoveL 运行成功")
            else:
                print("[INFO] MoveL 运行未完成")
                self.ResetAllError() # 尝试清除错误状态
                return # MoveL()未完成则直接结束
        else:
            print("[ERROR] MoveL 失败,错误码:", getrbtmotiondone_ret[0])
            self.ResetAllError() # 尝试清除错误状态
            return # MoveL()失败则直接结束


    def MoveJSeq(self, target_pos_seq, time_period=0.008, t_wait=0.02, granularity=0.1):
        """
        功能: 控制机器人关节空间运动到一系列目标位置

        参数:
            target_pos_seq ('list[list[float]]'): 
                目标关节位置序列, [[j1,j2,j3,j4,j5,j6],...], 单位: deg
            time_period ('float'): 
                控制指令周期, 单位: s
            granularity ('int'): 
                关节运动颗粒度, 将相邻两个关节位置分成等间隔的n份, n = 1/granularity
        """

        acc = 0.0   # 加速度百分比, 暂不开放, 默认为0.0
        vel = 0.0   # 速度百分比, 暂不开放, 默认为0.0
        filter_time = 0.0   # 滤波时间, 暂不开放, 默认为0.0
        gain = 0.0  # 目标位置的比例放大器, 暂不开放, 默认为0.0

        n = int(1/granularity)
        target_pos_seq_interp = []
        for i in range(6):
            arr = np.asarray(target_pos_seq)[:, i]
            expanded_arr = np.interp(np.linspace(0, len(arr) - 1, (len(arr)-1)*n+1), 
                                     np.arange(len(arr)), arr)
            target_pos_seq_interp.append(expanded_arr)

        target_pos_seq_smoothed  = np.asarray(target_pos_seq_interp).T    

        for index, jnt_pos in enumerate(target_pos_seq_smoothed):
            jnt_pos = numpy_to_list(jnt_pos)

            print("[INFO] ServoJ 目标关节位置:", jnt_pos)
            servoj_ret = self.robot.ServoJ(jnt_pos, acc, vel, time_period, filter_time, gain)
            time.sleep(t_wait)

            if servoj_ret != 0:
                print("[ERROR] ServoJ 失败,错误码:", servoj_ret)
                self.ResetAllError() # 尝试清除错误状态
                return # ServoJ()失败则直接结束
                
    
    def move_jntspace_path(self, target_pos_seq, time_period=0.008, t_wait=0.02, granularity=0.1, **kwargs):
        """
            Another name of MoveJSeq()
        """

        self.MoveJSeq(np.rad2deg(target_pos_seq), time_period, t_wait, granularity)
