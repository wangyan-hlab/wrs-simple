# fastsim_gui.py

## 说明
该文件实现了一个FastSimWorld类,它继承自World类,用于快速机器人仿真的GUI界面。

主要功能有:

- 创建各种GUI控件,如按钮、下拉菜单等,用于交互控制
- 模型管理: 可以导入/导出/编辑场景中的机器人模型、静态环境模型、工作物体模型
- 点位管理: 可以导入/导出/编辑机器人的各种姿态点位
- 路径管理: 可以导入/导出/编辑机器人的运动路径规划
- 任务管理: 可以设置执行路径和点位的顺序组合
- 机器人动画: 根据点位或路径数据移动并动画显示机器人
- 连接真实机器人: 可以切换到连接真实机器人进行控制

主要用到的函数有:

- create_xxx_gui: 创建各种GUI控件
- xxx_modeling: 场景建模相关函数
- xxx_teaching: 点位示教相关函数
- xxx_moving: 路径规划执行相关函数
- xxx_task: 任务编辑执行相关函数
- animation_moving: 机器人运动动画
- execute_moving: 控制真实机器人运动

这样的快速仿真界面可以方便进行机器人的编程与调试。

## 部分变量和函数简表
<table>

<tr>
<th>模块</th>
<th>变量/函数</th>  
<th>功能</th>
</tr>

<tr>
<td rowspan="9">GUI</td> 
<td>frame_main</td>
<td>主窗口框架</td>
</tr>

<tr>
<td>frame_cartesian</td>
<td>笛卡尔坐标系框架</td>
</tr>

<tr>  
<td>frame_middle</td>
<td>中间操作按钮框架</td>
</tr>

<tr>
<td>frame_joint</td>
<td>关节控制框架</td>
</tr>

<tr>
<td>frame_manager</td>
<td>管理菜单框架</td>
</tr>

<tr>
<td>xxx_button</td> 
<td>各种按钮</td>
</tr>

<tr>
<td>option_menu</td>
<td>选择框架选项</td> 
</tr>

<tr>
<td>slider_values</td>
<td>关节控制滑块和值</td>
</tr>

<td>create_xxx_gui()</td>
<td>创建各种GUI控件</td>
</tr>

<tr>
<td rowspan="8">模型</td>
<td>model_temp</td>
<td>模型信息字典</td>
</tr>

<tr>
<td>static_models</td>
<td>静态环境模型列表</td>
</tr>

<tr>
<td>wobj_models</td>
<td>工作物体模型列表</td>  
</tr>

<tr>
<td>robot_meshmodel</td>
<td>机器人视觉模型</td>
</tr> 

<tr>
<td>model_init_xxx</td>
<td>模型初始信息</td>
</tr>

<tr>
<td>xxx_modeling()</td>
<td>建模函数</td>
</tr>

<tr>
<td>edit_modeling()</td> 
<td>编辑模型函数</td>
</tr>

<tr>
<td>save/load_modeling()</td>
<td>导入导出模型</td> 
</tr>

<tr>
<td rowspan="6">点位</td>
<td>point_temp</td>
<td>点位信息字典</td>
</tr>

<tr>
<td>conf_meshmodel</td>
<td>点位预览模型</td> 
</tr>

<tr>
<td>enable_teaching()</td>
<td>开启示教</td>
</tr>

<tr>
<td>point_teaching()</td>
<td>点位采集函数</td>
</tr>

<tr>  
<td>edit_teaching()</td>
<td>编辑点位函数</td>
</tr>

<tr>
<td>save/load_teaching()</td>
<td>导入导出点位</td>
</tr>

<tr>
<td rowspan="7">路径</td>  
<td>path_temp</td>
<td>路径信息字典</td>
</tr>

<tr>
<td>path_meshmodel</td>
<td>路径预览模型</td>
</tr>

<tr>
<td>plan_moving()</td>
<td>规划路径函数</td>
</tr>

<tr>
<td>animation_moving()</td>
<td>路径动画函数</td> 
</tr>

<tr>
<td>edit_moving()</td>
<td>编辑路径函数</td>
</tr>

<tr>
<td>save/load_moving()</td> 
<td>导入导出路径</td>
</tr>

<tr>
<td>execute_moving()</td>
<td>控制执行路径</td>
</tr>

<tr>
<td rowspan="3">任务</td>
<td>task_temp</td>
<td>任务信息字典</td>
</tr>

<tr>
<td>edit_task()</td>
<td>编辑任务函数</td>  
</tr>

<tr>
<td>save/load_task()</td>
<td>导入导出任务</td>
</tr>

</table>