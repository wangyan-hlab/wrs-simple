from distutils.core import setup

setup(
    name="wrspkg",
    version="0.1.0",
    description="wrs packages",
    author="Weiwei Wan, Yan Wang",
    py_modules=[
        'basis.*',
        'drivers.*',
        'grasping.*',
        'helper.*',
        'manipulation.*',
        'modeling.*',
        'motion.*',
        'robot_con.*',
        'robot_sim.*',
        'visualization.*',
        'fr_python_sdk.*'
    ]
)