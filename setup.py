from distutils.core import setup

setup(
    name="wrspkg",
    version="0.1.0",
    description="wrs packages",
    author="Weiwei Wan",
    py_modules=[
        'basis.*',
        'drivers.*',
        'grasping.*',
        'helper.*',
        'manipulation.*',
        'modeling.*',
        'motion.*',
        'neuro.*',
        'robot_con.*',
        'robot_sim.*',
        'vision.*',
        'visualization.*',
        'fr_python_sdk.*'
    ]
)