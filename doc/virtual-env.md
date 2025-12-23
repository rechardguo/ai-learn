
# 创建虚拟环境（命名为'.venv'是常见约定）
python3 -m venv .venv

结构如下
```
.venv/
├── bin/            # 在 Unix/Linux 系统上
│   ├── activate    # 激活脚本
│   ├── python      # 环境 Python 解释器
│   └── pip         # 环境的 pip
├── Scripts/        # 在 Windows 系统上
│   ├── activate    # 激活脚本
│   ├── python.exe  # 环境 Python 解释器
│   └── pip.exe     # 环境的 pip
└── Lib/            # 安装的第三方库

```

# 导出requirements.txt

```
(.venv) pip freeze > requirements.txt
```

# chose a virtual environment

Ctrl + Shift + P

Python: Select Interpreter