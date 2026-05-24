# Codex Project Instructions

## Python environment

This project must use the Conda environment located at:

```text
E:\JetBrains\Anaconda3\envs\pytorch
```

Use this Python interpreter for all Python execution:

```text
E:\JetBrains\Anaconda3\envs\pytorch\python.exe
```

Do not use:

* system Python
* Microsoft Store Python
* PyCharm's default interpreter unless it points to this exact environment
* Conda base environment
* any other Conda environment

## Running Python files

When running Python scripts in this project, always use the explicit interpreter path:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe script_name.py
```

For example:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe main.py
```

If command-line arguments are needed, use:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe main.py --arg value
```

## Running modules

When running Python modules, use:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe -m module_name
```

For example:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe -m pytest
```

## Installing packages

Do not install packages into base or system Python.

When using pip, always use:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe -m pip install package_name
```

When checking installed packages, use:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe -m pip list
```

## Environment verification

Before running or debugging Python code, verify the interpreter with:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe -c "import sys; print(sys.executable)"
```

The expected output must be:

```text
E:\JetBrains\Anaconda3\envs\pytorch\python.exe
```

If the output points anywhere else, stop and correct the command before continuing.

## Testing

When running tests, use:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe -m pytest
```

If pytest is not installed, install it only into this environment:

```bat
E:\JetBrains\Anaconda3\envs\pytorch\python.exe -m pip install pytest
```

## Execution rule

Never run commands like:

```bat
python main.py
pip install package_name
pytest
```

unless it has first been confirmed that `python`, `pip`, and `pytest` resolve to:

```text
E:\JetBrains\Anaconda3\envs\pytorch
```

Prefer explicit full paths over activated shells.
