@echo off
REM Use %USERPROFILE% to dynamically reference the user's home directory
set BASE_PATH=%USERPROFILE%

REM Define the Python executable path
set PYTHON_EXEC=%BASE_PATH%\AppData\Local\Programs\Python\Python310\python.exe

REM Check if Python is installed
"%PYTHON_EXEC%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed at %PYTHON_EXEC%. Please install Python and try again.
    pause
    exit /b
)

REM Create the virtual environment
"%PYTHON_EXEC%" -m venv tf-GPU
if %errorlevel% neq 0 (
    echo Failed to create the virtual environment.
    pause
    exit /b
)

REM Activate the virtual environment
call tf-GPU\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required dependencies
python -m pip install tensorflow==2.10.0 tensorflow-io[tensorflow] numpy==1.25.2 scikit-learn==1.3.0 pandas==2.0.3 matplotlib==3.7.2 scikit-image==0.21.0 opencv-python ipykernel pillow==10.0.1

echo Virtual environment setup complete.
pause