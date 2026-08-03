@echo off
REM ============================================================
REM MPJ-SPARK Prototype - Windows Setup & Run Script
REM For ASUS Vivobook Q533MJ (16GB RAM, 16 cores)
REM ============================================================

echo === MPJ-SPARK Prototype Setup ===

REM Step 1: Check Python
echo [Step 1] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Step 2: Install PySpark
echo [Step 2] Installing PySpark...
pip install pyspark==3.5.4

REM Step 3: Check Java (needed for Spark)
echo [Step 3] Checking Java...
java -version
if errorlevel 1 (
    echo ERROR: Java not found! Install JDK 11+ from https://adoptium.net/
    pause
    exit /b 1
)

REM Step 4: Generate test data and run prototype
echo [Step 4] Running Multi-Driver WordCount Prototype...
echo.
echo --- Option A: Python prototype (RECOMMENDED) ---
echo python mpj_spark_prototype.py --workers 4 --generate 50 --compare
echo.
echo --- Option B: With more workers ---
echo python mpj_spark_prototype.py --workers 8 --generate 100 --compare
echo.

REM Run with default settings
python mpj_spark_prototype.py --workers 4 --generate 50 --compare

pause
