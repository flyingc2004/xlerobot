@echo off
chcp 65001 >nul
title 解锁COM5和COM6串口权限

:: 自动请求管理员权限
fltmc >nul 2>&1 || (
    powershell -Command "Start-Process cmd -ArgumentList '/c ""%~f0""' -Verb RunAs" >nul 2>&1
    exit /b
)

cls
echo ============== 正在解锁COM5和COM6串口权限 ==============
echo.

:: 核心修复：改用中文系统兼容的权限参数格式
echo 🔧 处理COM5...
icacls "\\.\COM5" /grant 所有人:RXW /T /C /Q
if %errorlevel%==0 (echo ✅ COM5 权限解锁成功) else (echo ❌ COM5 解锁失败（可能被占用/不存在）)

echo 🔧 处理COM6...
icacls "\\.\COM6" /grant 所有人:RXW /T /C /Q
if %errorlevel%==0 (echo ✅ COM6 权限解锁成功) else (echo ❌ COM6 解锁失败（可能被占用/不存在）)

echo.
echo ============== 操作完成 ==============
echo 提示：若失败请检查串口是否被占用（关闭串口助手/烧录工具）
echo.
pause >nul