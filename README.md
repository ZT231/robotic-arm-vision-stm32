# 六自由度机械臂视觉抓取系统（STM32F407 + FreeRTOS）

基于 STM32F407VET6 的六自由度机械臂平台，打通 **视觉识别 → 坐标解算 → 逆运动学 → 机械臂抓取** 全链路，并支持串口 / 蓝牙 / WiFi / RFID 四种控制通道共用同一套指令协议。

> 固件：Keil MDK（CubeMX 生成框架 + 手写模块）｜ 上位机：Python + OpenCV
> 完整开发记录与 16 条踩坑分析见 [readme.txt](readme.txt)

---

## 主要功能

- **视觉引导抓取**：USB 摄像头 HSV 颜色阈值 + 轮廓检测识别目标，四点 Homography 完成像素→毫米换算，平面几何逆运动学解算 6 路舵机脉宽；按目标形状自适应夹爪方向（沿边抓取），远目标自动前倾扩臂（上限 45°）。
- **多通道控制**：串口（PC 上位机）/ 蓝牙（HC-05）/ WiFi（ESP-01S）/ RFID（RC522）四通道复用同一套 `$CMD*CK\r\n` 协议（XOR 校验），可分别独立演示。
- **模式状态机**：MANUAL（电位器手控）/ CMD（待命）/ RECORD（示教录制）/ PLAYBACK（回放）/ ESTOP（急停）五态；急停冻结输出，仅 RESUME 可解除。
- **示教回放**：上位机 REC 采集轨迹，PLAY 一键回放。
- **安全机制**：每关节独立软限位（写舵机前 clamp）、斜坡限速（25µs/20ms ≈ 112°/s）、急停冻结输出。

## 硬件平台

| 部件 | 型号 / 说明 |
|---|---|
| 主控 | STM32F407VET6 |
| 舵机驱动 | PCA9685（I2C，16 路 50Hz PWM） |
| 电位器 | 6 路 ADC 采样（DMA 循环 + 滑动平均 + 回滞死区） |
| 摄像头 | USB 摄像头（上位机 OpenCV 处理） |
| 无线 | ESP-01S（WiFi）、HC-05（蓝牙） |
| 刷卡 | RC522（GPIO 软件 SPI） |

## 软件架构

固件基于 **FreeRTOS（CMSIS-RTOS v2）** 划分三个任务，按实时性分级：

| 任务 | 周期 | 优先级 | 栈 | 职责 |
|---|---|---|---|---|
| `StartServoTask` | 20ms | High | 4096 | 电位器/按键/RFID 采集、示教、`Arm_Update()` 刷新 6 路舵机 |
| `StartProtocolTask` | 5ms | Normal | 2048 | 解析 UART 帧、分发命令、执行运动 |
| `StartPrintTask` | 3s | Low | 2048 | 周期遥测打印 |

- **中断只收字节**：ISR 内仅做单字节接收与帧状态机，命令解析与执行全部放在任务上下文（中断优先级 5，匹配 `configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY`）。
- **两把互斥锁**：`arm_lock`（机械臂状态）、`print_lock`（串口输出），从不同时持有，避免死锁。
- **协议帧格式**：`$<CMD>[,<ARG>...]*<CK>\r\n`，`CK` = CMD 字段（`$` 与 `*` 之间）所有字节的 XOR，2 位大写十六进制；响应 `$OK[,<数据>]` / `$ERR,<code>`。
- **双缓冲应答**：活动槽 + 备用槽，临界区保护，保证连发命令的应答不丢失。

## 目录结构

```
Core/          固件源码（main / usart / bt / wifi / uart_protocol / rfid /
               pca9685 / arm_control / teach / freertos 等）
Drivers/       STM32 HAL 库（ST 官方）
Middlewares/   FreeRTOS + CMSIS（官方）
MDK-ARM/       Keil 工程（Robotic arm vision.uvprojx）
tools/         PC 端工具（arm_host.py 上位机、demo.py 多通道演示、
               web_control.py 网页控制）
vision/        Python 视觉抓取（grasp.py 主程序、detector.py、
               homography.py 标定、ik.py 逆运动学、set_roi.py 工作区 ROI、
               calib_joints.py 舵机标定、config.json 参数）
docs/          标定记录、串口联调测试清单、多通道接入指南、
               全功能演示流程、视觉识别测试清单
pdf_util/      演示 PDF 辅助工具
```

## 快速开始

### 1. 固件
Keil 打开 `MDK-ARM/Robotic arm vision.uvprojx` 编译下载。

### 2. 上位机
```bash
python tools/arm_host.py --port COMx
# 蓝牙：--port COMx 接 HC-05；WiFi：--wifi 192.168.4.1:9000
```

### 3. 视觉抓取（先在 vision 目录标定）
```bash
python homography.py --source 0 --mode points --width X --height Y --out homography.json
python set_roi.py --homography homography.json --width 250 --height 200
python grasp.py --source 0 --homography homography.json --port COMx \
       --grip-conv 90 --off-x 1.5 --off-y 1.0
```

> WiFi 热点 SSID/密码在 `Core/Src/wifi.c` 的 `WIFI_AP_SSID` / `WIFI_AP_PASS` 中配置。
> 仓库中已置为占位符（`YOUR_SSID` / `YOUR_PASSWORD`），运行前请改成自己的。

## 本人负责的模块

本项目固件由 CubeMX 生成外设框架，以下模块为本人手写实现：

- **自定义 UART 指令协议**：帧状态机、XOR 校验、双缓冲应答、多串口路由，以及 WiFi `+IPD` 上报 / `AT+CIPSEND` 封包的专用适配。
- **机械臂运动控制**：斜坡限速、逐轴软限位、`Arm_Update()` 20ms 刷新、五态状态机与急停逻辑。
- **FreeRTOS 三任务架构与两把互斥锁**设计（任务划分、优先级、栈大小、锁序）。
- **驱动与功能模块**：PCA9685 驱动、电位器 ADC+DMA 采样滤波、示教录制/回放。
- **视觉抓取链路**（上位机 Python）：HSV 阈值 + 形状分类、Homography 标定、平面几何逆运动学、沿边抓取与偏心补偿。

> **第三方代码声明**：`Drivers/`（ST HAL）、`Middlewares/`（FreeRTOS、CMSIS）为 ST / FreeRTOS 官方库，非本人编写。

## 开发中遇到的问题（节选）

完整 16 条见 [readme.txt](readme.txt)。几个典型问题：

- **多串口并发时偶发数据错乱**：三路串口共享同一对发送缓冲与应答方向变量，仅在多路命令同时触发时偶现；修复为双缓冲 + 临界区保护，避免 ISR 与任务穿插修改共享资源。
- **RC522 初始化时好时坏**：SPI 模式与模块时序不匹配、RST 空闲电平偏低；恢复 SPI Mode 0 并降低波特率，增加"芯片失联自动硬件复位 + 重配寄存器"兜底。
- **RC522 刷卡无反应（REQA fail）**：定时器应答窗口远大于接收等待循环时长，时序颠倒导致永远收不到中断标志；缩短窗口至 3.2ms 并加长循环。
- **J3/J4 大角度区线性度差**：舵机大角度输出非线性且角度测量方向取反；改为 5 点分段标定并校正方向符号。
- **标定后抓取位置偏移**：标定后挪动相机/机械臂底座导致 Homography 失效；改为固定安装，并实测填入 `config.json` 的 `base_x/base_y`。

## 演示要点（建议 15 分钟流程）

1. 系统架构讲解（三层架构 + 统一指令协议 + 多串口路由）
2. PC 上位机控制：PING/STATE/HOME、单轴微调、示教 REC→PLAY 回放
3. RFID 刷卡切换：钥匙扣刷入 MANUAL 电位器手控，卡片刷回 HOME/CMD
4. 视觉自动抓取：识别 → 分步抓取 → 码放，展示沿边夹取与远目标前倾
5. 无线通道：蓝牙 APP / WiFi TCP 客户端发指令控制

## 仓库说明

- **已脱敏**：WiFi 热点 SSID/密码为占位符，蓝牙名称/密码等演示参数不含真实凭据。
- **已在 `.gitignore` 排除**：Keil 构建产物与备份副本（`MDK-ARM/Robotic arm vision/`、`*.uvoptx`、`*.0000` 等）、CMSIS DSP 官方示例（`Drivers/CMSIS/DSP/Examples/`）、Python 缓存、`pdf_util/main.pdf` 与第三方手册解包文本、上位机演示产物（`tools/outputs/`）、视觉调试残留（`vision/*_result.png`、`vision/_vidframes/` 等）。
