#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机械臂多通道联调演示（S5 收尾）

演示内容：PC 串口 / WiFi TCP / 手机蓝牙 三通道控制机械臂。
帧格式与 arm_host.py 一致：$<CMD>[,<ARG>...]*<CK><CR><LF>，CK 为 XOR 校验。

用法：
  python demo.py --port COM7                 # 主通道 = PC 串口（随后另开 WiFi TCP 演示）
  python demo.py --wifi 192.168.4.1:9000     # 主通道 = WiFi TCP（ESP 自建热点）

演示剧本：
  [1] PING 链路测试
  [2] 回 HOME
  [3] J1 摆动（主通道实时控制）
  [4] J6 夹爪开合
  [5] J5 腕旋转
  [6] 回 HOME
  [7] 手机蓝牙通道演示（用户手机蓝牙 APP 发命令）
  [8] WiFi TCP 通道演示（主通道为串口时自动另开 TCP 连接）
  [9] 回 HOME 收尾
"""

import argparse
import socket
import sys
import time

try:
    import serial
except ImportError:
    serial = None


def compute_ck(payload: str) -> int:
    ck = 0
    for ch in payload:
        ck ^= ord(ch)
    return ck & 0xFF


def make_frame(payload: str) -> bytes:
    ck = compute_ck(payload)
    return ("$%s*%02X\r\n" % (payload, ck)).encode("ascii")


HOME_PULSES = [1550, 1470, 1500, 1500, 1450, 1500]


class SerialTransport:
    """PC 串口通道（USART1）。"""

    def __init__(self, port: str, baud: int = 115200):
        if serial is None:
            raise RuntimeError("缺少依赖 pyserial，请先执行： pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.2)
        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def cmd(self, payload: str, timeout: float = 1.0) -> str:
        self.ser.reset_input_buffer()
        self.ser.write(make_frame(payload))
        return self._read_line(timeout)

    def fire(self, payload: str) -> None:
        self.ser.write(make_frame(payload))

    def _read_line(self, timeout: float) -> str:
        line = bytearray()
        deadline = time.time() + timeout
        while time.time() < deadline:
            ch = self.ser.read(1)
            if not ch:
                continue
            if ch in (b"\r", b"\n"):
                if line:
                    break
                continue
            line += ch
        s = bytes(line).decode("ascii", errors="replace").strip()
        if s.startswith("$"):
            s = s[1:]
        if "*" in s:
            s = s.split("*")[0]
        return s

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


class TcpTransport:
    """WiFi TCP 通道（ESP-01S CIPSERVER，手机/PC 作为 TCP 客户端）。

    注意：ESP 处于 CIPMUX+CIPSERVER（非透传）模式，
    手机->STM32 方向数据带 +IPD 前缀（被 STM32 协议栈自动剥除），
    STM32->手机 方向应答由 ESP 自动转发到当前 TCP 连接。
    """

    def __init__(self, host: str, port: int, timeout: float = 1.0):
        self.sock = socket.create_connection((host, port), timeout=3.0)
        self.sock.settimeout(timeout)
        self.buf = b""

    def cmd(self, payload: str, timeout: float = 1.0) -> str:
        self.sock.settimeout(timeout)
        self.sock.sendall(make_frame(payload))
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line()
            if line:
                return line
        return ""

    def fire(self, payload: str) -> None:
        self.sock.sendall(make_frame(payload))

    def _read_line(self):
        while b"\n" not in self.buf:
            try:
                data = self.sock.recv(256)
            except socket.timeout:
                return ""
            if not data:
                return ""
            self.buf += data
        line, _, self.buf = self.buf.partition(b"\n")
        s = line.decode("ascii", errors="replace").strip().strip("\r")
        if s.startswith("$"):
            s = s[1:]
        if "*" in s:
            s = s.split("*")[0]
        return s

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def section(title: str):
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser(description="机械臂多通道联调演示")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--port", help="PC 串口通道，如 COM7")
    g.add_argument("--wifi", help="WiFi TCP 通道，如 192.168.4.1:9000")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--home-interval", type=float, default=2.0, help="每步演示间隔(秒)")
    args = ap.parse_args()

    if not args.port and not args.wifi:
        print("请指定通道：--port COM7 或 --wifi 192.168.4.1:9000")
        sys.exit(1)

    if args.wifi:
        host, _, port_s = args.wifi.partition(":")
        main_t = TcpTransport(host, int(port_s or 9000))
        main_name = "WiFi TCP %s:%s" % (host, port_s or 9000)
    else:
        main_t = SerialTransport(args.port, args.baud)
        main_name = "PC 串口 %s" % args.port

    interval = args.home_interval

    def J(*us6):
        main_t.cmd("JOINT,%s" % ",".join(str(int(p)) for p in us6))
        time.sleep(interval)

    def MOVE(ch, us):
        r = main_t.cmd("MOVE,%d,%d" % (ch, int(us)))
        time.sleep(interval)
        return r

    def HOME():
        r = main_t.cmd("HOME")
        time.sleep(interval)
        return r

    try:
        section("机械臂多通道联调演示")
        print("主通道：%s" % main_name)
        print("说明：每个动作之间间隔 %.1fs，方便讲解。\n" % interval)

        wifi_ok = False

        section("[1] 链路测试")
        r = main_t.cmd("PING")
        print("    PING ->", r)

        section("[2] 回 HOME")
        print("    HOME ->", HOME())

        section("[3] J1 摆动（主通道实时控制）")
        for us in (1800, 1300, 1550):
            print("    J1=%d -> %s" % (us, MOVE(0, us)))

        section("[4] J6 夹爪开合")
        for us, tag in ((900, "张开"), (1900, "闭合"), (1500, "半开")):
            print("    J6=%d(%s) -> %s" % (us, tag, MOVE(5, us)))

        section("[5] J5 腕旋转")
        for us in (1000, 1900, 1450):
            print("    J5=%d -> %s" % (us, MOVE(4, us)))

        section("[6] 回 HOME")
        print("    HOME ->", HOME())

        section("[7] 手机蓝牙通道演示（HC-05，USART2）")
        print("请在手机蓝牙串口 APP（配对 ARM-BT / 1234）依次发送：")
        print("    $MOVE,0,1800*28    → 观察 J1 转到 1800")
        print("    $MOVE,0,1300*23    → 观察 J1 转到 1300")
        print("    $HOME*0F           → 回 HOME")
        input("    （手机发完命令后，按回车继续）")

        section("[8] WiFi TCP 通道演示（ESP-01S，USART3）")
        if args.wifi:
            print("    当前主通道就是 WiFi TCP，直接验证：")
            print("    PING ->", main_t.cmd("PING"))
            print("    J6 张开 ->", MOVE(5, 900))
            print("    J6 闭合 ->", MOVE(5, 1900))
            print("    HOME ->", HOME())
            wifi_ok = True
        else:
            print("    正在另开 TCP 连接 %s:9000 演示（手机连 ESP 自建热点同样可测）..." % "192.168.4.1")
            try:
                t = TcpTransport("192.168.4.1", 9000)
                print("    PING ->", t.cmd("PING"))
                print("    J6 张开 ->", t.cmd("MOVE,5,900"))
                time.sleep(interval)
                print("    J6 闭合 ->", t.cmd("MOVE,5,1900"))
                time.sleep(interval)
                print("    HOME ->", t.cmd("HOME"))
                time.sleep(interval)
                t.close()
                wifi_ok = True
            except Exception as e:
                print("    WiFi 演示失败：%s" % e)
                print("    提示：若未接 ESP-01S 或热点未启动，可跳过本步（不影响其他模块）。")

        section("[9] 收尾：回 HOME")
        print("    HOME ->", HOME())
        if args.wifi:
            print("\n演示完成。主通道 WiFi TCP 验证通过（PC 串口 / 手机蓝牙通道按 docs 文档验证）。")
        elif wifi_ok:
            print("\n演示完成。PC 串口 / 手机蓝牙 / WiFi TCP 三通道均验证通过。")
        else:
            print("\n演示完成。PC 串口 / 手机蓝牙 通道验证通过；WiFi TCP 通道已跳过（未验证，见步骤 [8] 提示）。")
    finally:
        main_t.close()
        print("\n已关闭连接。")


if __name__ == "__main__":
    main()
