import serial
import time
import struct
import sys
import logging
import numpy as np
import rospy


class ForceSensorRS485:
    # 协议指令定义
    START_CMD = b'\xC0\xF7\xF7\xC0'  # 数据开始发送命令
    STOP_CMD = b'\xC0\xF8\xF8\xC0'   # 数据停止发送命令
    
    def __init__(self, port='/dev/ttyUSB1', baudrate=115200):
        """
        初始化RS485传感器连接
        :param port: 串口设备路径
        :param baudrate: 波特率，默认为115200
        """
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=8,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5
            )
            self.readforce_running = False
            self.forcedata = np.zeros((6,), dtype=np.float32)  # 初始化六维力数据

            print(f"已连接到串口: {port}, 波特率: {baudrate}")
            print(f"串口缓冲区大小: in_waiting={self.ser.in_waiting}")

        except serial.SerialException as e:
            print(f"无法打开串口 {port}: {e}")
            raise
    
    def start_stream(self):
        """发送开始指令启动数据流"""
        try:
            # 清空输入缓冲区
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print("已清空串口缓冲区")
            
            # 发送开始指令
            self.ser.write(self.START_CMD)
            self.ser.flush()  # 确保数据发送完成
            self.readforce_running = True
            print(f"已发送启动指令: {self.START_CMD.hex().upper()}")
            return True
        except Exception as e:
            print(f"发送启动指令失败: {e}")
            return False
        
    def stop_stream(self):
        """发送停止指令停止数据流"""
        try:
            self.ser.write(self.STOP_CMD)
            self.ser.flush()
            self.readforce_running = False
            print(f"已发送停止指令: {self.STOP_CMD.hex().upper()}")
        except Exception as e:
            print(f"发送停止指令失败: {e}")
    
    def parse_data(self, raw_data):
        """
        解析传感器数据包
        根据文档格式:
        0x20(1字节) + 0x4E(1字节) + [Fx, Fy, Fz, Mx, My, Mz](各2字节, 小端序) + 校验和(2字节)
        """
        try:
            # 验证数据长度和帧头
            if len(raw_data) != 16:
                return None
                
            if raw_data[0] != 0x20 or raw_data[1] != 0x4E:
                return None
                
            # 解析6个分量 (每个分量2字节，小端序有符号整数)
            # 跳过最后的2字节校验和
            components = struct.unpack('<6h', raw_data[2:14])
            
            # 力除以100，力矩除以1000
            fx, fy, fz, mx, my, mz = components
            values = [
                fx / 100.0,
                fy / 100.0,
                fz / 100.0,
                mx / 1000.0,
                my / 1000.0,
                mz / 1000.0
            ]
            
            return values
            
        except Exception as e:
            return None
    
    def save_forcedata(self):
        """
        启动数据流并保存六维力数据
        """
        if not self.start_stream():
            print("无法启动数据流")
            return
            
        buffer = bytearray()
        valid_count = 0
        
        print("开始接收六维力数据...")
        
        try:
            while self.readforce_running:

                st_time = time.perf_counter()

                # 读取串口数据
                data = self.ser.read(128) # 每次读取128字节
                
                if data:
                    buffer.extend(data)
                    
                    # 处理完整数据包 (每包16字节)
                    while len(buffer) >= 16:
                        # 查找有效帧头
                        header_found = False
                        for i in range(0, len(buffer) - 15):
                            if buffer[i] == 0x20 and buffer[i+1] == 0x4E:
                                # 提取完整帧
                                frame = bytes(buffer[i:i+16])
                                
                                # 解析数据
                                parsed = self.parse_data(frame)
                                
                                if parsed:
                                    valid_count += 1
                                    self.forcedata = np.array(parsed, dtype=np.float32)
                                    # print(f"接收到有效数据: {self.forcedata}")
                                    
                                # 移除已处理数据
                                del buffer[:i+16]
                                header_found = True
                                break
                        
                        if not header_found:
                            break
                
                # 暂停短暂时间避免CPU过载
                time.sleep(0.001)

                # print(f"当前接收耗时: {time.perf_counter() - st_time:.3f} 秒")
                
        except KeyboardInterrupt:
            print("用户中断接收")
        except Exception as e:
            print(f"接收异常: {str(e)}")
        finally:
            self.stop_stream()
            print(f"接收完成，共接收到有效数据包: {valid_count} 个")
    
    def measure_frequency(self, measurement_duration=5.0):
        """
        测量传感器实际数据接收频率
        :param measurement_duration: 测量持续时间(秒)
        """
        if not self.start_stream():
            print("无法启动数据流")
            return
            
        buffer = bytearray()
        valid_count = 0
        total_packet_count = 0
        start_time = time.time()
        last_packet_time = time.time()
        min_interval = 1.0
        max_interval = 0.0
        packet_intervals = []
        
        print(f"开始频率测量，持续 {measurement_duration:.1f} 秒...")
        
        try:
            while self.readforce_running and (time.time() - start_time) < measurement_duration:
                # 读取串口数据
                data = self.ser.read(128)
                total_packet_count += 1
                
                if data:
                    buffer.extend(data)
                    
                    # 处理完整数据包 (每包16字节)
                    while len(buffer) >= 16:
                        # 查找有效帧头
                        header_found = False
                        for i in range(0, len(buffer) - 15):
                            if buffer[i] == 0x20 and buffer[i+1] == 0x4E:
                                # 计算当前包与前一包的时间间隔
                                current_time = time.time()
                                interval = current_time - last_packet_time
                                last_packet_time = current_time
                                
                                # 更新最小/最大间隔
                                if interval < min_interval:
                                    min_interval = interval
                                if interval > max_interval:
                                    max_interval = interval
                                
                                # 记录间隔
                                packet_intervals.append(interval)
                                
                                # 提取完整帧
                                frame = bytes(buffer[i:i+16])
                                
                                # 解析数据（这里只是为了验证数据格式）
                                parsed = self.parse_data(frame)
                                print(type(parsed))
                                print(f"数据: {parsed} ")
                                
                                if parsed:
                                    valid_count += 1
                                
                                # 移除已处理数据
                                del buffer[:i+16]
                                header_found = True
                                break
                        
                        if not header_found:
                            break
                    
                    # 实时显示频率
                    if valid_count > 0 and time.time() - start_time > 0.1:
                        elapsed = time.time() - start_time
                        current_freq = valid_count / elapsed
                        sys.stdout.write(f"\r当前频率: {current_freq:.1f} Hz | 有效数据包: {valid_count} | 总数据包: {total_packet_count}         ")
                        sys.stdout.flush()
                
                # 暂停短暂时间避免CPU过载
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("用户中断测量")
        except Exception as e:
            print(f"测量异常: {str(e)}")
        finally:
            self.stop_stream()
            actual_duration = time.time() - start_time
            
            # 计算统计结果
            if actual_duration > 0.1 and valid_count > 0:
                avg_frequency = valid_count / actual_duration
                
                # 计算平均间隔
                avg_interval = sum(packet_intervals) / len(packet_intervals) if packet_intervals else 0
                
                print("\n测量完成:")
                print(f"┌{'─'*50}┐")
                print(f"│ 测量时长: {actual_duration:.3f} 秒")
                print(f"│ 有效数据包: {valid_count} 个")
                print(f"│ 理论频率: 500 Hz (根据文档)")
                print(f"│ 平均频率: {avg_frequency:.2f} Hz")
                print(f"│ 最大频率: {1 / min_interval:.2f} Hz")
                print(f"│ 最小频率: {1 / max_interval:.2f} Hz" if max_interval > 0 else "│ 最小频率: N/A")
                print(f"│ 平均间隔: {avg_interval*1000:.3f} ms")
                print(f"└{'─'*50}┘")
                
                # 输出频率变化统计
                if len(packet_intervals) > 100:
                    intervals_sorted = sorted(packet_intervals)
                    p95 = intervals_sorted[int(len(intervals_sorted) * 0.95)]
                    p99 = intervals_sorted[int(len(intervals_sorted) * 0.99)]
                    print(f"间隔分布统计 (毫秒):")
                    print(f"  最小值: {min_interval*1000:.3f} ms ({1/min_interval:.0f} Hz)")
                    print(f"  最大值: {max_interval*1000:.3f} ms ({1/max_interval:.0f} Hz)")
                    print(f"  平均值: {avg_interval*1000:.3f} ms")
                    print(f"  P95:   {p95 * 1000:.3f} ms ({1/p95:.0f} Hz)")
                    print(f"  P99:   {p99 * 1000:.3f} ms ({1/p99:.0f} Hz)")
            else:
                print(f"未能获取有效数据包，请检查传感器连接和配置")
                
            # 保存原始数据用于后续分析
            # self.save_interval_data(packet_intervals)

if __name__ == "__main__":
    # 根据实际设备修改端口号
    SENSOR_PORT = '/dev/ttyUSB1'  # Linux
    # SENSOR_PORT = 'COM3'          # Windows
    
    print("启动六维力传感器频率测试")
    
    try:
        # 创建传感器实例
        sensor = ForceSensorRS485(port=SENSOR_PORT)
        
        # 测试接收频率，持续10秒
        # sensor.measure_frequency(measurement_duration=100)

        # 启动数据流并测量频率
        sensor.save_forcedata()
        
    except Exception as e:
        print(f"程序错误: {str(e)}")
    finally:
        if 'sensor' in locals():
            sensor.stop_stream()
            sensor.ser.close()
            print("串口已关闭")
        print("程序退出")