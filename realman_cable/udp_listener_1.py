import socket
import json
import time



def send_cmd(client, cmd_6axis):
    ret = client.send(cmd_6axis.encode('utf-8'))
    # Optional: Receive a response from the server
    ret = client.recv(1024).decode()
    return ret


# 机械臂 ip
ip = '192.168.99.18'
port_no = 8080

# Create a socket and connect to the server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((ip, port_no))
print("机械臂第一次连接", ip)

# set_realtime_push = '{"command": "set_realtime_push", "cycle": 5, "enable":true, "force_coordinate":2, "port": 8099,"ip": "192.168.99.100"}'
# print(set_realtime_push)

# point6_00 = '{"command":"set_realtime_push", "cycle":5, "enable":true, "port":8089, "ip":"192.168.99.100", ' \
#             '"force_cordinate":0}\r\n '

# result = send_cmd(client, set_realtime_push)
# print('设置回传端口' + str(result))

# get_realtime_push = '{"command":"get_realtime_push"}'
# i=0
# j=0
# ret = send_cmd(client, get_realtime_push)
# print("设置回传" + ret)

# 设置要监听的 IP 和端口
UDP_IP = "192.168.99.100"
UDP_PORT = 8099

# 创建一个 UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sta_time = time.time()
# print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}...")
hand1 = None
# while time.time()-sta_time<1:
while True:
    st_time = time.time()
    # 接收 UDP 数据
    data, addr = sock.recvfrom(1024)  # 缓冲区大小为 1024 字节
    print(f"Time taken to receive data: {time.time() - st_time:.4f} seconds")
    # print(f"Received message from {addr}: {data}")
    # i += 1
    try:
        # 尝试解析 JSON 数据
        message = json.loads(data)
        # print(json.dumps(message, indent=4))

        # 获取当前时间戳
        timestamp = time.time()
        # 转换为可读性更好的格式
        current_time = time.ctime(timestamp)
        end_time = time.ctime(timestamp)
        # 输出当前时间
        print(current_time)

        # 访问特定的字段
        # # arm_err = message.get("arm_err")
        # hand = message.get("hand")
        # hand_angle = hand.get("hand_angle", {})
        # if hand1 is not None and hand_angle[0] != hand1:
        #     j+=1
        # hand1 = hand_angle[0]
        joint_status = message.get("joint_status", {})
        joint_current = joint_status.get("joint_current", [])
        joint_en_flag = joint_status.get("joint_en_flag", [])
        joint_err_code = joint_status.get("joint_err_code", [])
        joint_position = joint_status.get("joint_position", [])
        joint_temperature = joint_status.get("joint_temperature", [])
        joint_voltage = joint_status.get("joint_voltage", [])
        state = message.get("state")
        sys_err = message.get("sys_err")
        waypoint = message.get("waypoint", {})
        euler = waypoint.get("euler", [])
        position = waypoint.get("position", [])
        quat = waypoint.get("quat", [])
        six_force_sensor = message.get("six_force_sensor")
        force = six_force_sensor.get("zero_force", [])


        # 打印解析后的字段
        # print(f"Arm Error: {arm_err}")
        print(f"Joint Current: {joint_current}")
        print(f"Joint Enable Flags: {joint_en_flag}")
        print(f"Joint Error Codes: {joint_err_code}")
        print(f"Joint Positions: {joint_position}")
        print(f"Joint Temperatures: {joint_temperature}")
        print(f"Joint Voltages: {joint_voltage}")
        print(f"State: {state}")
        print(f"System Error: {sys_err}")
        print(f"Waypoint Euler Angles: {euler}")
        print(f"Waypoint Position: {position}")
        print(f"Waypoint Quaternion: {quat}")
        print(f"force: {six_force_sensor}")
        print(f"zero force: {force}")
        # print(f"hand: {hand_angle}")
        # hand_angle = hand_angle[0]








    except json.JSONDecodeError:
        print("Received data is not valid JSON")

