from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import asyncio

# 创建FastAPI应用
app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)
# 定义请求模型
class ChatRequest(BaseModel):
    message: str

# 添加FastAPI路由
@app.post("/handle_order")
async def chat_endpoint(request: ChatRequest):
    if "USB" in request.message or "usb" in request.message:
        os.system("python3 ./force_control_realman.py")
        return {"status": "success", "response": "正在执行操作。"}
    else:
        return {"status": "success", "response": "当前无法执行该操作。"}

async def main():
    # 启动FastAPI服务器
    config = uvicorn.Config(app, host="192.168.12.52", port=8282, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
    