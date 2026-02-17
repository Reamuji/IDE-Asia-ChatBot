from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chat import generateResponse

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.get("/")
async def root():
    return FileResponse("../html/index.html")

@app.post("/getBotResponse")
async def chat(data: ChatRequest):
    return generateResponse(data.messages)