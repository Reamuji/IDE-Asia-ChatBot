from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from chat import generateResponse

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# @app.get("/")
# async def root():
#     # return FileResponse("../html/index.html")
#     return FileResponse("../html/index.html")


@app.post("/getBotResponse")
async def chat(data: ChatRequest):
    return generateResponse(data.messages)

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_DIR = BASE_DIR / "html"
app.mount("/", StaticFiles(directory=HTML_DIR, html=True), name="html")