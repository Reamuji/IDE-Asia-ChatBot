from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return "a a maiku ?"

# @app.get("/items/{item_id}")
# async def read_item(item_id: str):
#     return {"item_id": item_id}