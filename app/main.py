from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.predict_obj import predict_obj
import os

app = FastAPI()

@app.get("/")
def root():
    html_file_path = "app/static/index.html"
    return FileResponse(html_file_path)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # write image to path
    upload_path = f"app/static/uploads/{image.filename}"
    with open(upload_path, "wb") as f:
        f.write(await image.read())
    
    # predict
    result_path = predict_obj(upload_path)

    # response
    relative_input_path = upload_path.replace("app/static/", "")
    relative_result_path = result_path.replace("app/static/", "")

    return JSONResponse(content={
        "input": f"/static/{relative_input_path}",
        "result": f"/static/{relative_result_path}"
    })

# change relative path
app.mount("/static", StaticFiles(directory="app/static"), name="static")