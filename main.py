import os
import uvicorn
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Response, UploadFile, File, HTTPException, status
from sqlalchemy import text
from connect import create_connection_pool
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
logger.info("Loading the model...")
model = tf.keras.models.load_model('./best_model.h5')
logger.info("Model loaded successfully.")

app = FastAPI(title="ML Try FastAPI")

pool = create_connection_pool()

# Define class names
class_names = [
    'Cabai Sehat', 'Keriting Daun pada Cabai', 'Bercak Daun pada Cabai',
    'Kutu Putih pada Cabai', 'Virus Kuning pada Cabai',
    'Bercak Bakteri pada Tomat', 'Hawar Daun pada Tomat', 'Tomat Sehat',
    'Busuk Daun pada Tomat', 'Jamur Daun pada Tomat',
    'Virus Mosaik pada Tomat', 'Bercak Daun Septoria pada Tomat',
    'Tungau Laba-Laba pada Tomat', 'Bercak Daun pada Tomat',
    'Virus Kuning pada Tomat'
]

# In-memory storage for prediction history
prediction_history = []

# Image processing function
def process_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((150, 150))
        image = np.array(image) / 255.0
        return image
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def fetch_disease_info_by_disease_name(disease_name):
    with pool.connect() as conn:
        sql_statement = text(
            "SELECT p.plant_name, di.disease_name, di.solution, di.source, di.cause, di.prevention_method "
            "FROM plants p "
            "JOIN disease_info di ON p.plant_id = di.plant_id "
            "WHERE di.disease_name = :disease_name;"
        )
        sql_statement = sql_statement.bindparams(disease_name=disease_name)
        result = conn.execute(sql_statement)
        query_results = result.fetchall()

    # Handle missing disease info
    if not query_results:
        return []

    formatted_results = {
        'plant_name': query_results[0][0],
        'disease': query_results[0][1],
        'solution': query_results[0][2],
        'source': query_results[0][3],
        'cause': query_results[0][4],
        'prevention_method': query_results[0][5]
    }

    return formatted_results


# Health check endpoint
@app.get("/")
def index():
    return Response(content="API WORKING", status_code=200)

# Endpoint for image prediction
@app.post("/predict_image")
async def predict_image(photo: UploadFile = File(...)):
    try:
        logger.info("Received image for prediction.")
        if photo.content_type not in ["image/jpeg", "image/png"]:
            logger.error("File is not an image.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is Not an Image")

        contents = await photo.read()
        processed_image = process_image(contents)
        processed_image = np.expand_dims(processed_image, axis=0)

        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class]
        
        disease_info = fetch_disease_info_by_disease_name(predicted_class_name)
        
        if predicted_class_name in ['Cabai Sehat', 'Tomat Sehat']:
            predicition = "Tanaman kamu sehat"
            response = {
            'status' : 'success',
            'message' : 'Berhasil memprediksi gambar',
            'result' : predicition
           
        }
        else:
            predicition = "Tanaman kamu terjangkit penyakit"
            response = {
            'status' : 'success',
            'message' : 'Berhasil memprediksi gambar',
            'result':  disease_info
           
        }
            
        # Save prediction to history
        prediction_history.append(response)
        logger.info(f"Prediction: {response}")

        return JSONResponse(content=response, status_code=200)

    except HTTPException as http_err:
        logger.error(f"HTTP exception: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Endpoint to get prediction history
@app.get("/gethistory")
def get_history():
    logger.info("Fetching prediction history.")
    return JSONResponse(content={"history": prediction_history}, status_code=200)

# Endpoint to get disease information by plant name
@app.get("/disease_info/{plant_name}")
def get_disease_info(plant_name: str):
    try:
        logger.info(f"Fetching disease info for plant: {plant_name}")
        disease_info = fetch_disease_info_by_disease_name(plant_name)
        if not disease_info:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plant or disease info not found")
        return JSONResponse(content=disease_info, status_code=200)
    except HTTPException as http_err:
        logger.error(f"HTTP exception: {http_err.detail}")
        raise http_err
    except Exception as e:
        logger.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host='localhost', port=8080)
    logger.info("Server started successfully.")
