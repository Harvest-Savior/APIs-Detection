# Machine Learning Model API
#### Overview
This API handles user input in the form of images of chili and tomato plant diseases and returns the disease name, diagnosis, and recommended medication for the predicted plant disease. The API checks the uploaded file format (JPEG or PNG), processes the image, predicts diseases in chili and tomato plants using machine learning models, and returns the disease name, diagnosis, and recommended medication based on data stored in the database.

# Access Our Deployed API
jan lupa isi ini los
# Table Of Content
1. **[Endpoint Description](#endpoint-description)**
    i. **[Login for Access Token (`/token`)](#login-for-access-token-token)**
    - **[Request](#request)**
    - **[Response](#response)**

    ii. **[Image Prediction (`/predict_image`)](#image-prediction-predict_image)**
    - **[Request](#request-1)**
    - **[Response](#response-1)**

    iii. **[Prediction History (`/user_predictions`)](#prediction-history-user_predictions)**
    - **[Request](#request-2)**
    - **[Response](#response-2)**
2. **[Deploying the Application to Cloud Run](#deploying-the-application-to-cloud-run)**

# Endpoint Description
###  Login for Access Token (`/token`)
#### Request:
- **Endpoint:** `/token`
- **Method:** `POST`
- **Request Body (Form-data):**
  ```plaintext
  username: usernameExample
  password: example@example.com
#### Response
- **Success (200): OK**
  ``` json
  {
    "access_token": "token",
    "token_type": "bearer"
  }
### Image Prediction (`/predict_image`)
- **Endpoint:** `/predict_image`
- **Method:** `POST`
- **Headers:**
    - `Authorization`: `Bearer <token>` : use a token from login
- **Request Body (Form-data):**
      `photo`: Image file (JPEG or PNG)
#### Response
- **Success (200): OK**
  ``` json
  {
    "status": "success",
    "message": "Berhasil memprediksi gambar",
    "image": "url image",
    "prediction": "Tanaman kamu terjangkit penyakit",
    "result": {
        "plant_name": "Tomato",
        "disease": "disease",
        "solution": "solustion",
        "prevention_method": "method"
    },
    "medication_recommendations": []
  }
- **Error (500): Internal Server Error**
   ``` json
   {
  "detail": "Internal Server Error"
  }
### Prediction History (`/user_predictions`)
- **Endpoint:** `/user_predictions`
- **Method:** `GET`
- **Headers:**
    - `Authorization`: `Bearer <token>` : use a token from login
#### Response
- **Success (200): OK**
  ``` json
  {
  "history": [
    {
      "status": "success",
      "message": "Berhasil memprediksi gambar",
      "image": "image url",
      "result": "Tanaman kamu sehat"
    }
- **Error (500): Internal Server Error**
   ``` json
   {
  "detail": "Internal Server Error"
  }
# Deploying the Application to Cloud Run
# Developers
   - [Carlos Daeli](https://github.com/carllosnd)
   - [Berlian Ndruru](https://github.com/berlianndruru)
