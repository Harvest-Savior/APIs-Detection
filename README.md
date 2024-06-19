# Machine Learning Model API
#### Overview
This API handles user input in the form of images of chili and tomato plant diseases and returns the disease name, diagnosis, and recommended medication for the predicted plant disease. The API checks the uploaded file format (JPEG or PNG), processes the image, predicts diseases in chili and tomato plants using machine learning models, and returns the disease name, diagnosis, and recommended medication based on data stored in the database.

# Access Our Deployed API
jan lupa isi ini los

# Endpoint Description\
###  Login for Access Token
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
### Image Prediction
- **Endpoint:** `//predict_image`
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
    "medication_recommendations": [medicine]
}
