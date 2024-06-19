# Machine Learning Model API
#### Overview
This API handles user input in the form of images of chili and tomato plant diseases and returns the disease name, diagnosis, and recommended medication for the predicted plant disease. The API checks the uploaded file format (JPEG or PNG), processes the image, predicts diseases in chili and tomato plants using machine learning models, and returns the disease name, diagnosis, and recommended medication based on data stored in the database.

# Access Our Deployed API
jan lupa isi ini los

# Endpoint Description
#### Request:

- **Endpoint:** `/token`
- **Method:** `POST`
- **Headers:**
    - `Content-Type`: `multipart/form-data`
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
