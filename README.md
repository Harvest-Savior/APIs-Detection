# Machine Learning Model API
#### Overview
This API handles user input in the form of images of chili and tomato plant diseases and returns the disease name, diagnosis, and recommended medication for the predicted plant disease. The API checks the uploaded file format (JPEG or PNG), processes the image, predicts diseases in chili and tomato plants using machine learning models, and returns the disease name, diagnosis, and recommended medication based on data stored in the database.

# Access Our Deployed API
[Disease Detection API](https://ml-api-main-2yxfend4ya-et.a.run.app/)
# Table Of Content
1. **[Endpoint Description](#endpoint-description)**
   - **[Login for Access Token (`/token`)](#login-for-access-token-token)**
         - **[Request](#request)**
         - **[Response](#response)**
   - **[Image Prediction (`/predict_image`)](#image-prediction-predict_image)**
         - **[Request](#request-1)**
         - **[Response](#response-1)**
   - **[Prediction History (`/user_predictions`)](#prediction-history-user_predictions)**
         - **[Request](#request-2)**
         - **[Response](#response-2)**
2. **[Preparation and Requirements](#prepartion-and-requirements)**
3. **[Start APIs-Detection Locally](#start-apis-detection-locally)**
4. **[Deploying the Application to Cloud Run](#deploying-the-application-to-cloud-run)**

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
# Preparation and Requirements
Before start this repository on your local machine, there are some preparations that need to be made:
#### 1. Create Instance Database on [Google CLoud MySQL](https://cloud.google.com/sql/docs/mysql/create-instance?hl=id)
- Note the Instance name, username, password and database name to be use on this project
#### 2. Create a Cloud Storage Buckets
- Make this bucket public
#### 3. Generate a Service Account Keys for Buckets and save the key to be use on this project later
#### 4. Download [Google Cloud SDK](https://cloud.google.com/sdk/docs/install?hl=id) to be used on deployment later
# Start APIs Detection Locally
Starting this repository locally on your own machine can be done in the following way:
#### 1. Clone APIs Detection Repository
```
git clone <repository_link>
cd <repository_directory>
```
#### 2. Install all Dependencies
```
pip install -r requirements.txt
```
#### 3. Modify Database Connection
- Navigate to `connect.py`
- Change the user, password, db and instance according to your instance that you just created earlier
#### 4. Insert the Service Account and rename into `storageServiceAccount.json`
#### 5. Modify Buckets Name
- Navigate to `main.py`
- Find this code:
  ```
  # Google Cloud Storage bucket details
  bucket_name = 'hs-ml-detection'
  ```
- Change the buckets name with you current bucket name
#### 6. Run the Server Locally
```
py main.py
```
#### 7. Accessing the Endpoint
- View Swagger UI: `http://localhost:8000/docs` on your browser
# Deploying the Application to Cloud Run
- Login into your Cloud Account
  ```
  gcloud auth application-default login
  ```
- Direct into you current project
  ```
  gcloud config set project <your-project-id>
- Deploy your project into Cloud Run
  ```
  gcloud run deploy --source . | gcloud run deploy --source --port=8000
- Set the name for the server and choose the region
# Developers
   - [Carlos Daeli](https://github.com/carllosnd)
   - [Berlian Ndruru](https://github.com/berlianndruru)
