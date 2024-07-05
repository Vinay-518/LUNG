import json
from PIL import Image
import io
import os
import numpy as np
import streamlit as st
from streamlit import session_state
from tensorflow.keras.models import load_model
from keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None
    
    
def predict(path, model):
    classes_dir = ["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"]
    img = image.load_img(path, target_size=(350, 350))
    norm_img = image.img_to_array(img) / 255
    input_arr_img = np.array([norm_img])
    pred = np.argmax(model.predict(input_arr_img))
    return classes_dir[pred]


def generate_medical_report(predicted_label):
    # Define class labels and corresponding medical information for lung diseases
    lung_info = {
        "Adenocarcinoma": {
            "report": "The patient appears to have adenocarcinoma, a type of non-small cell lung cancer. Early detection and treatment are crucial for better outcomes.",
            "preventative_measures": [
                "Avoid smoking and exposure to secondhand smoke",
                "Maintain a healthy lifestyle with a balanced diet and regular exercise",
                "Regular screening for lung cancer can help in early detection",
            ],
            "precautionary_measures": [
                "Consult with an oncologist for personalized treatment options",
                "Consider biopsy or imaging tests for further evaluation",
            ],
        },
        "Large cell carcinoma": {
            "report": "It seems like the patient is dealing with large cell carcinoma, a type of non-small cell lung cancer. Prompt medical attention is necessary to determine the best course of action.",
            "preventative_measures": [
                "Quit smoking immediately if applicable",
                "Adopt a healthy lifestyle with a focus on nutritious foods and regular physical activity",
                "Regular check-ups and screenings are important for individuals at risk",
            ],
            "precautionary_measures": [
                "Seek consultation with an oncologist for treatment options",
                "Discuss the possibility of undergoing a biopsy or imaging tests for further diagnosis",
            ],
        },
        "Normal": {
            "report": "Great news! The patient's lung health appears to be normal. It's essential to maintain a healthy lifestyle to continue promoting lung health.",
            "preventative_measures": [
                "Avoid smoking and exposure to pollutants",
                "Engage in regular physical activity to support lung function",
                "Consider regular check-ups to monitor overall health status",
            ],
            "precautionary_measures": [],
        },
        "Squamous cell carcinoma": {
            "report": "The patient seems to have squamous cell carcinoma, a type of non-small cell lung cancer. Timely intervention and treatment are necessary to manage the condition effectively.",
            "preventative_measures": [
                "Quit smoking immediately if applicable",
                "Follow a nutritious diet and maintain a healthy weight",
                "Regular screening for lung cancer can aid in early detection",
            ],
            "precautionary_measures": [
                "Consult with an oncologist for treatment options tailored to individual needs",
                "Consider undergoing further diagnostic tests such as biopsy or imaging studies",
            ],
        },
    }

    # Retrieve medical information based on predicted label
    medical_report = lung_info[predicted_label]["report"]
    preventative_measures = lung_info[predicted_label]["preventative_measures"]
    precautionary_measures = lung_info[predicted_label]["precautionary_measures"]

    # Generate conversational medical report with each section in a paragraphic fashion
    report = (
        "Lung Disease Report:\n\n"
        + medical_report
        + "\n\nPreventative Measures:\n\n- "
        + ",\n- ".join(preventative_measures)
        + "\n\nPrecautionary Measures:\n\n- "
        + ",\n- ".join(precautionary_measures)
    )
    precautions = precautionary_measures

    return report, precautions


def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")



import cv2 
import numpy as np

import sys  # Import sys module for sys.exit()

def save_lungs_image(image_file, json_file_path="data.json"):
    try:
        if image_file is None:
            st.warning("No file uploaded.")
            return

        if not session_state["logged_in"] or not session_state["user_info"]:
            st.warning("Please log in before uploading images.")
            return

        # Load user data from JSON file
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        # Find the user's information
        user_found = False
        for user_info in data["users"]:
            if user_info["email"] == session_state["user_info"]["email"]:
                user_found = True

                image = Image.open(image_file)

                if image.mode == "RGBA":
                    image = image.convert("RGB")

                # Convert image bytes to Base64-encoded string
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

                # Check if the image is grayscale using OpenCV
                image_path = "uploaded_image.jpg"  # Temporarily save the image to disk
                image.save(image_path)
                if not is_grayscale(image_path):
                    st.error("Inappropriate image: Please upload a grayscale lungs image.")
                    sys.exit()  # Exit script execution

                # Update the user's information with the Base64-encoded image string
                user_info["lungs"] = image_base64

                # Save the updated data to JSON
                with open(json_file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)

                session_state["user_info"]["lungs"] = image_base64

                st.success("Lungs image uploaded successfully!")
                return

        if not user_found:
            st.error("User not found.")

    except Exception as e:
        st.error(f"Error saving lungs image to JSON: {e}")




def is_grayscale(image_path):
    try:
        # Load the image
        image = cv2.imread(image_path)

        # Check if the image was loaded properly
        if image is None:
            raise ValueError("Image not found or unable to load.")

        # Check if the image has 3 channels and if all channels are the same
        if len(image.shape) == 3 and image.shape[2] == 3:
            if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 1], image[:, :, 2]):
                return True
            else:
                return False
        elif len(image.shape) == 2:
            return True  # Grayscale image
        else:
            raise ValueError("Unsupported image format.")
    except Exception as e:
        st.error(f"Error checking grayscale: {e}")
        return False


def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "report": None,
            "precautions": None,
            "lungs":None

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None



def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")

        # Open the JSON file and check for the 'lungs' key
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == user_info["email"]:
                    if "lungs" in user and user["lungs"] is not None:
                        image_data = base64.b64decode(user["lungs"])
                        st.image(Image.open(io.BytesIO(image_data)), caption="Uploaded lungs Image", use_column_width=True)

                    if isinstance(user_info["precautions"], list):
                        st.subheader("Precautions:")
                        for precautopn in user_info["precautions"]:
                            st.write(precautopn)                    
                    else:
                        st.warning("Reminder: Please upload lungs images and generate a report.")
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
def fetch_precautions(user_info):
    return (
        user_info["precautions"]
        if user_info["precautions"] is not None
        else "Please upload lungs images and generate a report."
    )


def main(json_file_path="data.json"):
    st.sidebar.title("Lungs CT-Scan disease prediction system")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "Upload lungs Image", "View Reports"),
        key="Lungs CT-Scan disease prediction system",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Upload lungs Image":
        if session_state.get("logged_in"):
            st.title("Upload lungs Image")
            uploaded_image = st.file_uploader(
                "Choose a lungs image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
            )
            if st.button("Upload") and uploaded_image is not None:
                st.image(uploaded_image, use_column_width=True)
                # st.success("lungs image uploaded successfully!")
                save_lungs_image(uploaded_image, json_file_path)
                model = load_model("EffNetModel.h5")
                condition = predict(uploaded_image, model)
                report, precautions = generate_medical_report(condition)

                # Read the JSON file, update user info, and write back to the file
                with open(json_file_path, "r+") as json_file:
                    data = json.load(json_file)
                    user_index = next((i for i, user in enumerate(data["users"]) if user["email"] == session_state["user_info"]["email"]), None)
                    if user_index is not None:
                        user_info = data["users"][user_index]
                        user_info["report"] = report
                        user_info["precautions"] = precautions
                        session_state["user_info"] = user_info
                        json_file.seek(0)
                        json.dump(data, json_file, indent=4)
                        json_file.truncate()
                    else:
                        st.error("User not found.")
                st.write(report)
        else:
            st.warning("Please login/signup to upload a lungs image.")

    elif page == "View Reports":
        if session_state.get("logged_in"):
            st.title("View Reports")
            user_info = get_user_info(session_state["user_info"]["email"], json_file_path)
            if user_info is not None:
                if user_info["report"] is not None:
                    st.subheader("lungs Report:")
                    st.write(user_info["report"])
                else:
                    st.warning("No reports available.")
            else:
                st.warning("User information not found.")
        else:
            st.warning("Please login/signup to view reports.")

if __name__ == "__main__":
    
    initialize_database()
    main()
