# Use the official Python image from Docker Hub with Python 3.10
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Update setuptools and pip to the latest versions
RUN pip install --no-cache-dir --upgrade setuptools pip

# Install dependencies from requirements.txt with a longer timeout and specific index URL
RUN pip install --no-cache-dir --default-timeout=100 --index-url=https://pypi.org/simple/ -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8501 to the outside world
EXPOSE 8501

# Command to run the application when the container starts
CMD ["streamlit", "run", "app.py"]
