# Use official Python image
# Use official Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the model file is copied
COPY my_model.joblib /app/


# Command to start the Flask API
CMD ["python", "app.py"]

