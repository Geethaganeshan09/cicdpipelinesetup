# Use official Python image
# Use official Python base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
COPY app.py .
COPY my_model.joblib .
RUN pip install --no-cache-dir -r requirements.txt
# Command to start the Flask API
CMD ["python", "app.py"]


