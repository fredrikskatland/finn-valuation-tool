# Use the official Python slim image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application folder and subfolders into the container
COPY data /app/data
COPY models /app/models
COPY my_model /app/my_model
COPY utils /app/utils
COPY schemas /app/schemas
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
