# Use the official Python base image
# FROM python:3.8-slim
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8501

# Command to run the app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]