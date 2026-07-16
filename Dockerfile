# Use the official Python image as a parent image
FROM python:3.13-slim

# Set environment variables to prevent Python from writing .pyc files to disk
# and to ensure stdout/stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We ignore the 'Training' section of requirements.txt for production container to keep it small, 
# but pip will just install everything in requirements.txt. 
# Alternatively, if you want a smaller image, you could strip out peft/bitsandbytes. 
RUN pip install --no-cache-dir -r requirements.txt
# Force click upgrade due to gTTS/typer conflict (as debugged locally)
RUN pip install --no-cache-dir click==8.4.2 --ignore-installed

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
