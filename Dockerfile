# Base image for a minimal Python 3.12 runtime.
FROM python:3.12-slim

# Set the working directory inside the container.
WORKDIR /app

# Add metadata about the image.
LABEL maintainer="Pranay Sharma <pranay.sharma@example.com>" \
      org.opencontainers.image.title="ORBIT" \
      org.opencontainers.image.version="0.1.0"

# Copy dependency metadata first to maximize Docker layer caching.
COPY requirements.txt ./requirements.txt

# Install Python dependencies without caching to keep the image small.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project source into the image.
COPY . ./

# Run the smoke test at build time to verify the image is functional.
RUN python smoke_test.py

# Default command for interactive use or container startup.
CMD ["python", "smoke_test.py"]
