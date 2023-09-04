FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt update && \
        apt install -y \
        rsync \
        nano \
        curl \
        openssh-server \
        python3.10 \
        htop \
        screen \
        && rm -rf /var/lib/apt/lists/*

# Change SSH port
RUN sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config

# Add a new user named "clemens" and set password
RUN useradd -m -s /bin/bash clemens && \
    echo 'clemens:clemens' | chpasswd

# Add the new user to the sudo group
RUN usermod -aG sudo clemens

# Allow the new user to execute sudo without a password prompt (not recommended in production)
RUN echo 'clemens ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Restart SSH service
RUN service ssh restart

# Expose the new SSH port
EXPOSE 2222

CMD service ssh start && /bin/bash