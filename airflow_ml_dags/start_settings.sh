#!/bin/bash

export DATA_FOR_PROD="2022-11-27"
export DATA_DIR="$(pwd)/data"
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)");
