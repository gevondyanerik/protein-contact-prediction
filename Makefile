IMAGE_NAME = protein_contact_prediction_image
CONTAINER_NAME = protein_contact_prediction_container

# Build the Docker image
build-image:
	docker build -t $(IMAGE_NAME) .
	@echo "Docker image protein_contact_prediction_image built."

# Run the container in detached mode (if not already running).
# The container will start the training process (in the background) and remain alive.
start-container:
	@if docker ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Container $(CONTAINER_NAME) is already running."; \
	elif docker ps -aq -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Container $(CONTAINER_NAME) exists but is stopped. Starting container..."; \
		docker start $(CONTAINER_NAME) >/dev/null 2>&1; \
		echo "Container $(CONTAINER_NAME) started."; \
	else \
		echo "Container not found. Running container..."; \
		docker run -d --name $(CONTAINER_NAME) \
		-v $(shell pwd):/workdir \
		-p 8888:8888 -p 5000:5000 $(IMAGE_NAME) >/dev/null 2>&1; \
		echo "Container $(CONTAINER_NAME) started."; \
	fi

# Open an interactive shell in the running container.
enter-container: start-container
	docker exec -it $(CONTAINER_NAME) bash

# Setup tunnel using ./scripts/setup_mlflow_tunnel.sh
# Launch MLflow UI inside the container (http://localhost:5000).
start-mlflow: start-container
	@if docker exec $(CONTAINER_NAME) pgrep -f "mlflow ui" >/dev/null; then \
		echo "MLflow UI is already running."; \
	else \
		echo "MLflow UI not running. Starting MLflow UI..."; \
		docker exec -it $(CONTAINER_NAME) mlflow ui --host 0.0.0.0 --port 5000; \
	fi

# Setup tunnel using ./scripts/setup_jupyter_tunnel.sh
# Launch JupyterLab inside the container (http://localhost:8888).
start-jupyter: start-container
	@if docker exec $(CONTAINER_NAME) pgrep -f "jupyter lab" >/dev/null; then \
		echo "JupyterLab is already running."; \
	else \
		echo "JupyterLab not running. Starting JupyterLab..."; \
		docker exec -it $(CONTAINER_NAME) jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --NotebookApp.token='' --NotebookApp.password=''; \
	fi

# Remove the container.
remove-container:
	@if docker ps -aq -f name=$(CONTAINER_NAME) | grep -q .; then \
		if docker ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
			echo "Stopping container $(CONTAINER_NAME)..."; \
			docker stop $(CONTAINER_NAME) >/dev/null 2>&1; \
			echo "Container $(CONTAINER_NAME) stopped."; \
		else \
			echo "Container $(CONTAINER_NAME) exists but is not running."; \
		fi; \
		echo "Removing container $(CONTAINER_NAME)..."; \
		docker rm $(CONTAINER_NAME) >/dev/null 2>&1; \
		echo "Container $(CONTAINER_NAME) removed."; \
	else \
		echo "Container $(CONTAINER_NAME) does not exist."; \
	fi

# Stop the container.
stop-container:
	@if docker ps -q -f name=$(CONTAINER_NAME) | grep -q .; then \
		echo "Stopping container $(CONTAINER_NAME)..."; \
		docker stop $(CONTAINER_NAME) >/dev/null 2>&1; \
		echo "Container $(CONTAINER_NAME) stopped."; \
	else \
		echo "Container $(CONTAINER_NAME) is not running."; \
	fi

# Restart the container.
restart-container: remove-container start-container

# Stop MLflow process inside the container.
stop-mlflow:
	@if docker ps -q -f name=$(CONTAINER_NAME) | grep -q . ; then \
		if docker exec $(CONTAINER_NAME) pgrep -f "mlflow ui" >/dev/null; then \
			echo "Stopping MLflow UI..."; \
			docker exec $(CONTAINER_NAME) pkill -f "mlflow ui" || true; \
		else \
			echo "MLflow UI is not running in container $(CONTAINER_NAME)."; \
		fi \
	else \
		echo "Container $(CONTAINER_NAME) is not running. Nothing to stop for MLflow."; \
	fi

# Stop JupyterLab process inside the container.
stop-jupyter:
	@if docker ps -q -f name=$(CONTAINER_NAME) | grep -q . ; then \
		if docker exec $(CONTAINER_NAME) pgrep -f "jupyter[- ]lab" >/dev/null; then \
			echo "Stopping JupyterLab..."; \
			docker exec $(CONTAINER_NAME) pkill -f "jupyter[- ]lab" || true; \
		else \
			echo "JupyterLab is not running in container $(CONTAINER_NAME)."; \
		fi \
	else \
		echo "Container $(CONTAINER_NAME) is not running. Nothing to stop for Jupyter."; \
	fi

# Clean the system from all unused Docker caches, images, containers, networks and volumes.
prune:
	docker system prune -a --volumes -f
	@echo "Docker caches, images, containers, networks and volumes cleaned."
