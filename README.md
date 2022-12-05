## Triton Inference Server client
This is a small FastAPI server for interacting with Triton Inference Server, using the Triton Client library.

## Installation
- Clone the repository and move to the cloned repo directory.
- Run the following command.
```bash
pip install -r requirements.txt
```
- Start uvicorn
```bash
uvicorn server:app --reload
```
## Directory structure
```
controller-temp
 ┣ dog-pics
 ┃ ┣ ...
 ┣ inferencecore
 ┃ ┣ __init__.py
 ┃ ┣ clientcore.py
 ┃ ┣ clientutils.py
 ┃ ┣ imgutils.py
 ┃ ┗ utils.py
 ┣ README.md
 ┣ __init__.py
 ┣ requirements.txt
 ┣ server.py
 ┗ vars.py
```

## Project structure
![Project Structure](./docs/project_structure.png)