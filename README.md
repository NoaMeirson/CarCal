# CarCal

## Setup

Clone the repository:

git clone https://github.com/NoaMeirson/CarCal.git
cd CarCal

Create a virtual environment:

python3 -m venv .venv

Activate the virtual environment (Mac / Linux):

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

## Running the Project

Run the main program:

python main.py

(Replace `main.py` with the actual entry file if different.)

## Development Notes

- The `.venv` folder is **not included in the repository**.
- Each developer should create their own virtual environment locally.

If new dependencies are installed, update the requirements file:

pip freeze > requirements.txt

Then commit the updated file:

git add requirements.txt
git commit -m "update requirements"
git push

## Project Structure

CarCal/

API/  
Client/  
Engine/  
Contracts/  
models.py  
requirements.txt  
README.md  
.gitignore