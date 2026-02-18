# API Service

src/ – All API source code

## The following not implemented yet, they need to be under src:
main.py – API entry point and application setup

routes/ – Defines API endpoints and request/response flow

services/ – API internal business logic and data preparation

clients/ – Communication layer with external services
## ------------------------------------------------

APIConfig.py – API configuration parameters

# Engine Service

src/ – All Engine source code

## The following not implemented yet, they need to be under src:
main.py – Engine entry point and service initialization

inference/ – Model execution logic

fusion/ – Logic for combining multiple model outputs
## ------------------------------------------------

models/ – Model loading and abstraction layer

EngineConfig.py – Engine configuration parameters

# Client

src/ – Web client source code

ClientConfig.js – Client-side configuration

# Contracts

The Contracts module defines the shared data structures used by the API, Engine, and Client.
It ensures that all services use the same data formats and expectations.

pycache/
Python auto-generated cache files. Not part of the project logic.

examples/
Simple examples that show how to use the contracts and data structures.

schemas/
Formal definitions of data structures (input and output formats, validation rules).

init.py
Marks the Contracts folder as a Python package.

generate_schemas.py
Script for automatically generating schema files from the models or definitions.

models.py
Shared logical data models used inside the code (not machine learning models).
## Harel- these are NOT DL models please do not erase i will not aprove

README.md
Documentation explaining how the Contracts module is structured and used.