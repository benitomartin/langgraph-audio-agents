# langgraph-audio-agents

![Diagram](static/image.png)

<div align="center">

<!-- Project Status -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/badge/python-3.12.8-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

<!-- Providers -->


</div>

## Table of Contents

- [langgraph-audio-agents](#-project_name-)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Module 1](#module-1)
    - [Module 2](#module-2)
    - [Testing](#testing)
    - [Quality Checks](#quality-checks)
  - [License](#license)

## Overview

langgraph-audio-agents is a Python project designed to [briefly describe purpose].  
It provides [key features] and is built with maintainability and scalability in mind.

## Project Structure

```text
├── .github
├── src
├── tests
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── uv.lock
```

## Prerequisites

- Python 3.12
- XXX

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:benitomartin/.git
   cd 
   ```

2. Create a virtual environment:

   ```bash
   uv venv
   ```

3. Activate the virtual environment:

     ```bash
     source .venv/bin/activate
     ```

4. Install the required packages:

   ```bash
   uv sync --all-groups --all-extra
   ```

5. Create a `.env` file in the root directory:

   ```bash
    cp env.example .env
   ```

## Usage

### Configuration

Configure API keys, model names, and other settings by editing:

src/configs/settings.py
src/configs/config.yaml

### Module 1

(Add description or usage example)

### Module 2

(Add description or usage example)

### Testing

Run all tests:

```bash
make tests
```

### Quality Checks

Run all quality checks (lint, format, type check, clean):

```bash
make all
```

Individual Commands:

- Display all available commands:

    ```bash
    make help
    ```

- Check code formatting and linting:

  ```bash
  make all-check
  ```

- Fix code formatting and linting:

  ```bash
  make all-fix
  ```

- Clean cache and build files:

    ```bash
    make clean
    ```

## License

This project is licensed under the  - see the [LICENSE](LICENSE) file for details.