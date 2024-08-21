
# GAN Fashion MNIST practice

This project implements a Generative Adversarial Network (GAN) for training on the Fashion MNIST dataset. The primary goal of this project is to practice training a GAN model and experimenting with different training strategies. The code includes support for conditional GANs, different strategies for discriminator training, and tensorboard logging for visualizing training progress.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [License](#license)

## Installation

To run this project, you'll need to have Python installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/OrlovAlexandr/GAN_mnist_practice.git
cd GAN_mnist_practice
pip install -r requirements.txt
```

Ensure you have the appropriate versions of the dependencies listed in `requirements.txt`.

## Usage

You can train the model using the following command:

```bash
python main.py
```

This will start the training process. The training logs, including loss metrics and images generated during training, will be saved to the `imgs/trains` directory. You can monitor the training process using Tensorboard.

### Web Interface

A Flask-based web interface is provided to control and monitor the training process. To start the web server, run:

```bash
python app.py
```

Once the server is running, open your web browser and go to `http://127.0.0.1:5000`. The interface allows you to:
- **Start/Stop Training**: Control the training process with simple buttons.
- **Update Configuration**: Modify training parameters.
- **Monitor Progress**: View the latest generated images directly from your browser.


## Features

- **Conditional GAN Support**: Allows the model to be conditioned on specific labels, enabling the generation of digit images with specific characteristics.
- **Discriminator Training Strategies**: Includes options for using gradient penalty or weight clipping during discriminator training.
- **Tensorboard Integration**: Logs metrics, histograms, and images to Tensorboard for easy monitoring of the training process.
- **Flask Web Interface**: Manage and monitor the GAN training process from your browser.
- **Save Model and Output**: Automatically saves the trained generator model and generated images at specified intervals.

## Project Structure

- `app.py`: Flask application entry point.
- `config.py`: Configuration file for setting up model parameters, options, and other settings.
- `main.py`: Main training script logic.
- `models.py`: Contains the GAN model definitions (generator, discriminator).
- `train.py`: The main script for training the GAN model.
- `requirements.txt`: Lists the required Python packages.
- `templates/index.html`: HTML templates for the web interface.
- `assets/`: Assets folder with images for this Readme.'
- `static/`: Static assets for the web interface.
- `utils/`: Utility folder with functions for initializing weights, computing gradient penalties, etc.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
