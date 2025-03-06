from setuptools import setup, find_packages


setup(
    name="obsOp",
    version="1.0",
    packages=find_packages(),
    entry_points = {
              'console_scripts': [
                  'prediction_data = Training_data.Training_data_static:main',
                  'GNN = Train_model_and_make_predictions.Predictions:main',
              ],
      },
)
