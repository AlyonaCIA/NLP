# Solar Physics and Machine Learning NLP Project

This project focuses on downloading, processing, and analyzing abstracts of academic papers related to solar physics and machine learning using Natural Language Processing (NLP) techniques. The project is structured into different directories to keep data, code, and results organized.

## Project Structure

```
.
├── README.md
├── data
│   └── solar_ml_abstracts.csv           # Contains the downloaded abstracts data
├── models
├── notebooks
│   └── Astrophysics_NLP_Sentiment_Analysis.ipynb  # Jupyter Notebook for analysis
├── requirements.txt                     # List of dependencies
├── results
└── src
    ├── fetch_solar_articles.py          # Script to fetch articles from arXiv
    ├── fetch_solar_articles..py         # (Possible duplicate, remove if unnecessary)
    └── model_training.py                # Placeholder for model training script
```

## Getting Started

### Prerequisites

Before running the project, ensure you have Python installed on your system. It's recommended to use a virtual environment to manage dependencies.

### Setting Up the Environment

1. **Create and Activate a Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the Required Packages**:

   Install the necessary Python packages listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   If you haven't created a `requirements.txt` yet, you can do so by running:

   ```bash
   pip freeze > requirements.txt
   ```

3. **Install Additional Dependencies**:

   If `matplotlib` or other packages are not installed, you can add them using:

   ```bash
   pip install matplotlib
   ```

### Fetching Articles from arXiv

The script `fetch_solar_articles.py` is used to download abstracts from arXiv related to solar physics and machine learning.

1. **Run the Fetch Script**:

   ```bash
   python3 src/fetch_solar_articles.py
   ```

   This script will download up to 1000 articles and save the data in `data/solar_ml_abstracts.csv`.

### Working with the Data in Jupyter Notebook

1. **Start Jupyter Notebook**:

   Launch Jupyter Notebook to begin analyzing the downloaded data:

   ```bash
   jupyter notebook
   ```

2. **Load the Data**:

   In your notebook (e.g., `Astrophysics_NLP_Sentiment_Analysis.ipynb`), load the CSV data:

   ```python
   import pandas as pd

   # Load the data
   data = pd.read_csv('../data/solar_ml_abstracts.csv')

   # Display the first few rows of the dataset
   data.head()
   ```

### Visualizing the Data

To visualize the distribution of articles over time:

1. **Ensure `matplotlib` is Installed**:

   If `matplotlib` is not installed, add it to your environment:

   ```bash
   pip install matplotlib
   ```

2. **Plot the Data**:

   ```python
   import matplotlib.pyplot as plt

   # Convert the 'published' column to datetime format
   data['published'] = pd.to_datetime(data['published'])

   # Plot the distribution of articles over time
   plt.figure(figsize=(10, 6))
   data['published'].hist(bins=30)
   plt.title('Distribution of Articles Over Time')
   plt.xlabel('Publication Date')
   plt.ylabel('Number of Articles')
   plt.show()
   ```

### Troubleshooting

If you encounter issues with importing packages or running scripts, consider the following:

1. **Verify the Python Environment**:

   Ensure that the packages are installed in the correct Python environment. You can check which Python environment is being used by running:

   ```python
   import sys
   print(sys.executable)
   ```

2. **Restart Jupyter Kernel**:

   After installing new packages, restart the Jupyter kernel:

   ```bash
   Kernel > Restart Kernel
   ```

### Further Analysis

With the data loaded, you can proceed with various NLP tasks such as sentiment analysis, topic modeling, or word cloud generation. These analyses can be documented and expanded upon in your Jupyter Notebook.
