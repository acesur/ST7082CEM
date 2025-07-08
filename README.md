# 🌱 Big Data Analytics for Smart Agriculture using PySpark and PowerBI

## 📖 Overview

This project demonstrates the application of big data analytics techniques to smart agriculture through comprehensive analysis of plant physiological data. Using Apache Spark for distributed processing and PowerBI for visualization, the system analyzes IoT-based agricultural sensor data to understand plant growth patterns, predict plant health, and identify optimal growing conditions.

### 🎯 Key Objectives

- **Demonstrate scalable big data processing** for agricultural sensor data using PySpark
- **Apply machine learning techniques** including classification, regression, and clustering
- **Extract actionable insights** for precision agriculture applications
- **Create interactive visualizations** for agricultural decision-making support

## ✨ Features

### 🔍 Advanced Analytics
- **Multi-class Classification**: Plant health status prediction with 100% accuracy
- **Regression Analysis**: Chlorophyll content prediction (R² = 0.997)
- **Clustering Analysis**: Natural plant grouping identification (Silhouette = 0.5401)
- **Correlation Analysis**: Physiological parameter relationship mapping

### 📊 Comprehensive Visualizations
- Interactive correlation heatmaps
- K-means clustering scatter plots
- Statistical distribution analysis
- Machine learning performance comparisons
- Time-series trend analysis

### ⚡ Big Data Processing
- Distributed processing with Apache Spark
- Optimized data transformations and feature engineering
- Scalable machine learning pipelines
- Real-time analytics capabilities

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Apache Spark** | Distributed data processing | 3.5.1 |
| **PySpark** | Python API for Spark | 3.5.1 |
| **Python** | Core programming language | 3.8+ |
| **PowerBI** | Data visualization | Latest |
| **scikit-learn** | Machine learning algorithms | - |
| **pandas** | Data manipulation | - |
| **matplotlib/seaborn** | Statistical plotting | - |
| **Google Colab** | Development environment | - |

## 📊 Dataset Information

- **Source**: Kaggle - Advanced IoT Agriculture 2024 Dataset
- **Size**: 6.58 MB
- **Records**: 30,000+ observations
- **Features**: 14 physiological parameters
- **Format**: CSV

### 🌿 Physiological Parameters

1. **ACHP** - Average Chlorophyll Content
2. **PHR** - Plant Height Rate
3. **AWWGV** - Average Wet Weight of Growth Vegetative
4. **ALAP** - Average Leaf Area
5. **ANPL** - Average Number of Plant Leaves
6. **ARD** - Average Root Diameter
7. **ADWR** - Average Dry Weight of Root
8. **PDMVG** - Percentage of Dry Matter for Vegetative Growth
9. **ARL** - Average Root Length
10. **AWWR** - Average Wet Weight of Root
11. **ADWV** - Average Dry Weight of Vegetative Plants
12. **PDMRG** - Percentage of Dry Matter for Root Growth

## 🚀 Getting Started

### Prerequisites

```bash
# Install Java (required for PySpark)
sudo apt-get install openjdk-8-jdk

# Install Python dependencies
pip install pyspark pandas numpy matplotlib seaborn scikit-learn
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-agriculture-analytics.git
cd smart-agriculture-analytics
```

2. **Set up Spark environment**
```bash
# Download and extract Apache Spark 3.5.1
wget https://archive.apache.org/dist/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3.tgz
tar -xzf spark-3.5.1-bin-hadoop3.tgz
```

3. **Configure environment variables**
```bash
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
export SPARK_HOME="/content/spark-3.5.1-bin-hadoop3"
```

4. **Install Python packages**
```bash
pip install findspark pyspark kaggle
```

### 📥 Data Setup

1. **Download dataset from Kaggle**
```bash
kaggle datasets download -d wisam1985/advanced-iot-agriculture-2024
```

2. **Extract and place in project directory**
```bash
unzip advanced-iot-agriculture-2024.zip
mv Advanced_IoT_Dataset.csv data/
```

## 💻 Usage

### Basic Analysis

```python
# Initialize Spark session
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SmartAgriculture") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Load and analyze data
df = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv("data/Advanced_IoT_Dataset.csv")

# Basic statistics
df.describe().show()
```

### Machine Learning Pipeline

```python
# Feature engineering and ML pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

# Prepare features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Train model
rf = RandomForestClassifier(featuresCol="scaled_features", 
                           labelCol="label", 
                           numTrees=100)
```

## 📈 Key Results

### 🎯 Model Performance

| Model | Task | Accuracy/R² | RMSE |
|-------|------|-------------|------|
| **Random Forest** | Classification | 100.0% | - |
| **Decision Tree** | Classification | 100.0% | - |
| **Random Forest** | Regression | 0.997 | 0.244 |
| **Linear Regression** | Regression | 0.620 | 2.717 |

### 🔍 Key Insights

- **Resource Allocation Trade-offs**: Negative correlation between chlorophyll content and structural growth
- **High-Performance Plant Identification**: Class TB plants consistently show superior performance
- **Growth Stage Optimization**: Mature plants demonstrate optimal biomass ratios
- **Phenotype Classification**: 6 distinct plant clusters identified for targeted interventions

## 📁 Project Structure

```
smart-agriculture-analytics/
│
├── data/                          # Dataset files
│   └── Advanced_IoT_Dataset.csv
│
├── notebooks/                     # Jupyter/Colab notebooks
│   └── Big_Data_Analytics.ipynb
│
├── src/                          # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── machine_learning.py
│   └── visualization.py
│
├── visualizations/               # Generated plots and charts
│   ├── correlation_matrix.png
│   ├── clustering_analysis.png
│   ├── statistical_distribution.png
│   └── regression_performance.png
│
├── powerbi/                      # PowerBI files
│   ├── dashboard.pbix
│   └── export_data.csv
│
├── docs/                         # Documentation
│   └── project_report.pdf
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## 🎨 Visualizations

The project generates several key visualizations:

1. **Correlation Matrix**: Physiological parameter relationships
2. **Clustering Analysis**: K-means plant groupings
3. **Statistical Distributions**: Performance by plant characteristics  
4. **Model Performance**: Regression analysis results
5. **PowerBI Dashboard**: Interactive analytics interface

## 🌐 PowerBI Dashboard Features

- **Real-time Monitoring**: 50K plants tracking
- **Performance Metrics**: 39.95% high performance rate
- **Cluster Analysis**: Machine learning insights
- **Temporal Trends**: Chlorophyll changes over time
- **AI-Powered Insights**: Performance driver identification

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### 🔧 Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for changes

## 📚 References

1. Wolfert, S., et al. (2017). Big data in smart farming–a review. *Agricultural Systems*, 153, 69-80.
2. Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture*, 147, 70-90.
3. Liakos, K. G., et al. (2018). Machine learning in agriculture: A review. *Sensors*, 18(8), 2674.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Suresh Chaudhary**
- University: Coventry University
- Module: ST7082CEM - Big Data Management and Data Visualisation
- Email: [suresh.chaudhary1@yahoo.com]
- LinkedIn: [(https://www.linkedin.com/in/sur3sh/)]

## 🙏 Acknowledgments

- **Coventry University** for providing the academic framework
- **Mr. Siddhartha Neupane** for module guidance
- **Kaggle Community** for the dataset
- **Apache Spark Team** for the excellent big data framework
- **PowerBI Team** for visualization capabilities

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@misc{chaudhary2025smartagriculture,
  title={Big Data Analytics for Smart Agriculture using PySpark and PowerBI},
  author={Chaudhary, Suresh},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/smart-agriculture-analytics}}
}
```

---

⭐ **Star this repository if you found it helpful!**

🐛 **Found a bug?** [Open an issue](https://github.com/yourusername/smart-agriculture-analytics/issues)

💡 **Have suggestions?** [Start a discussion](https://github.com/yourusername/smart-agriculture-analytics/discussions)
