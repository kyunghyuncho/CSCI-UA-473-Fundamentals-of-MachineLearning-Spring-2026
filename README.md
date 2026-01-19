# CSCI-UA-473-Fundamentals-of-MachineLearning-Spring-2026

An interactive,hands-on machine learning course using real-world datasets. Learn ML concepts through visualization and experimentation with the TMDB Movies and NYC Airbnb datasets.

## 📚 Course Overview

This repository provides an **interactive learning environment** for machine learning fundamentals. Using Streamlit, students can:
- Visualize how ML algorithms work in real-time
- Experiment with model architectures and hyperparameters
- Understand abstract concepts through concrete examples
- Build intuition through hands-on exploration

**Target Audience**: Students with basic Python knowledge who want to learn machine learning through practice rather than just theory.

**Prerequisites**:
- Python programming (functions, classes, basic libraries)
- Basic linear algebra (vectors, matrices)
- Elementary statistics (mean, variance, distributions)
- **No prior ML experience required!**

---

## 🎯 Learning Objectives

By completing this course, you will understand:

1. **Embeddings**: How to convert text and images into numerical vectors
2. **Similarity Search**: How to find similar items using vector representations
3. **Data Management**: Best practices for train/validation/test splits
4. **Dimensionality Reduction**: PCA and MDS for visualization and compression
5. **Classification**: Predicting categories with neural networks
6. **Regression**: Predicting continuous values and quantifying uncertainty

---

## 🗂️ Course Structure

### Lesson 1: From Text to Numbers (Embeddings)
**Learning Goals**:
- Understand what embeddings are and why they're fundamental to modern ML
- See how transformer models (Nomic AI) convert text to 768-dimensional vectors
- Compare text embeddings vs image embeddings for the same item
- Visualize high-dimensional vectors

**Key Concepts**: Vector spaces, semantic similarity, transformer models, dimensionality

### Lesson 2: Vector Space & Retrieval
**Learning Goals**:
- Learn similarity metrics (cosine similarity, Euclidean distance, dot product)
- Build a semantic search engine using embeddings
- Compare text-based vs image-based retrieval
- Understand nearest neighbor search

**Key Concepts**: Cosine similarity, k-NN search, query vs document embeddings, retrieval systems

### Lesson 3: Data Splits (Train/Val/Test)
**Learning Goals**:
- Understand why we split data and what each split is for
- Learn about data leakage and how to prevent it
- Analyze distribution differences across splits
- Create reproducible random splits

**Key Concepts**: Generalization, overfitting, data leakage, holdout validation, stratification

### Lesson 4: PCA (Principal Component Analysis)
**Learning Goals**:
- Understand dimensionality reduction and why it's useful
- Learn PCA through deep autoencoders (non-linear PCA)
- Train an autoencoder with configurable architecture
- Visualize high-dimensional data in 2D/3D
- Project custom inputs into the learned space

**Key Concepts**: Autoencoders, reconstruction loss, bottleneck representations, variance preservation, denoising

### Lesson 5: MDS (Multidimensional Scaling)
**Learning Goals**:
- Learn an alternative to PCA: preserving distances instead of variance
- Understand different distance functions (Euclidean, cosine, correlation)
- Compare MDS vs PCA visualizations
- Apply MDS to both text and image embeddings

**Key Concepts**: Distance preservation, stress minimization, metric MDS, kernel methods, Nyström approximation

### Lesson 6: Genre Classification (Multi-Label)
**Learning Goals**:
- Understand classification vs regression
- Learn multi-label classification (one item, multiple categories)
- Train an MLP classifier with weighted loss
- Analyze per-class performance metrics
- Test predictions on new inputs

**Key Concepts**: Multi-label classification, binary cross-entropy, sigmoid activation, class imbalance, precision/recall/F1

### Lesson 7: Regression (Predicting Values)
**Learning Goals**:
- Learn three types of regression: point estimation, probabilistic (MDN), quantile (SQR)
- Understand when to use each approach
- Handle skewed distributions with log transforms
- Quantify uncertainty in predictions
- Visualize predicted distributions

**Key Concepts**: Mean Squared Error, Mixture Density Networks, quantile regression, pinball loss, uncertainty quantification, heteroscedasticity

---

### Lesson 8: Cross-Modal Retrieval
**Learning Goals**:
- Understand how to map different modalities (text, images) to a shared vector space
- Train a Dual Encoder model using Contrastive Learning
- Build a search engine that finds images from text descriptions (and vice-versa)
- Perform zero-shot classification using cross-modal embeddings

**Key Concepts**: Dual Encoders, Contrastive Loss (InfoNCE), CLIP-style training, multimodal embeddings, zero-shot learning

### Lesson 9: Reinforcement Learning
**Learning Goals**:
- Understand Reinforcement Learning (RL) basics
- implemented the REINFORCE algorithm (Policy Gradient)
- Train an agent to land a lunar lander safely
- Understand the importance of a Baseline (Value Function)

**Key Concepts**: Policy Gradients, REINFORCE, Value Functions, Variance Reduction, OpenAI Gymnasium

### Lab 0: It's Time to Try Vibe Coding
**Learning Goals**:
- Install antigravty IDE, uv environment.
- Clone the repository and run the streamlit app successfully
- Use antigravty IDE to vibe code comments. Use your brain to understand the code and debug.
- Check access to Campuswire.

**Key Concepts**: Environment Setup, UV, Antigravty IDE, Git, Vibe Coding

---

## 🖥️ Technical Requirements

### Software
- **Python**: 3.10 or higher (3.11 recommended)
- **Operating System**: macOS or Linux (Windows is not supported officially)
- **RAM**: 16GB minimum (32GB recommended for faster training)

### Hardware Recommendations
- **GPU**: Optional but recommended (CUDA-compatible NVIDIA GPU)
  - With GPU: Model training takes seconds to minutes
  - CPU only: Training can take 5-10x longer (still manageable)
- **Internet**: Required for initial dataset download

---

## 🚀 Installation Guide

### Step 0: Clone the Repository

We use Git for version control. Git is a distributed version control system that allows you to track changes in your code, collaborate with others, and revert to previous versions if needed.

**Why use Version Control?**
- **History**: Track every change made to the codebase.
- **Collaboration**: Work with teammates without overwriting each other's code.
- **Safety**: Revert breaking changes and keep a backup of your work.

**Basic Commands**:
- `git clone <url>`: Download a repository to your computer.
- `git status`: See which files have changed.
- `git add .`: Stage all changed files for commit.
- `git commit -m "message"`: Save staged changes with a message.
- `git push`: Upload commits to GitHub.
- `git pull`: Download the latest changes from GitHub.

Now let's clone the course folder to laptop!

```bash
git clone https://github.com/kyunghyuncho/CSCI-UA-473-Fundamentals-of-MachineLearning-Spring-2026.git
cd CSCI-UA-473-Fundamentals-of-MachineLearning-Spring-2026
```

(Optional) Here is an interactive [course](https://learngitbranching.js.org/) if you want to learn more about Git.

### Step 1: Download IDE


**Background: What is an IDE?**
An **IDE** (Integrated Development Environment) unifies coding, debugging, and testing tools to streamline development.

**AI-Native IDEs**
The standard has been **VS Code** with **GitHub Copilot**, but tools like **Cursor**, **Claude Code**, and **Antigravity** are evolving the concept. Most of these are **forks of VS Code**, meaning they support your favorite extensions but bake AI into the core for a smoother "**Vibe Coding**" experience compared to using AI as just a plugin.

The doc will be focused on antigravity, but you can use either antigravity or VSCode (with copilot).

#### Antigravity Guide

Campuswire [post](https://campuswire.com/c/GFC1A6E10/feed/3) by Kyunghyun:


> if you plan to use Antigravity instead of VS Code, please get Google Gemini: Free Pro Plan for Students. this will give you a much higher rate limit on Gemini models when using them via Antigravity:

> https://gemini.google/students/

> once you get Gemini Pro Plan, you can download Antigravity and start vibe-coding!

> https://antigravity.google/

#### Copilot Guide

Campuswire [post](https://campuswire.com/c/GFC1A6E10/feed/2) by Kyunghyun:

> 1. make sure you get an education pack for Github before the semester begins by going to

> https://education.github.com/pack

> 2. once you get the education pack, make sure you enable Github Copilot Pro by

> Free access to Copilot Pro while you're a student. To enable Copilot Pro, go to your account settings and under Code, planning, and automation, select Copilot to sign up for free.

> 3. then, you can download VS Code and start using Copilot as your coding assistant:

> https://code.visualstudio.com/

#### Getting started with AI-powered IDE

1. Open the IDE and select "Open Folder" to open the `CSCI-UA-473-Fundamentals-of-MachineLearning-Spring-2026` folder (what you just cloned in step 0).
2. Open the terminal (shortcut: Ctrl + Shift + ` or move your mouse to the top of the screen and click **"Terminal" -> "New Terminal"**).
3. Open the file `scripts/lab0/setup_env.sh`. Try to understand what it does. Do **NOT** run the script until you roughly understand what it is doing.

Some important notes for beginners:
1. You might see green and red text that are semitransparent. Those are suggestions from the AI assistant. You can ignore them for now. Reject them using the `esc` key. If you want to accept (try not to do it at this stage), use the `tab` key.
2. If you ever accidentally press accept or delete, don't panic. You can always revert the changes by undoing using `Ctrl+Z` or running `git restore <filename>` (for any tracked files).

### Step 2: Create a Virtual Environment

Run the following script to set up the environment. Read the script to understand what it's doing while you wait for the installation to complete.

```bash

source scripts/lab0/setup_env.sh
```

> **Why?** Think of a virtual environment like a **separate kitchen** for each meal you cook. In one kitchen (project), you might need specific ingredients (libraries) like `pytorch 2.0`, while another kitchen needs `pytorch 1.0`. By keeping them separate, you avoid mixing ingredients that don't taste good together!
>
> In production settings, people often use stronger isolation tools like **Docker** or **enroot** (containers), which ship the entire kitchen (OS + libraries) along with the meal.


If you ever messed up the environment, delete and restart by running
```bash
rm -rf .venv
```

In the future, you can activate the environment with
```bash
source scripts/lab0/activate_env.sh
```

### Step 3: Install Dependencies

uv is a package manager that will install all dependencies listed in `pyproject.toml`.

```bash
uv sync
```

**Dependencies include**:
- `streamlit` - Interactive web interface
- `torch` + `pytorch-lightning` - Deep learning framework
- `transformers` - For loading embedding models
- `sentence-transformers` - Text embeddings (Nomic AI)
- `pandas`, `numpy` - Data manipulation
- `altair`, `plotly` - Visualizations
- `scikit-learn` - ML utilities
- `Pillow` - Image processing
- `pre-commit` - Automation for code quality checks

### What is a Pre-commit Hook?
A pre-commit hook is a script that runs automatically before you commit changes to git. In this course, we use it to enforce code style and catch errors early. It will run:
- **Black**: To format your code automatically.
- **Flake8**: To check for syntax errors and style violations.
- **Isort**: To sort your imports alphabetically.

If any check fails, the commit is blocked until you fix the issues (or they are fixed automatically by the tool).

Verify that the installation was successful by running:
```bash
python scripts/lab0/verify_installation.py
```

---

## 📊 Dataset Preparation

This step will be updated once you're ready to implement the algorithms learned in lecture!

---

## ▶️ Running the Application

Start the Streamlit dashboard:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

**Navigation**:
- Use the **sidebar** to select lessons (Lesson 1-7)
- Use the **Dataset Selection** dropdown to switch between TMDB and Airbnb
- Each lesson has multiple tabs for different activities

---

## 📁 Project Structure

```
new-course-machine-learning/
├── app.py                      # Main entry point (Streamlit homepage)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── pages/                      # Streamlit lesson pages (auto-discovered)
│   ├── 1_Lesson_1_Embeddings.py
│   ├── 2_Lesson_2_Retrieval.py
│   ├── 3_Lesson_3_Data_Splits.py
│   ├── 4_Lesson_4_PCA.py
│   ├── 5_Lesson_5_MDS.py
│   ├── 6_Lesson_6_Genre_Classification.py
│   ├── 7_Lesson_7_Regression.py
│   ├── 8_Lesson_8_Cross_Modal_Retrieval.py
│   ├── 9_Lesson_9_Reinforcement_Learning.py
│   └── 10_Lab_0_Trying_Vibe_Coding.py
│
├── utils/                      # Shared utility modules
│   ├── data_loader.py         # Loading/saving datasets
│   ├── dataset_config.py      # Dataset configuration
│   ├── embedding.py           # Text embedding (Nomic AI)
│   ├── image_embedding.py     # Image embedding (DINOv2)
│   ├── image_utils.py         # Image downloading/processing
│   ├── genre_utils.py         # Genre encoding/decoding
│   ├── metrics.py             # Evaluation metrics
│   └── ui.py                  # Streamlit UI components
│
├── models/                     # ML model implementations
│   ├── common.py              # Shared components (ResidualBlock)
│   ├── autoencoder.py         # Deep autoencoder for PCA
│   ├── mds_computation.py     # Multidimensional Scaling
│   ├── genre_classifier.py    # Multi-label MLP classifier
│   ├── genre_training.py      # Genre classifier training
│   ├── regression.py          # Regression models (MLP, MDN, SQR)
│   ├── regression_training.py # Regression training
│   └── training.py            # Shared training utilities
│
├── scripts/                    # Development/diagnostic scripts
│   ├── README.md              # Scripts documentation
│   ├── verify_retrieval.py    # Test retrieval functionality
│   ├── verify_regression.py   # Test regression models
│   ├── inspect_genres.py      # Inspect genre data
│   ├── deduplicate_data.py    # Remove duplicate movies
│   └── debug_airbnb.py        # Airbnb dataset diagnostics
│
├── process_data.py             # TMDB dataset processing
├── process_airbnb.py           # Airbnb dataset processing
├── process_images.py           # Add movie posters to TMDB
│
├── data/                       # Data directory (created on first run)
│   ├── raw/                   # Raw CSV files
│   ├── processed/             # Processed parquet files with embeddings
│   ├── posters/               # Movie poster images (TMDB)
│   └── airbnb_images/         # Listing photos (Airbnb)
│
└── checkpoints/                # Saved model checkpoints (auto-created)
```


<!-- ---

## 🔧 Development Guide

### Adding a New Lesson

1. Create `pages/N_Lesson_N_Title.py` (N = lesson number)
2. Follow the existing structure:
   - Import `display_dataset_selector()` and `display_footer()` from `utils.ui`
   - Use `st.set_page_config()` for page settings
   - Load data with cache invalidation pattern
   - Add educational markdown sections
3. The lesson will auto-appear in the sidebar

### Adding a New Dataset

1. Add configuration to `utils/dataset_config.py`:
   ```python
   DATASET_CONFIGS = {
       "new_dataset": {
           "name": "Display Name",
           "path": "data/processed/new_dataset.parquet",
           "title_col": "title_column_name",
           "text_col": "description_column_name",
           # ... other columns
       }
   }
   ```

2. Create processing script `process_new_dataset.py` following `process_airbnb.py` pattern

3. Update lessons to handle new dataset (most are already dataset-agnostic) -->

### Code Style

- **Docstrings**: Google-style for all functions/classes
- **Comments**: Explain "why" not just "what"
- **Educational focus**: Code should teach, not just work
- **Type hints**: Use for clarity (not enforced)
- **Pre-commit Hooks**: We automatically run `black` (formatting), `isort` (imports), `flake8` (linting), and `autoflake` on every commit to ensure code quality.

---

## ❓ Common Issues & Troubleshooting

<!-- ### Issue: "Data not found" error when starting a lesson

**Solution**: You need to process the dataset first:
```bash
python process_data.py          # For TMDB
# OR
python process_airbnb.py        # For Airbnb
```

### Issue: Slow embedding generation

**Solutions**:
- **Use GPU**: Install PyTorch with CUDA support
- **Reduce dataset size**: Use `--sample` flag for Airbnb

### Issue: Out of memory during training

**Solutions**:
- Reduce batch size in the UI
- Use fewer training epochs
- Close other applications
- Use a smaller hidden layer size

### Issue: Model training seems stuck

**Solution**: The progress bar sometimes lags. Wait a few seconds. If truly stuck:
- Refresh the page
- Reduce max epochs
- Try CPU instead of GPU (or vice versa)

### Issue: Images not displaying

**Solutions**:
- Run `python process_images.py` to download posters (TMDB)
- For Airbnb, don't use `--skip-images` flag
- Check `data/posters/` or `data/airbnb_images/` directories exist -->

### Issue: ImportError or ModuleNotFoundError

**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Streamlit says "Please wait" forever

**Solution**:
- Check terminal for error messages
- Try restarting: `Ctrl+C` then `streamlit run app.py` again
- Clear cache: Press `C` in the running app

---

## 📖 References & Citations

### Models Used

- **Nomic AI Embed Text v1.5**: Text embeddings
  [https://huggingface.co/nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

- **DINOv2**: Image embeddings
  [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)

### Datasets

- **TMDB 5000 Movies**:
  Source: [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
  Hosted: [Hugging Face - AiresPucrs/tmdb-5000-movies](https://huggingface.co/datasets/AiresPucrs/tmdb-5000-movies)

- **NYC Airbnb Open Data**:
  Source: [Inside Airbnb](http://insideairbnb.com/)
  Direct: [NYC Listings (Oct 2025)](https://data.insideairbnb.com/united-states/ny/new-york-city/2025-10-01/data/listings.csv.gz)

### Frameworks & Libraries

- **PyTorch & PyTorch Lightning**: Deep learning framework
- **Streamlit**: Interactive web applications for ML
- **Sentence Transformers**: Easy access to embedding models
- **Scikit-learn**: Classical ML utilities
- **Altair & Plotly**: Declarative visualizations

### Research Papers (for deeper learning)

- Bishop, C. M. (1994). "Mixture Density Networks" (MDN)
- Koenker, R. & Bassett, G. (1978). "Regression Quantiles" (Quantile Regression)
- Hinton, G. E. & Salakhutdinov, R. R. (2006). "Reducing the Dimensionality of Data with Neural Networks" (Autoencoders)
- Borg, I. & Groenen, P. J. (2005). "Modern Multidimensional Scaling" (MDS)

---

## 🤝 Contributing

Contributions are welcome! Potential areas:

- Add new lessons (e.g., clustering, time series)
- Support additional datasets
- Improve visualizations
- Add more model architectures
- Enhance documentation
- Fix bugs or improve performance

Please ensure code maintains the **educational** focus - it should be clear and well-commented, even if slightly verbose.

---

## 📄 License

This project is for educational purposes.

Dataset licenses:
- TMDB: Check TMDB terms of use
- Airbnb: Creative Commons CC0 1.0 Universal (Public Domain Dedication)

Model licenses:
- Nomic AI: MIT License
- DINOv2: Apache 2.0 License

---

## 🙏 Acknowledgments

- **TMDB** for providing the movie dataset
- **Inside Airbnb** for the NYC listings data
- **Nomic AI** and **Meta AI** for the embedding models
- **Streamlit** team for making interactive ML accessible
- All open-source contributors to PyTorch, transformers, and scientific Python stack

---

**Happy Learning! 🎓🚀**

For questions or issues, please open a GitHub issue or refer to the troubleshooting section above.
