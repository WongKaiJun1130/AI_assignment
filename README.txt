BOOK RECOMMENDER SYSTEM

==================================================

1. SYSTEM OVERVIEW
   ==================================================

This project is a Book Recommender System developed using Python and Streamlit.

The system provides book recommendations to users using four recommendation methods:

1. Popularity-Based Recommendation
2. Content-Based Recommendation
3. Collaborative Filtering
4. Hybrid Recommendation

The Hybrid Recommendation method combines Content-Based, Collaborative Filtering, and Popularity-Based recommendation scores to provide a more balanced recommendation result.

==================================================
2. TECHNOLOGY USED
==================

Programming Language:

* Python

Framework:

* Streamlit

Recommendation Techniques:

* Popularity-Based Recommendation
* Content-Based Filtering
* Collaborative Filtering
* Hybrid Recommendation

Content-Based Technique:

* TF-IDF

Collaborative Filtering:

* Pearson Correlation

==================================================
3. PROJECT FILES
================

Main files may include:

* book_recommender_full.py
  Main Streamlit application.

* recommender.py
  Contains the recommendation methods and recommendation logic.

* requirements.txt
  Contains the Python libraries required to run the system.

* dataset/
  Contains the dataset used by the Book Recommender System.

* model/
  Contains saved models or processed data, if applicable.

* Report/
  Contains the project report.

==================================================
4. REQUIREMENTS
===============

The following software is required:

* Python 3.x
* Visual Studio Code (recommended)
* Internet browser

Required Python libraries are listed in:

requirements.txt

==================================================
5. INSTALLATION
===============

Step 1:
Install Python 3.x on the computer.

Step 2:
Open the project folder using Visual Studio Code.

Step 3:
Open the Terminal in Visual Studio Code.

Step 4:
Install the required Python libraries by running:

pip install -r requirements.txt

==================================================
6. HOW TO RUN THE SYSTEM
========================

Step 1:
Open the project folder in Visual Studio Code.

Step 2:
Open the Terminal.

Step 3:
Run the following command:

streamlit run book_recommender_full.py

Step 4:
Streamlit will provide a local URL in the Terminal.

Example:

http://localhost:8501

Step 5:
Open the URL in a web browser to access the Book Recommender System.

==================================================
7. HOW TO USE THE SYSTEM
========================

1. Open the Book Recommender System in the web browser.

2. Select or enter a book according to the available system interface.

3. The system processes the selected book.

4. The system generates the Top 10 recommended books.

5. Users can view the recommended book titles and other available information.

==================================================
8. RECOMMENDATION METHODS
=========================

Popularity-Based Recommendation:
Recommends books based on their overall popularity and weighted ratings.

Content-Based Recommendation:
Recommends books that are similar to the selected book based on book information such as title, author, and publisher. TF-IDF is used to represent the textual information.

Collaborative Filtering:
Uses user-rating behaviour to identify books that are related through similar user preferences.

Hybrid Recommendation:
Combines multiple recommendation methods to improve recommendation performance.

The Hybrid method used in this project combines:

* 40% Content-Based score
* 45% Collaborative Filtering score
* 15% Popularity-Based score

==================================================
9. EVALUATION
=============

The recommendation methods were evaluated using:

* Precision@10
* Recall@10
* F1@10

A rating of 8 or above was treated as positive feedback.

The evaluation was performed using 100 query books.

Evaluation results:

Popularity-Based:
Precision@10 = 0.0520
Recall@10    = 0.0520
F1@10        = 0.0520

Content-Based:
Precision@10 = 0.0760
Recall@10    = 0.0780
F1@10        = 0.0767

Collaborative Filtering:
Precision@10 = 0.0900
Recall@10    = 0.0900
F1@10        = 0.0900

Hybrid:
Precision@10 = 0.0980
Recall@10    = 0.1000
F1@10        = 0.0987

Based on the evaluation results, the Hybrid Recommendation method achieved the highest F1@10 score and was selected as the best-performing method in this project.

==================================================
10. DATASET
===========

The dataset used in this project is the Book-Crossing dataset.

The dataset is included in the project ZIP file if the file size allows it.

If the dataset is stored separately due to its large file size, a dataset download link is provided in the accompanying document.

==================================================
11. IMPORTANT NOTES
===================

* Make sure the dataset is placed in the correct folder before running the system.
* Make sure all required Python libraries are installed.
* Run the Streamlit application from the project folder.
* The file names and folder locations should not be changed unless the corresponding paths in the code are also updated.
* The recommendation results depend on the dataset and preprocessing methods used in this project.

==================================================
12. PROJECT REPORT
==================

The project report is included together with the prototype.

Please refer to the report for:

* System design
* Recommendation methods
* Implementation
* Evaluation
* Discussion and interpretation
* Project results

==================================================
END OF README
=============
