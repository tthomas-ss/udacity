# Libraries

This project is delivered with a jupyter notebook, installed with miniconda.  The python environment is version 3.6, and the 
notebook runs on my conda default environment that is somewhat bloated.. requirements.txt attached.  "Non-standard" libraries used:
- scipy
- sklearn
- BeautifulSoup
- folium
- matplotlib
- seaborn
- pandas
- numpy

# Motivation

This notebooks is part of a project in Udacity's Data Scientist Nanodegree.  The purpose is to get some experience with the CRISP-DM (Cross-industry standard process for data mining) cycle by:

1) Picking a dataset.

2) Pose at least three questions related to business or real-world applications of how the data could be used.

3) Create a Jupyter Notebook or Python script to prepare and analyze data

4) Communicate business insights through a github repository and a blog post.

I chose to work on a Airbnb dataset with listings in Oslo, Norway and attempted to answer  the following business questions:

* Which areas of Oslo have the most Airbnb listings?
* Which characteristics a property drives the Airbnb listing prices in Oslo?
* Is there a relationship between official housing prices and the prices people are asking for on Airbnb?
* Is it possible to use machine learning techniques to predict the price of a listing?


# Summary

I found the most listings were in 5 central boroughs, and that the number of people the property could accommodate was the most important factor for listing prices.

The relationship between official real estate prices and Airbnb listing prices was close to non-existent - i.e. people with property in the expensive parts of town do not command a higher rent on Airbnb.

Finally, my efforts to use machine learning to predict the price of listings did not give very good results.  If time permitted, I would have spent more time on feature selection/engineering and tuning the algorithms.

# Files

- ./README.md - this file.
- ./DataScience-PresentationProject.ipynb - Jupyter notebook containing all data wranling, analysis and ML code.
- https://medium.com/@thomas.schoyen/an-analysis-of-airbnb-in-oslo-norway-cdaa420c47f0 - blog post.
- ./data/* - raw data files.
- ./html/* - saved versions of folium maps.
- ./img/* - saved images for the blog post

# How to interact with this project.

* For non-techincal people - read the DataScience-PresentationProject-Blog.md
* For techincal people - review the DataScience-PresentationProject.ipynb notebook.
* To continue working - clone the repository, or start from scratch with the data in the ./data folder.

# Licensing, Authors, Acknowledgements, etc.

Thanks to Udacity for supplying some of the code used in lectures.

