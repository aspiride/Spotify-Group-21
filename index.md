---
title: Motivation and Project Statement
nav_include: 1
---

<a href="https://aspiride.github.io/Spotify-Group-21/EDA"> <button>EDA</button> <a>

# Spotify Recommendation Algorithm
Group 21: Alex Spiride, Dean Hathout, and Eric Zhu

## Motivation & Literature Review

Why do we need a music recommendation system? The reason is that an enormous number of songs are available from online resources, Spotify, for example, has more than 40 million songs. On the other hand, an average listener could listen to only a few hundred songs, unaware of millions of songs that may interest him or her. Moreover, the demand for music is highly personalized, and a universal recommendation may not work well. There are many key factors to consider, including listener's music preference, emotional mood, and listening intent and context in order to make an effective recommendation. It is simply impossible for a human to do it, and therefore we need to implement a music recommendation system by exploiting the power of data science.

Recommendation systems are already pervasive our lives, especially in online services such as movies, books, and shopping. Music recommendation shares many similarities with other types of recommendation system, but also has its own special features <sup>[1-4]</sup>: 
1. The length of a typical music piece is about 3 minutes in contract to 90 minutes in movies, resulting in higher tolerance for occasional poor recommendation.
2. The consumption of music pieces are usually sequential, putting forward a challenge to arrange a number of songs in a proper order.
3. Re-recommendation of a music piece is acceptable or even appreciated, which is dramatically different from recommendation for movies or books.

Music recommendation systems can be classified into collaborative filtering recommenders, content-based recommenders, and hybrid recommenders incorporating two or more types of strategies <sup>[5]</sup>. Collaborative filtering recommenders make use of the playing history of music listeners, assuming that listeners sharing interests for some music pieces would probably have similar preference for others. Content-based recommenders, on the other hand, focus on the audio features of a particular music piece, e.g., tempo, acousticness, liveness, loudness, speechiness, and make recommendations based on a listener's preference in the feature space. It is worth mentioning that deep learning is particularly relevant for music recommendation systems due to the availability of big data and the capability for modeling sophisticated nonlinear mapping.

## Problem Statement

The goal of this project is to make music recommendation for a provided playlist. We have investigated two popular databases, Million Playlist Dataset and Million Song Dataset, listed in the project instruction. We have implemented both content-based recommenders and collaborative filtering recommenders where the former has better performance for long playlists while the latter is good for short ones. We have explored various machine learning algorithms well beyond what have been taught in AC-209A, including unsupervised clustering, matrix factorization, embedding technique, and Y-shaped neural networks <sup>[6-12]</sup>. The technical details and results are presented in the python notebooks.  


### References

[1] Berenzweig, Adam, Beth Logan, Daniel P.W. Ellis and Brian Whitman. A Large-Scale Evaluation of Acoustic and Subjective Music Similarity Measures. Proceedings of the ISMIR International Conference on Music Information Retrieval (Baltimore, MD), 2003.

[2] Logan, B., A Content-Based Music Similarity Function, (Report CRL 2001/02) Compaq Computer Corporation Cambridge Research Laboratory, Technical Report Series (Jun. 2001).

[3] Shedl, M. et al, Current Challenges and Visions in Music Recommender Systems Research, https://arxiv.org/pdf/1710.03208.pdf.

[4] Shedl, M., Peter Knees, and Fabien Gouyon, New Paths in Music Recommender Systems, RecSys’17 tutorial, http://www.cp.jku.at/tutorials/mrs_recsys_2017/slides.pdf.

[5] Shuai Zhang, Lina Yao, Aixin Sun and Yi Tay, Deep Learning based Recommender System: A Survey and New Perspectives, arXiv:1707.07435v6 [cs.IR] 4 Sep 2018.

[6] Rounak Banik, Recommender Systems in Python: Beginner Tutorial, https://www.datacamp.com/community/tutorials/recommender-systems-python

[7] Nicolas Hug, Understanding matrix factorization for recommendation, http://nicolas-hug.com/blog/matrix_facto_1

[8] TensorFlow Team, Introducing TensorFlow Feature Columns, https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html

[9] Nipun Batra, Recommender Systems in Keras, https://nipunbatra.github.io/blog/2017/recommend-keras.html

[10] Yuhang Lin, Fayha Almutairy, Nisha Chaube, Music Recommendation System: Music + ML, https://www.uvm.edu/~ylin19/files/Music_Recommer_System_Presentation.pdf

[11] Matt Murray, Building a Music Recommender with Deep Learning, http://mattmurray.net/building-a-music-recommender-with-deep-learning/

[12] Eric Le, How to build a simple song recommender system, https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85
