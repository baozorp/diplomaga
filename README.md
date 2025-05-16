# Stochastic Optimization-Based Recommender System

This project is a recommender system based on stochastic optimization for solving a scientific problem. The system combines three algorithms: collaborative filtering, item-based similarity, and distance-based similarity, whose results are integrated into a unified solution. The unique aspect of this system lies in its incorporation of the recommendation results iteratively into the stochastic optimization process. As a result of the solution, a CSV file is generated, containing sorted objects along with their scores.

## Problem Statement

The recommender system is designed to address the scientific problem of providing personalized recommendations to users based on their past actions and preferences. Our aim is to offer highly relevant and personalized recommendations by considering the similarity between users and objects.

## Methods and Approaches

To solve this problem, we have employed a combination of three algorithms:

1. Collaborative Filtering: This method analyzes user preferences and their interaction history with objects. We utilize algorithms that identify similarities between users and recommend objects that have been positively evaluated by other users with similar interests and preferences.

2. Item-based Similarity System: In this method, we analyze the similarity between objects based on their attributes and characteristics. We identify objects with similar attributes and recommend them to users based on their positive evaluations of similar objects.

3. Distance-based Similarity System: This method determines the distance or similarity between objects based on certain metrics or features. We utilize algorithms that find objects closest to those positively evaluated by the user.

## Integration with Stochastic Optimization

A key feature of this project is the integration of the recommender system with the stochastic optimization process. This integration allows for the incorporation of the recommendation results to optimize the selection of an optimal route. By leveraging the recommendation scores obtained from the system, the stochastic optimization process can make informed decisions to identify the most favorable route.

The resulting solution provides a CSV file containing sorted objects and their scores, allowing users to make informed decisions based on the recommended optimal route.
