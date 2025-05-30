# Hybrid Recommendation System with Route Optimization

This project implements a **hybrid recommendation system** integrated with a **route optimization module** to generate a personalized sequence of objects (e.g., monuments or exhibits) for a user. The system combines:

- Collaborative filtering
- Content-based filtering
- Heuristic route optimization

The system produces an optimized route based on merged recommendation scores.

---

## 📌 Problem Statement

The system addresses the task of generating **personalized recommendations** by leveraging user behavior data and item features. The goal is to select and order a subset of objects that are most relevant to the user, while also optimizing the sequence for efficient visiting.

---

## ⚙️ Methods and Architecture

### 1. **Collaborative Filtering**
Analyzes historical interactions between users and objects to find similar users. Items preferred by similar users are recommended.

### 2. **Content-Based Filtering**
Utilizes item attributes (e.g., architecture, type, material) to identify and recommend items similar to those previously liked by the user.

### 3. **Heuristic Route Optimization**
A heuristic algorithm (HSATS) is used to determine the best sequence for visiting recommended objects under time constraints. Recommendation scores are used as rewards in the optimization.

---

## 📁 Project Structure

```
diploma-master/
│
├── data
│
├── recommendation_systems/
│   ├── content_based.py
│   ├── collaborative_system.py
│   └── merge_recommendations.py
│
├── HSATS/
│   ├── heuristic.py
│   └── candidates_generator.py
│
├── maps_integration/
│   └── points.py
│
├── map.html
├── config.yaml
├── main.py
└── README.md
```

---

## 🚀 Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python main.py
```

This will:
- Generate content-based and collaborative recommendations
- Merge scores
- Optimize route based on recommendations
- Generate `points.json` for visualization

---

## 🗺️ Visualizing the Final Route on a Map

Because browsers block `file://` access to local JSON files, run a local server:

### Step 1: Start server

```bash
cd diploma-master
python -m http.server 8080
```

### Step 2: Open in browser

```
http://localhost:8080/map.html
```

---

## 👨‍💻 Author

This project was developed as part of a master's thesis to demonstrate how recommendation and optimization techniques can be combined to solve a real-world problem.
