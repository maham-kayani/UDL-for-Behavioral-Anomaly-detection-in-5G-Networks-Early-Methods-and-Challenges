# UDL-for-Behavioral-and-Quality-Anomaly-Detection-in-5G-Networks-Early-Methods-and-Challenges
An unsupervised machine learning pipeline designed to detect unusual smartphone usage and behavioral patterns (such as extreme battery drain and heavy data usage) using a Deep FeedForward Autoe

## 📊 Dataset & What We Started With
* **Dataset:** [Smartphone Usage and Behavioral Dataset on Kaggle](https://www.kaggle.com/datasets/bhadramohit/smartphone-usage-and-behavioral-dataset?resource=download) (by Bhadra Mohit).
* **What it covers:** Daily mobile usage of 1,000 users, tracking screen time, app usage, battery drain, apps installed, and data usage across different demographics.
* **Our Features:** We focused on App Usage Time, Screen-on Time, Battery Drain, Apps Installed, and Data Usage.

---

## ⚙️ 1. How We Preprocessed the Data
* **Feature Selection & Domain Knowledge:** Before doing any clustering, we relied heavily on field knowledge to pick the right features. Additionally, state-of-the-art research highlights good techniques to use before clustering, and Shannon entropy is one of the best methods for this.
* **Shannon Entropy Check:** Before clustering, we ran Shannon entropy checks. This is a great state-of-the-art method to see how rich and informative our features are before doing any heavy lifting.
* **Feature Scaling:** We put all our values through `StandardScaler` so everything scaled nicely with a mean of 0 and standard deviation of 1.

---

## 🔍 2. Unsupervised Clustering
* **Finding the Best Clusters:** We used the **Elbow Method** and **Silhouette Method** to figure out the ideal number of clusters.
* **What We Found (4 Clusters):** After looking at the values manually, we split our dataset into two main groups:
  * **Clusters 0 & 2 (Normal Ground):** The majority of users who show normal, everyday battery and data habits.
  * **Clusters 1 & 3 (Abnormal Behavior):** A smaller group of users showing much higher, unusual activity.

---

## 🧠 3. Our FeedForward Deep Autoencoder Model
* **Why this model?** We chose a FeedForward Autoencoder because it has low computational cost and runs super fast.
* **Encoder (ReLU):** We compressed our 5 features into a **3-dimensional bottleneck** (mapping things like how data and battery usage relate to app counts). We used **ReLU** to catch those tricky non-linear patterns.
* **Decoder (Linear Activation):** We recommend using a **Linear activation** instead of **Sigmoid**. Why? Because we scaled our data using `StandardScaler`, which creates negative numbers. Since Sigmoid only works for values between 0 and 1, it would clip our negative values. Linear activation lets the model output any number without cutting off our negative data.

---

## 📈 4. Reconstruction Error & Results
* **Error Calculation (MSE):** We measured the Mean Squared Error between the original input and the reconstructed output. Since our model was trained strictly on normal users, any weird behavior caused a massive error spike.
* **Setting the Threshold:** We set our threshold at the **60th percentile**. In the real world, you can tweak this using field knowledge depending on how strict you want your alerts to be.
* **Our Metrics:**
  * **Accuracy:** 98%
  * **Precision:** 90%
  * **Recall:** 97%
  * **F1-Score:** 93%

*(Check the attached diagram for our complete end-to-end workflow!)*

---

## 💡 How It All Works Together
1. We pick our features using Shanon Entropy.
2. We use clustering to separate normal users from the outliers by optimal k-clustering followed by two techniqued Entropy and Silhoutte.
3. We feed normal behavior into our autoencoder to learn a clean baseline.
4. We check the reconstruction error against our 60th-percentile threshold to catch anomalies with high precision and recall.

---

## ⚠️ Limitations We Noticed
* **Threshold Sensitivity:** While the 60th percentile worked great for us here, real-world user habits change over time, meaning thresholds would need to adapt dynamically.
