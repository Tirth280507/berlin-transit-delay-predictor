# 🚆 Berlin Transit Delay Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automated_Data_Collection-2088FF?logo=githubactions&logoColor=white)
![Status](https://img.shields.io/badge/status-live-brightgreen)

**Predicts how likely a Berlin (VBB) bus/train route is to be delayed** —
based on route, day of week, and hour — trained on real-time delay data I
collected myself, since no public dataset of actual (vs. scheduled) transit
times exists for free.

### 🔗 [**Try the live app →**] https://transit-delay-predictor-kgkpsvm22e9ktruf7ultc5.streamlit.app/ 

<!-- Add a screenshot here once you have one:
![App screenshot](path/to/screenshot.png)
-->

---

## 📋 Table of contents
- [Why this project](#-why-this-project)
- [Results](#-results-honest-numbers-not-just-headline-accuracy)
- [How it works](#-how-it-works)
- [Challenges I hit and fixed](#-challenges-i-hit-and-fixed-along-the-way)
- [Quick start](#-quick-start)
- [Repo structure](#-repo-structure)
- [Tech stack](#-tech-stack)
- [Future improvements](#-possible-future-improvements)

---

## 💡 Why this project
Most beginner ML projects use the same 3-4 textbook datasets (Titanic, Iris,
Heart Disease). This one uses a **real, messy, self-collected dataset** from
my own city's public transit system — including building the entire data
collection pipeline myself, not just downloading a ready-made CSV.

## 📊 Results (honest numbers, not just headline accuracy)
Trained and compared **3 model types** — Logistic Regression, Random Forest,
and Gradient Boosting — on ~36,000 real route/day/hour records. Gradient
Boosting performed best, and was deliberately tuned to prioritize **catching
real delays** over raw accuracy, since missing a delay matters more than an
occasional false alarm for a commuter.

| Metric | Score |
|---|---|
| ✅ Catches real delays (recall) | **61%** |
| ✅ Correct when it flags a delay (precision) | **60%** |
| Overall accuracy | ~78% |

> **For context:** always guessing "on time" gets ~73% accuracy but catches
> **0%** of real delays. This model is a genuine, meaningful improvement
> over doing nothing — not just a headline-accuracy illusion.

## ⚙️ How it works
1. **Static schedule data** — VBB's free GTFS feed (routes, stops, planned times)
2. **Real delay data** — collected by polling VBB's live GTFS-Realtime feed
   automatically via a scheduled GitHub Action, logging scheduled vs. actual
   arrival times continuously over several weeks (**~1,000 routes**, all
   **7 days**, **23 of 24 hours** covered)
3. **Model** — a Gradient Boosting classifier (scikit-learn), predicting
   delay risk for a given route + day + hour
4. **App** — a Streamlit dashboard where a user picks a route/day/time and
   sees the predicted delay risk live

## 🐛 Challenges I hit and fixed along the way
- **File size limit:** raw per-stop logging grew past GitHub's 100MB file
  limit → redesigned to a bounded rolling summary (route/day/hour averages)
  instead of an ever-growing log
- **Timezone bug:** GitHub Actions runs in UTC, not local time → fixed by
  explicitly using the `Europe/Berlin` timezone so day/hour patterns match
  real commute times
- **Class imbalance:** the default model was biased toward the majority "on
  time" class → caught this by checking precision/recall instead of trusting
  accuracy alone, then fixed it with class weighting, tuned by comparing
  multiple weight values

## 🚀 Quick start
```bash
git clone https://github.com/Tirth280507/transit-delay-predictor.git
cd transit-delay-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repo structure
```
scripts/            # data collection, exploration, training & tuning scripts
data/                # collected datasets, trained model files
app.py               # the Streamlit app
.github/workflows/   # scheduled data-collection automation
```

## 🛠️ Tech stack
Python · pandas · scikit-learn · Streamlit · GTFS / GTFS-Realtime · GitHub Actions

## 🔮 Possible future improvements
- Add weather data as a feature
- Predict specific delay minutes, not just risk category
- Move from hour-level to 30-minute-level time granularity

---

## 👤 Author
**Tirth Chovatiya**
