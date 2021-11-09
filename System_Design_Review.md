# System Design Interview Review 

**Han Sun**



## General System Design

### How would you measure how much users liked videos?

TBF.



### How to store most popular tags in the past 24 hours

**Abstract question: top-k items over a recent time window**

1. Maintain a hashmap in memory to store all tags with their counts. Any new event comes, update the count of the tag;
2. Maintain two heaps, one maximum and one minimum. After the tag count hashmap update, check the updated tag count with the two heap to see if it's a top-K tag by comparing it with the minimum heap. If yes, add it to both heaps and update the datestamp;
3. The top-k tags are always from the maximum heap;
4. Add a datestamp value to the items stored in heap, and update the heap every minute to pop out-of-date tags off.

**Algorithm version: top-k frequent elements**

1. Use heap O(Nlog(k))
2. Use bucket sort O(N)



### Design autocomplete search system

Constraints: only English, no spell check, no personalized, no localized

Request flow:

User request -> load balancer -> Nodes (N1, N2, N3, N4) -> Tries (T1, T2, T3)

distributed cache for most recently requests

zoo-keeper for distributing to different Tries, a-k is on T1; k-z is on T2; 

Data collection flow:

API: <phrase, weights> (coming from separate data aggregation services)

Aggregators (A1, A2, A3) -> insert to DB;

DB format is in hourly: phrase, hourstamp, sum of weights;

Appliers (Ap1, Ap2, Ap3) -> update Tries



## Machine Learning System Design

- **Step 1:** Clarify requirements (5 minutes)
- **Step 2:** High-level design (5 minutes)
- **Step 3:** Data deep-dive (10 minutes)
- **Step 4:** Machine learning algorithms (10 minutes)
- **Step 5:** Experimentation (5 minutes)



### How to build a news classifier for articles? 



### How does Facebook news feed work?



### Feature Store

**Feature data:** BERT feature vector

raw data (HDFS, DB, S3) -> batch-processing (Spark, SQL, Python, TensorFlow) -> data quality check -> <ID, feature> -> Feature Store

**Streaming data (runtime features):** # of pages visited in last 30 mins

streaming data (logs, API calls) -> stream-processing (Kafka, Spark) -> <ID, feature> -> Feature Store

**ML deployment:**

Non-latency required: conduct offline experiment and production model retraining;

Latency required: online ML model serving

Components: HDFS/S3/DB data warehouse; meta-data registry; precomputed/caching <ID, feature> for model serving; monitoring: make sure data is consistent between storage and caching



### Build a recommendation page for friend-family Airbnb listing



### Build a recommendation model for Pinterest similar pins

