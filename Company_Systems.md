# Machine Learning Systems at Big Companies

**Han Sun**



## Pinterest

### Pinterest Home-Feed

The first layers are the candidate generators that are based on different algorithms or models. For example, the Pinterest deploys three candidate models, a topic candidate generator, a polaris candidate generator and their unique Pixie random walk generator. All the candidates are mixed and ranked through a ranking model. The output of the ranking model then fits to a blender that includes ads, rules, etc. and finally sends to home feed. 

**Two tower model**: the two tower model is used for ranking. One tower takes the pin's content and makes it through several layers to become a pin's embedding. The other tower takes the user features as well as user engaging histories and becomes a user embedding. A final dot-product is between those two embeddings for ranking. 

**Features:** user side: user engagement history features + other user features -> user embedding; pin side: pin's recent performance + topic categorical features -> pin embedding

**Serving**: the pin's embeddings are usually stable so they are precomputed before serving. They are computed in an offline workflow. For fresh pins (within a few hours of uploading), they will need to be computed online, namely, a real-time fresh pin embedding computation. User embedding is computed online.

**Metrics:** offline: recall/precision at top k; online: engagement wins: total saves and closeups increase; total hide drop; increased diversity (some kind of measure).

### Pinterest Ads Recommendation System

**Automatic bid**: similar to Yahoo's programmatic bid, the bid price is adjusted automatically in real-time (through control) to achieve the advertiser's goal. For a given ad group, the loss auction data is used to compute how much more it increase so that it can win the bid. Similarly, for all winning auctions, we could calculate the minimum bid decrease to lose that auction. Combining those two cumulative counts, we can recommend the bid change corresponding to highest winning auctions to advertisers. 

**Ads ranking**: a GBDT classification model is used to predict CTR and descendingly rank notifications. 

**Lookalike audiences:** regression-based and similarity-based approaches; regression-based approaches perform supervised learning on each seed list and excel at encoding seed list structure (lower bias, higher variance), and similarity-based approaches perform user representation learning which solves data sparsity (lower variance, higher bias); 

for each advertiser, we reference the seed list as positive examples and use sampled monthly active users (MAUs) from the targeting country as negative examples, and build a binary classifier using a multi-layer perceptron (MLP) neural network;

weighted binary cross-entropy loss function: weighted by user engagement (CTR) with min-max normalization;

Audience expansion is essentially a ranking problem: find the most similar *k* users from all eligible candidate users; basically, we rank all those eligible based on their similarity scores against a seed and choose top *k*.

### What Attract Me for Pinterest? 

1. ML centered product, state of the art ML focused -> learned from the ML day
2. abundant large scale data content
3. uniqueness of Pinterest's features: search, retrieval, relevance
4. advanced models and infra: models like transformers become available at production
5. Social network nature: graph
6. A recommendation engine for idea searching
7. no baby sitting production pipelines and models so I can focus on more interesting and challenging modeling works

### Random Notes from Pinterest ML Day

#### High level directions

Challenges to balance pinner and creator -> long term engagement driving

How to model long term value for ads (reinforcement learning -> Tomas sampling)

Ads targeting: advertiser to specify their audience - build a bridge between ads and users

71% of Pinterest search is 1-3 words: knowledge graph for search

Key difference between homefeed and search: homefeed does not have a query -> deep understanding of both short term and long term interest. Explicit way to model: knowledge graph; Implicit way: deep learning model; optimize for engagement; transformer based embedding; text and image based embedding

two stage for ads: targeting stage for advertiser; ads relevant models; engagement optimized models

**Unified visual embedding at Pinterest:**

- visual search engine
- daily visual search: 30M, daily active visual search users: 12M
- embedding-based image classifier
- PinSAGE
- train one embedding for each task -> expensive; train a unified (multi-task) visual embedding for multi-tasks;
- how to leverage Pinterest images for unified embeddings: transformer (attention is all you need); BERT: pretrained on large scale; An image is worth 16x16 words (vision transformer: image patches are input);
- ~2.88 avg. labels/images 1329M images, 18k classes; 
- model training: Kubernetes-like platform; multi-node training; streaming data loader; mixed precision training;
- hybrid funnel vision transformer;
- at small size of data: CNN (ResNet101) is better than transformer (hybrid funnel ViT); at large size of data: transformer becomes better;
- takeaway: multi-task embedding -> improvements benefit all tasks; improving downstream applications == adding new task; billion-scale pretraining + large model is great!

**Representation Learning (PinSAGE)**

- content on Pinterest is either: a person, a business: how to make sense of everything? 
- content; search query; user;
- Pin-board graph: bipartite -> understand content better -> Pin Embeddings -> graph neural network; represent a node by embedding; 
- transformer encoder: self node image+text embedding; 50 neigbhours; random walk; feature 4;
- what can we do with strong content embedding:
  - search query: -> 
- random pins are negative samples
- online metric: search product retrieval: product long-clicks, product impressions
- how to learn user embedding? Unsupervised: aggregate use's history content embedding (median); supervised: deep learning approach -> user sequence activity for past year (deal with different length of sequence) -> encode last 255 actions: P2P: click pinID;
- random walks will pad zeros to it if not enough neighbors are there;
- negative sampling is hardest thing: instead of pure random, there is some sort of dependence; some medium-hard negative samples are there

**User Journal Modeling on Pinterest Ads** (Yi-Ping Hsu)

data: short-term interest: 20 most recent user engaged pins for past 7 days -> embedding for user

next: more user events; learn native embeddings; time-aware model

long term interest: collect 500 user events from past 3 months;

**TensorFlow Serving**

how to serve sequence features? -> daily sequence generation workflow -> batch processing -> sequence event handler;

model serving latency: transformer has +3ms over attention model;

latency and CPU cost are sensitive to inference batch size;

MKL-DNN for contraction kernel helped reducing fusedMatMul by 40%;

**warm-up queries** reduce latency for newly deployed models (first coupled of minutes latency is very high).

**Inspired Shopping on Pinterest**

help shoppers find the products to buy effortlessly and with confidence;

build new candidate sources and recommenders;

improve ranking by adding conversion objectives;

candidate retrieval: collaborative filtering; multi-representation of entities

two tower model: see photos;

metrics: product detail page engagement

ranking: multi-task learning: P(engagement|query product, user) -> 4 output heads: save, closeups, clicks, long clicks -> CTR, long CTR;

optimize purchases in ranking: w1*engagement score + w2 * conversion score; p(checkout|impression) = p(checkout|click) * p(click|impression)

learn p(checkout|impression) is very challenging since the data is too sparse, ratio is too low. The overall checkout | click is more relevant. Future work include multi-task learning. 

**Data Management**

MapReduce, Spark

Metrics and Monitoring: QPS, data size, ...

600+ signals, 180 users

ML features store: entity type -> entity key -> feature id -> unified feature representation

training datasets: flatten feature representation tabular dataset

EzFlow -> easy to process dataset by standardizing the data processing flow including event parsing, label assignment, downsampling, etc. 

platforms define interfaces: for people to collaborate effectively; for machines to enable common tools -> data interfaces

### Future of Pinterest

understand utilities of pinner and creator, their objectives -> two side recommendation system -> measurement, how the impact is on each party, define success measurement; system and technology, how to test/run long term impact

how to incentivize creators to continue create content: keep creator engaged



## LinkedIn

### LinkedIn Home-Feed

**Features**:  **identity** - who are you? where do you work? what are your skills? who are you connected with? **content** - how many times was it updated/viewed? how many times was it liked? How old is it? What language is it? What companies, people, or topics are mentioned? **behavior** - what have you liked or posted? Who do you interact with most frequently? Where do you spend most time in your news feed?

**Labels**: traditionally CTR is used, however it is too simplistic. Additional metrics are: time spend reading, insights from your social graph, and additional signals from user experience research. All those metrics are combined together to personalize the feed. 

**Online metrics**: average CTR, number of positive reactions, number of comments, number of shares, number of additional follows, click to landing page (conversion), click to LinkedIn page.

**Feed content**: trending news, jobs, courses, or updates from connections

**Models**: logistic regression, gradient-boosted decision trees and neural networks. LinkedIn specific, boosted decision tables. A decision table is a mapping from a sequence of Boolean tests to a real value. As for the ensemble of many decision trees, a special data structure is designed to skip lots of tests needed. **Response grouping for shared learning:** For optimal transfer learning among objectives, we group objectives into two categories: (1) passive consumption oriented, (2) active consumption oriented. As a result, we split our deep learning network into two towers representing each category, shown as two different colored towers in figure above. We call this “two-tower multi-task deep learning setup.” 

We optimize for cross entropy loss per objective to train a multi-layer network for both the towers. We identified the following key challenges and learnings for the model training process:

- **Model variance:** We observed significant variance in model performance, especially for sparse objectives (e.g., reshare) that correlated with output in both our offline evaluation metrics as well as online A/B testing. We identified the initialization and the optimizers (such as [Adam](https://arxiv.org/abs/1412.6980)) that contribute significantly to variance in the early stage of training. A warm start routine to gradually increase the learning rate helped to overcome a majority of the variance problem.
- **Model calibration:** Our feature and model score monitoring infrastructure (such as [ThirdEye](https://engineering.linkedin.com/blog/2019/01/introducing-thirdeye--linkedins-business-wide-monitoring-platfor)) helped to identify several model calibration challenges, especially at the interaction stage with modeling components external to the deep learning setup. O/E ratio mismatch among different objectives (compared to our previous setup) was one such challenge, and we identified several sampling schemes for negative response training data affecting O/E ratios.  
- **Feature normalization:** Our XGBoost based feature design provides the model with an embedding lookup layer that avoids the feature normalization issues for model training. However, as we expanded into embeddings based features, we realized that normalization would play a major role into the training process. Batch normalization and/or having a translational layer helped alleviate some of these problems.
- **Feature embedding:** XGBoost can encode a derived feature by outputting a selected leaf node. These can be enumerated across the whole forest to set up a contiguous, dense space that fits naturally into embedding lookup. This direct embedding lookup avoids any issue of normalization of numeric features, as the embedding space is chosen by the training process itself regardless of eccentricities of input features. These embeddings are a natural fit for feeding into further layers of a neural network. While this limits TensorFlow to only consider the feature splits that XGBoost chose, by having sufficiently deep trees and a large forest ensemble, we were able to get good coverage of our feature space.
- **Converting feature space:** Another advantage of utilizing XGBoost as encoding input to TensorFlow as an embedding is that it doesn’t require expensive [conversions involving vocabulary maps](https://github.com/linkedin/Avro2TF) from human readable features into a space for easy embedding/matrix multiplication. Previously, adopting this into models required expensive lookups/conversions on the critical path, which was measured at one point to be around 20% of latency in the feed model during [tensor](https://engineering.linkedin.com/blog/2020/feed-typed-ai-features) migration. While not needed for the second pass ranking system due to the above mentioned migration, XGBoost normalization allowed for easy experimentation with TensorFlow in our models that hadn’t yet fully migrated to tensor features.

**Ranking score**: Fp(passive consumption) refers to passive consumption related objectives (member behaviors) such as clicks, [dwell time](https://engineering.linkedin.com/blog/2020/understanding-feed-dwell-time), etc. Fa(active consumption) refers to active contribution interaction objectives such as comments, reshares, etc. Fo(Other)accounts for objectives that do not fall into these two categories, such as [creator side feedback](https://engineering.linkedin.com/blog/2018/10/linkedin-feed-with-creator-side-optimization). Alpha and lambda are tuning parameters balancing for a healthy ecosystem of increased active consumption, while not hurting more passive members’ interests

**Future**: build an automatic framework for easy adopting new models, new features and conducting A/B tests; developing per-member models for personalized feeds; building an interest graph that allows measuring member-to-topic affinity and topic-to-topic relatedness.

**LinkedIn knowledge graph**: it is a kind of entity taxonomy: identifier, definition, canonical name, synonyms in different languages and attributes. Generate rules to identify inaccurate entities. 

- generate candidates: entity candidates are common phrases in member profiles and job descriptions based on intuitive rules;

- disambiguate entities: by representing each phrase as a vector of top co-occurred phrases in member profiles and job descriptions, we developed a soft clustering algorithm to group phrases. An ambiguous phrase can appear in multiple clusters and represent different entities;

- de-duplicate entities: multiple phrases can represent the same entity if they are synonyms of each other. By representing each phrase as a word vector (e.g., produced by a [word2vec model](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) trained on member profiles and job descriptions), we run a clustering algorithm combined with manual validations from taxonomists to de-duplicate entities;

- Translate entities into other languages. Given the power-law nature of the member coverage of entities, linguistic experts at LinkedIn manually translate the top entities with high member coverages into international languages to achieve high precision, and [PSCFG-based machine translation models](http://www.aclweb.org/anthology/P/P05/P05-1.pdf#page=291) are applied to automatically translate long-tail entities to achieve high recall.

- Entity attributes are categorized into two parts: relationships to other entities in a taxonomy, and characteristic features not in any taxonomy. For example, a company entity has attributes that refer to other entities, such as members, skills, companies, and industries with identifiers in the corresponding taxonomies; it also has attributes such as a logo, revenue, and URL that do not refer to any other entity in any taxonomy. The former represents edges in the LinkedIn knowledge graph, which will be discussed in the next section. The latter involves feature extraction from text, data ingestion from search engine, data integration from external sources, and crowdsourcing-based methods, etc.

  All entity attributes have confidence scores, either computed by a machine learning model, or assigned to be 1.0 if attributes are human-verified. The confidence scores predicted by machines are calibrated using a separate validation set, such that downstream applications can balance the tradeoff between accuracy and coverage easily by interpreting it as probability.
  
  

## DoorDash

#### Explore page content generation process

- **Candidate Retrieval:** Fetch data sources from external services that provide the content of the page, such as the Search Service for stores and the Promotion Service for carousels’ metadata. In this case, we only fetch data sources once for the contents on the entire explore page to avoid duplicate calls.
- **Content Grouping:** Grouping content into a set of collections that can be later used for ranking and presentation, such as grouping stores based on association of carousels or store list on the explore page. 
- **Ranking:** Rank the entities within each grouped collection. This step involves resolving the correct model ID, generating the feature values, and making a call to the machine learning prediction service to compute the scores for each ranked candidate. 
- **Experience Decorator:** For the unique set of stores across all collections, we need to hydrate them from external data sources for more user experience-related information, including fetch ETA, delivery fee, images URL, and ratings for stores being displayed.
- **Layout Processor:** This processor collects all the data being fetched and produces placeholders for different presentation styles, including the explore page, form data models for carousels, store lists, and banners.
- **Post Processor:** Rank and post-process all the elements, such as carousels and store lists, on the explore page that are being processed so far in a programmatic way to optimize the user experience.

#### Supply and Demand Balance

**Affected parties:**

- For consumers, a lack of Dasher availability during peak demand is more likely to lead to order lateness, longer delivery times, or inability to request a delivery and having to opt for pick up. 
- For Dashers, a lack of orders leads to lower earnings and longer and more frequent shifts in order to hit personal goals.
- For merchants, an undersupply of Dashers leads to delayed deliveries, which typically results in cold food and a decreased reorder rate.

**Measurement:** 

- Measure at localized level, e.g., New York at dinner time of a particular weekday; quantified as dasher-hours;

**Forecasting model:**

- time-series regression using gradient boosting - lightGBM;
- multivariate problem: need to predict the dasher-hours at thousands of localized levels;
- Extrapolation: for a new city that no historical data is available, use deep learning for latent information extraction (embedding vectors) from features such as population size, traffic conditions, number of available merchants, climate, and geography;
- counterfactuals: in LightGBM, approximate counterfactuals can be generated by changing the inputs that go into the model at inference time;
- **Missing confounding variables**: A model lacking knowledge of weather or holidays might learn that high incentives lead to fewer Dasher hours, when the causal relationship is simply missing a covariate link. 

**Optimizer:** 

**Mixed-integer programming:** subjected to business and financial constraints

**Uncertainty:** Our forecasts tend to be noisiest in the [long tail](https://en.wikipedia.org/wiki/Long_tail) of small regions that have few Dashers and few orders. Because the count of these regions is large and they exhibit high variance, if we don’t explicitly account for this uncertainty we are more likely to generate estimates that by chance will have high undersupply, and thus over-allocate incentives to places that exhibit high variance relative to places that have low variance. To address the issue of variance, we generate expected estimates of hours gap from forecasts using a resampling process. By performing resampling, we essentially measure the impact of undersupply in the context of the likelihood of that happening. 

**Unbiased Predictions:** e generally recommend decoupling forecasting components from decision-making components. Most optimization systems work better if the inputs have stable statistical properties where the forecast predictions are unbiased estimates. For example, it can be tempting to start using an asymmetric loss function in forecasting to align with whether we care more about underpredicting or overpredicting the output. Although this approach is perfect for a [variety of problems](https://doordash.engineering/2021/04/28/improving-eta-prediction-accuracy-for-long-tail-events/#:~:text=Our ETAs predict actual delivery,and when the food arrives.&text=If the ETA is underestimated,and customers will be dissatisfied.) where the output of an ML model is immediately used to drive the decision, for problems where the ML predictions are simply another input into a broader optimization engine, it is best to generate unbiased predictions.

#### Improving ETA Prediction Accuracy for Long-tail Events

- Incorporating real-time delivery duration signals
- Incorporating features that effectively captured long-tail information 
- Using a custom loss function to train the model used for predicting ETAs

**Outliers vs long tails:** Typically they are less than 1% of the data. On the other hand, tail events are less extreme values compared to outliers but occur with greater frequency. In the online retailer example, an outlier might look like a sudden spike in demand when their product happens to be mentioned in a viral social media post. It’s typically very difficult to anticipate and prepare for these outlier events ahead of time, but manageable because they are so rare. On the other hand, tail events represent occurrences that happen with some amount of regularity (typically 5-10%), such that they should be predictable to some degree. 

**Why?** The primary reason is that we often have a relatively small amount of data in the form of ground truth, factual data that has been observed or measured, and can be analyzed objectively. A second reason why tail events are tough to predict is that it can be difficult to obtain leading indicators which are correlated with the likelihood of a tail event occurring. Here, leading indicators refer to the features that correlate with the outcome we want to predict. An example might be individual customers or organizations placing large orders for group events or parties they’re hosting. Since retailers have relatively few leading indicators of these occurrences, it’s hard to anticipate them in advance.

**balancing speed vs. quality**

**Tail Events:**

- Merchants might be busy with in-store customers 
- There could be a lot of unexpected traffic on the road
- The market might be under-supplied, meaning we don’t have enough Dashers on the road to accommodate orders
- The customer’s building address is either hard to find or difficult to enter

**Metric:** on-time percentage, or the percentage of orders that had an accurate ETA with a +/- margin of error as the key north star metric we wanted to improve;

**Key Points:** 

- Historical features : Instead of directly using marketplace health as a continuous feature, we decided to use a form of target-encoding by splitting up the metric into buckets and taking the average historical delivery duration within that bucket as the new feature. With this approach, we directly helped the model learn that very supply-constrained market conditions are correlated with very high delivery times — rather than relying on the model to learn those patterns from the relatively sparse data available;
- Real-time features: Instead, we monitor real-time signals which implicitly capture the impact of those events on the outcome variable we care about — in this case, delivery times. For example, we look at average delivery durations over the past 20 minutes at a store level and sub-region level;
- Custom loss function: a custom asymmetric MSE loss function; by using this approach we need to explicitly state that a late delivery is X times worse than an early delivery.

#### HomeFeed

Retrieval (candidate generator); 1500 -> pre-ranking (logistic regression); 100 -> ranking -> carousals, all restaurants -> reranker/blender/rule-based

**Carousals**: fastest near you; your favorites; most popular local restaurants; other horizontals, groceries, local liquids; try something new; national favorites; new on DoorDash.

**Carousal Ranking:** using a weight for each stores in the carousal and calculate some kind of aggregated level of prediction score;

**Retrieval**: nearest k distance/local, national popular chain stores, past ordered (personalized), nearby popularities, some randomized entities for exploration purposes with less volumes, promoted ones

#### Architecture of the DoorDash ML Platform

**Feature Store** – Low latency store from which Prediction Service reads common features needed for evaluating the model. Supports numerical, categorical, and embedding features.

**Realtime Feature Aggregator** – Listens to a stream of events and aggregates them into features in realtime and stores them in the Feature Store. These are for features such as historic store wait time in the past 30 mins, recent driving speeds, etc.

**Historical Aggregator** – This runs offline to compute features which are longer-term aggregations like 1W, 3M, etc. These calculations run offline. Results are stored in the Feature Warehouse and also uploaded to the Feature Store.

**Prediction Logs** – This stores the predictions made from the prediction service including the features used when the prediction was made and the id of the model used to make the prediction. This is useful for debugging as well as for training data for the next model refresh.

**Model Training Pipeline** – All the production models will be built with this pipeline. The training script must be in the repository. Only this training pipeline will have access to write models into the Model Store to generate a trace of changes going into the Model Store for security and audit. The training pipeline will eventually support auto-retraining of models periodically and auto-deploy/monitoring. This is equivalent to the CI/CD system for ML Models.

**Model Store** – Stores the model files and metadata. Metadata identifies which model is currently active for certain predictions, defines which models are getting shadow traffic.

**Prediction Service** – Serves predictions in production for various use cases. Given a request with request features, context (store id, consumer id, etc) and prediction name (optionally including override model id to support A/B testing), generates the prediction.

#### Retraining Machine Learning Models in the Wake of COVID-19

**Percentile demand model**:  GBM model, quantile loss function, 

**Features**: demand X days ago, the number of new customers gained, and whether a day is a holiday

The model makes thousands of demand predictions a day, one for each region and 30 minute time frame (region-interval). We weigh the importance of a region-interval’s predictions by the number of deliveries that region-interval receives, calculating each-day’s overall weighted demand prediction percentile (WDPP).



## Airbnb

### Improving Deep Learning for Ranking Stays at Airbnb

- **Architecture:** Can we structure the DNN in a way that allows us to better represent guest preference?

  Two tower structure, one is for listing embedding with listing features as input, another is ideal listing embedding, or user embedding. The optimization goal is to make these two embeddings similar. 

- **Bias:** Can we eliminate some of the systematic biases that exist in the past data?

  Previously ranked higher listings tends to be booked more than others and this goes into a loop. To avoid this bias, introduce a dropout rate for the specific position feature. In training, the position feature is set to 0 with a probability of 15%. This additional information lets the DNN learn the influence of both the position and the quality of the listing on the booking decision of a user. While at inference, this position feature is set as 0. 

- **Cold start**: Can we correct for the disadvantage new listings face given they lack historical data?

  For new listings, we can adopt similarity measure based on geographic location and capacity and apply that to find its similar listings. Then, cold-start unavailable features will be an aggregated measure from these similar neighbors. 

- **Diversity of search results:** Can we avoid the majority preference in past data from overwhelming the results of the future?

  In the re-ranking stage, deploy diversity models to promote diversified listings by manipulating the listings. In Airbnb, a so-called query context embedding is applied. 

### WIDeText: A Multimodal Deep Learning Framework

A unified framework to simplify, expedite, and streamline the development and deployment process for this type of multimodal classification tasks.

**Image channel:** listing images, amenities images;

**Text channel:** image captions, reviews, descriptions;

**Dense channel:** categorical features (country and region), numerical features (number of guests, number of rooms, review scores), amenity types and counts (list of amenities: beds, pillows, microwaves, air-conditions, can come from amenity detection models) -> converted to embeddings by GBDT or FM;

**Wide channel:** existing embeddings generated elsewhere

### Deep Learning at Airbnb

**Offline metrics:** precision @ K: set a rank threshold K, compute % relevant in top K, mean average precision (MAP): average precision @ K, Normalized Discounting Cumulative Gain (NDCG)

**Lambdarank NN:** changed the loss function to a NDCG based loss. The data are separated as a pairwise preference formulation where the listings seen by a booker were used to construct pairs of {booked listing, not-booked listing} as training examples. During training we minimized cross entropy loss of the score difference between the booked listing over the notbooked listing. Weighing each pairwise loss by the difference in NDCG resulting from swapping the positions of the two listings making up the pair. 

**Listing ID:** Listings, on the other hand, are subjected to constraints from the physical world. Even the most popular listing can be booked at most 365 times in an entire year. Typical bookings per listing are much fewer. This is fundamental limitation generates data that is very sparse at the listing level. This overfitting is a direct fallout of this limitation.

**Feature Engineering:** ratios, averaging over windows, numerical feature normalization.

### Listing Embeddings in Search Ranking (used in similar carousal and personalization)

There exist several different ways of training embeddings. We will focus on a technique called [Negative Sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). It starts by initializing the embeddings to random vectors, and proceeds to update them via stochastic gradient descent by reading through search sessions in a sliding window manner. At each step the vector of the **central listing** is updated by pushing it closer to vectors of **positive context listings:** listings that were clicked by the same user before and after central listing, within a window of length *m* (*m* = 5)*,* and pushing it away from **negative context listings:** randomly sampled listings (as chances are these are not related to the central listing).

- **Using Booked Listing as Global Context:** We used sessions that end with the user booking the listing (purple listing) to adapt the optimization such that at each step we predict not only the neighboring clicked listings but also the eventually booked listing as well. As the window slides some listings fall in and out of the context set, while the booked listing always remains within it as global context (dotted line) and is used to update the central listing vector.
- **Adapting to Congregated Search:** Users of online travel booking sites typically search only within a single market, i.e. in the location they want to stay at. As a consequence, for a given *central listing*, the *positive* *context listings* mostly consist of listings from the same market, while the *negative context listings* mostly consists of listings that are not from the same market as they are sampled randomly from entire listing vocabulary. We found that this imbalance leads to learning sub-optimal within-market similarities. To address this issue we propose to add a set of random negatives *Dmn,* sampled from the market of the central listing.

**Cold-start Embeddings.** Every day new listings are created by hosts and made available on Airbnb. At that point these listings do not have an embedding because they were not present our training data. To create embeddings for a new listing we find 3 geographically closest listings that do have embeddings, and are of same listing type and price range as the new listing, and calculate their mean vector.

**Evaluation:** By calculating cosine similarities between embeddings of the clicked listing and the candidate listings we can rank the candidates and observe the rank position of the booked listing.

