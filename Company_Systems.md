# Company Systems

Han Sun



### Pinterest Home-Feed

The first layers are the candidate generators that are based on different algorithms or models. For example, the Pinterest deploys three candidate models, a topic candidate generator, a polaris candidate generator and their unique Pixie random walk generator. All the candidates are mixed and ranked through a ranking model. The output of the ranking model then fits to a blender that includes ads, rules, etc. and finally sends to home feed. 

**Two tower model**: the two tower model is used for ranking. One tower takes the pin's content and makes it through several layers to become a pin's embedding. The other tower takes the user features as well as user engaging histories and becomes a user embedding. A final dot-product is between those two embeddings for ranking. 

**Serving**: the pin's embeddings are usually stable so they are precomputed before serving. They are computed in an offline workflow. For fresh pins (within a few hours of uploading), they will need to be computed online, namely, a real-time fresh pin embedding computation. 

### Pinterest Ads Recommendation System

**Automatic bid**: similar to Yahoo's programmatic bid, the bid price is adjusted automatically in real-time (through control) to achieve the advertiser's goal. For a given ad group, the loss auction data is used to compute how much more it increase so that it can win the bid. Similarly, for all winning auctions, we could calculate the minimum bid decrease to lose that auction. Combining those two cumulative counts, we can recommend the bid change corresponding to highest winning auctions to advertisers. 

**Ad ranking**: a GBDT classification model is used to predict CTR and descendingly rank notifications. 

### Pinterest Data Management for Computer Vision



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