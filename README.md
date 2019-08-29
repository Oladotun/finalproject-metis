# Final Metis  Project 5: Using Classification to Interprete and Predict Kickstarter Project 
## Dotun Opasina 

## SCOPE:

Kickstarter a crowdfunding website helped people raise a total of [$1.6 Billion ](https://www.cbsnews.com/news/inside-kickstarter-crowdfunding-ideas-that-fail-to-materialize/) so far. For so many campaigns that succeed on kickstarter, there are are even so many that do not. My goal is to create a predictor website that helps fund seekers to predict whether their campaign will suceed or not.

## METHODOLOGY:
1. Get Kickstarter campaigns from 2013-2019 from [Kickstarter scraper website](https://webrobots.io/kickstarter-datasets/) <br>
2. Download data set into MongoDB and do some data preprocessing <br>
3. Load MongoDb Data into Pandas<br>
4. Create some EDA and Data visualization <br>
   - Best times and days to create a campaign that will suceed? <br>
   - What categories of campaigns get funded the most? <br>
      -  Connect that information to US market for those categories to see whether the size of market affects how they are funded ? <br>
5. Feature Engineering and selection of Kickstarter campaign observations <br>
6. Utilize classification algorithms such as Neural Networks, Logistic regression for prediction<br>
7. Perform Natural Language Processing on some of the columns e.g the text description of the campaigns is suitable for sentinment analysis and topic modeling .<br>
8. Build predictor output website <br>

## DATA SOURCES:
-  [Kickstarter](https://webrobots.io/kickstarter-datasets/) <br>
-  USA market data <br>

## TARGET
- MVP: Perform EDA visualizations and Feature selections using classification algorithms.
- Final: Build boostrap designed Flask predictor Website to allow people


## THINGS TO CONSIDER
- Selecting the proper features to predict the result of a campaign might be challenging as my datasets contains about 25 different columns .


