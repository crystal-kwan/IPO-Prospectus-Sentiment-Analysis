# IPO-Prospectus-Sentiment-Analysis
Analyse Forward-Looking Statements (FLSs) in final IPO prospectuses (SEC Form 424B4) to predict first-day-returns

NLP and ML algorithms used: nltk.tokenize, nltk.corpus, TfidVectorizer, FinBERT, Vader SentimentIntensityAnalyser, Naive Bayes, transformers, AutoTokenizer, AutoModelForSequenceClassification, sklearn, etc.



## Introduction
IPO prospectuses may be of high informational value for investors in pre-IPO and post-IPO valuation as it reflects the management’s prospects of future trajectory of the company. This project aims to provide actionable intelligence for investors in predicting IPO underpricing and returns.


## Data and Methodology
The paper adopts a keyword-based approach to identify FLSs in the prospectuses, by a list of terms such as “intend”, “believe”, etc. that convey expectations, discussions of business plans, and imply uncertainty of future events/ outcomes. An example of FLS identified is: “We expect to face intense competition in the commercial space market and other industries in which we may operate.” Stock price data is scrapped from IPOScoop.com and stored in IPO_data.csv, from which returns is derived. Corresponding IPO prospectuses is obtained using SEC_EDGAR_downloader and stored in text_data.csv.


## Data Analysis
The analytical pipeline experiments with three machine learning algorithms. Firstly, FinBERT, a pre-trained BERT language model in the finance domain is deployed. FinBERT outputs the percentages of the text categorized as “positive”, “negative”, and “neutral”. These 3 generated figures are inputted as independent variables in a linear regression, where the stock returns of the test set is then predicted. The second model used is VADER SentimentIntensityAnalyser which assigns 3 similar scores as FinBERT and a compound score. Lastly, multinomial naïve Bayes Model is selected, it classifies returns as 1 if stock return is positive and 0 otherwise. The model processes the vectorized matrix and determines stock return outcomes.
