## Applications of Customer Segmentation and Bayesian Modelling in Market Mix Models

Businesses are undergoing a digital transformation. It is through this digital transformation that companies have developed digital methods to reach customers through advertising. Therefore advertisers seek to properly measure the effectiveness of the different marketing/media channels. Since the adoption of additional advertisement streams, it has become necessary to be able to measure the effectiveness of online and offline media and the contributions they have towards achieving business goals.

The project aims to understand market mix modelling and consumer psychology, study existing methods, and implement a significant model that enhances insights that help marketers and analysts make strategic decisions regarding advertising campaigns and products. The end goal caters to budget optimization for different marketing channels to maximize sales, get a better reach, and improve customer acquisition. This can be achieved by fulfilling the following objectives:

1. Identify the effect of different media channels on different customer segments.
2. Generate media insights, and study media channels’ Return on Investment (ROI) contribution to sales via Bayesian MMM.
3. Perform budget optimization and allocation for greater ROI.
4. Perform a comparative analysis of the Bayesian model with a frequentist method (linear regression).
 
<br/>

### Project Workflow

The project is essentially divided into 2 phases - in Phase 1, we gather customer data to perform customer segmentation using K-means clustering to get five different customer segments as outputs, based on the annual income and spending scores of the customers. The clusters are then used in the MMM dataset as dummy variables to run Phase 2 of the model, which is essentially performing market mix modelling.

![image](https://github.com/rachitasingh26/MMM_app/assets/87617147/d75f63ff-5d3b-4090-862d-27c8df6bc7f3)

<br/>
<br/>

### Project Description

The MMM project has 4 modules or components, all of which run in sequence to yield the overall optimised result and insights. For a detailed walkthrough of the project, refer to the [MMM Project Documentation](https://github.com/user-attachments/files/16054718/MMM.Project.Documentation.pdf)

![image](https://github.com/rachitasingh26/MMM_app/assets/87617147/821d815a-0b51-4059-9c29-f374ba7b0ce6)

K-Means clustering, Bayesian Regression and Sequential Least Squares Quadratic Programming (a type of gradient optimisation algorithm) have been used throughout the project. LightweightMMM, a Market Mix Modelling framework by Google was helpful in terms of incoporating our beliefs, specifiying priors and likelihoods. The foundation of this project is to differentiate between frequentist (linear regression) and bayesian models, and implement a model that takes the frequesntists model limitations into account as well. 

While the model is succesful in accounting for uncertainity and seasonality variations that could affect the performance of media channels, we encountered a few limitations as well. We have assumed that seasonal variation is constant and have thus worked on an additive model, but future work could include working on a multiplicative model which is useful when seasonal variation increases over time. Additionally, there are times when a media channel performs well in one region and does poorly in some other. To address this, implementing a geo-level MMM model is a possibility that can generate more granular insights based on data from specific geographic regions. Dealing with selection bias is crucial for MMM models to improve reliability on results. Selection bias occurs when an input media variable is correlated with an unobserved demand variable (e.g. seasonality, ad targeting), which in turn drives sales.

<br/>
<br/>

### Running the project

The project is deployed via a web app on Streamlit Cloud. It can be accessed via this link - https://market-mix-modeling.streamlit.app/

To run the project locally on your system:
1. Download the _app.py_ file from this repository.
2. If you do not have streamlit installed, you can install it via this command - _pip install streamlit_.
3. Once streamlit is installed, open terminal/command line on your system or on Anaconda (preferred) and enter a folder or directory for accessing the file. FOr example, if the file has been saved to your downloads, navigate to that folder using _cd Downloads_ command.
4. Next, input _streamlit run app.py_ command to run the file. The deployed web application will open on a web browser via Streamlit Cloud.



