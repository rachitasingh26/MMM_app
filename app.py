import jax.numpy as jnp
import numpyro
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import plot
from lightweight_mmm import preprocessing
from lightweight_mmm import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

@st.cache_data
def load_and_process_data(df_main):
    df_main = pd.read_csv("MMM_Data.csv")
    mdsp_cols = ["Newspaper", "Radio", "Social Media", "TV"]
    hldy_cols = [col for col in df_main.columns if 'hldy_' in col]
    seas_cols = [col for col in df_main.columns if 'seas_' in col]
    control_vars =  hldy_cols + seas_cols
    sales_cols =['Sales']
    clusters_rename = {'Cluster_0.0': 'Cluster_1', 'Cluster_1.0': 'Cluster_2', 'Cluster_2.0': 'Cluster_3', 'Cluster_3.0': 'Cluster_4', 'Cluster_4.0': 'Cluster_5'}
    df_main = df_main.rename(columns=clusters_rename).dropna()
    return mdsp_cols,control_vars, sales_cols, df_main


def fit_mmm_model(df_main):
    mdsp_cols,control_vars, sales_cols, df_main = load_and_process_data(df_main)
    SEED = 105
    data_size = len(df_main)
    n_media_channels = len(mdsp_cols)
    n_extra_features = len(control_vars)
    media_data = df_main[mdsp_cols].to_numpy()
    extra_features = df_main[control_vars].to_numpy()
    target = df_main['Sales'].to_numpy()
    costs = df_main[mdsp_cols].sum().to_numpy()
    test_data_period_size = 24
    split_point = data_size - test_data_period_size
    media_data_train = media_data[:split_point, ...]
    media_data_test = media_data[split_point:, ...]
    extra_features_train = extra_features[:split_point, ...]
    extra_features_test = extra_features[split_point:, ...]
    target_train = target[:split_point]
    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean, multiply_by=0.15)
    media_data_train = media_scaler.fit_transform(media_data_train)
    extra_features_train = extra_features_scaler.fit_transform(extra_features_train)
    target_train = target_scaler.fit_transform(target_train)
    costs = cost_scaler.fit_transform(costs)
    mmm = lightweight_mmm.LightweightMMM(model_name="hill_adstock")
    number_warmup=1000
    number_samples=1000
    mmm.fit(media=media_data_train, media_prior=costs, target=target_train, extra_features=extra_features_train, number_warmup=number_warmup, number_samples=number_samples, media_names = mdsp_cols, seed=SEED)
    return mmm, target_scaler, cost_scaler, n_media_channels, extra_features, extra_features_scaler, media_scaler, SEED, media_data


def clusters():
    st.title("Customer Segmentation")
    st.empty()
    st.write("The dataset comprises of 200 data points. The features of the dataset include Customer ID, Customer age, Customer gender, Annual income and Spending score.")
    st.empty()
    uploaded_file = st.file_uploader("Upload the dataset", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    def main():
        st.empty()
        st.subheader("Understanding the dataset:")
        st.empty()
        st.write("Before we begin with the clustering process, it is crucial to understand the dataset and perform essential EDA processing.")
        st.write("The following distplots reveal key insights into a dataset characterized by varying age, annual income, and spending scores. The age distribution displays a bimodal pattern, with predominant concentrations around the 20-30 and 50-60 age brackets, suggesting a significant representation of younger adults and middle-aged individuals. In terms of annual income, the data skews towards the lower end, with most individuals earning between 50k and 75k, indicating this as a common economic demographic. Lastly, the spending score is rather uniformly spread across the spectrum, with a modest uptick in the middle range (40-60), pointing to an even distribution of spending habits among the subjects, without a distinct bias towards higher or lower scores.")
        st.empty()
        # Distplot for Age, Annual Income, and Spending Score
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        for i, feature in enumerate(features):
            sns.distplot(df[feature], bins=20, ax=ax[i])
            ax[i].set_title(f'Distplot of {feature}')
        st.pyplot(fig)

        st.empty()
        st.write("The following countplot shows that the number of female customers are higher than the number of male customers. This is just to understand the gender variation in the population that may be useful while marketing certain products/services.")
        st.empty()
        # Countplot for Gender
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.countplot(y='Gender', data=df, ax=ax)
        st.pyplot(fig)

        st.empty()
        st.write("The scatter plots reveal no apparent correlation between age and annual income across genders, showing individuals of varying ages across all income brackets. The distribution across gender is also fairly even, suggesting that age and gender may not be primary indicators of income levels. In contrast, the relationship between annual income and spending score displays some degree of positive correlation; as income increases, spending scores tend to rise. Nonetheless, the presence of high-income individuals with varied spending scores indicates that factors beyond income influence spending behavior. These patterns could imply the presence of distinct customer segments, potentially valuable for targeted marketing strategies.")
        # Scatter plots for Age vs Annual Income and Annual Income vs Spending Score by Gender
        for feature, xlabel, ylabel, title in [
            ('Age', 'Age', 'Annual Income (k$)', 'Age vs Annual Income w.r.t Gender'),
            ('Annual Income (k$)', 'Annual Income (k$)', 'Spending Score (1-100)', 'Annual Income vs Spending Score w.r.t Gender')]:
        
            fig, ax = plt.subplots(figsize=(15, 6))
            for gender in ['Male', 'Female']:
                subset = df[df['Gender'] == gender]
                ax.scatter(subset[feature], subset[ylabel], s=200, alpha=0.5, label=gender)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            st.pyplot(fig)
        
        st.empty()
        st.subheader("Performing K-means Clustering:")
        st.empty()
        st.write("To determine the number of clusters, WCSS method is used to generate an elbow graph. The WCSS is the sum of the variance between the observations in each cluster. It measures the distance between each observation and the centroid and calculates the squared difference between the two.")
        st.write("From the following graph, we can observe that between number of cluster = 4 to number of cluster = 6 there has been substantial decrease(an elbow) hence, we chose the K value for our dataset as 5. After determining the optimal number of clusters, K Means algorithms is run to obtain the five unique clusters or customer segments.")
        st.empty()
        # Elbow Method Graph for KMeans Clustering
        X = df.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)']].values
        scaler = MinMaxScaler().fit(X)
        X_scaled = scaler.transform(X)
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(range(1, 11), wcss, color='green', linestyle='dashed', linewidth=3,
                    marker='o', markerfacecolor='blue', markersize=12)
        plt.title('The Elbow Point Graph')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.grid(True)
        st.pyplot(fig)

        st.empty()
        st.write("After performing clustering, we can derive the following insights - ")
        st.empty()
        # KMeans Clustering Visualization
        kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['green', 'yellow', 'red', 'purple', 'blue']
        for i in range(5):
            plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='*', label='Centroids')
        plt.title('Customer Groups')
        plt.xlabel('Annual Income')
        plt.ylabel('Spending Score')
        plt.legend()
        st.pyplot(fig)
        st.empty()
        st.write("1. Cluster 1: These are average income earners with average spending scores. They are cautious with their spending.")
        st.write("2. Cluster 2: The customers in this group are high income earners and with high spending scores. They bring in profit. Discounts and other offers targeted at this group will increase their spending score and maximize profit.")
        st.write("3. Cluster 3: This group of customers have a higher income but they do not spend more at the store. One of the assumption could be that they are not satisfied with the services rendered. They are another ideal group to be targeted by the marketing team because they have the potential to bring in increased profit.")
        st.write("4. Cluster 4: Low income earners with low spending score.")
        st.write("5. Cluster 5: These are low income earning customers with high spending scores. One assumption for this can be that they enjoy and are satisfied with the services rendered.")
        st.empty()
        st.write("Based on the derived insights, cluster 2,3 and 5 are the focus segments that are crucial in maximising sales.")
        
    if __name__ == '__main__':
        main()
    
    
def impact():
    st.title("Media Channel Impact")
    st.empty()
    st.write("The identified customer segments are incorporated in the MMM dataset as dummy variables. The MMM dataset consists of the following features – Media channels (Newspaper, TV, Radio, Social Media), Sales, Holidays, Seasonality and Clusters.")
    st.empty()
    upload_file = st.file_uploader("Upload the dataset", type=['csv'])
    if upload_file is not None:
        df_main = pd.read_csv(upload_file)
        st.write(df_main)
    st.empty()
    st.write("To understand the impact of different media channels on different customer segments, we ran a regression model by initialising interaction terms (media x cluster) and assessed the impact on the basis of coefficient values.")
    st.write("There are four media channels, namely – newspaper, TV, radio and social media, along with five distinct customer segments. The visualisation depicts either a positive or negative impact of a media channel on different customer segments, and the impacts are based on the coefficient values of the interaction terms (media x cluster). A positive impact estimates that running ads through that media channel could convert that specific group of customers into users of their product/service. A negative impact estimates the opposite.")
    mdsp_cols,control_vars, sales_cols, df_main = load_and_process_data(df_main)
    media_channels = ["Newspaper", "Radio", "Social Media", "TV"]
    clusters = ['Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5']
    for media in media_channels:
        for cluster in clusters:
            interaction_term = f"{media}_X_{cluster}"
            df_main[interaction_term] = df_main[media] * df_main[cluster]
    X_columns = media_channels + clusters + [f"{media}_X_{cluster}" for media in media_channels for cluster in clusters]
    X = df_main[X_columns]
    y = df_main['Sales']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    interaction_coeffs = {term: coef for term, coef in model.params.items() if '_X_' in term}
    media_channels = ["Newspaper", "Radio", "Social Media", "TV"]
    clusters = ['Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5']
    n_clusters = len(clusters)
    cluster_positions = np.arange(len(media_channels))
    bar_width = 0.1
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, cluster in enumerate(clusters):
        impacts = [interaction_coeffs[f'{media}_X_{cluster}'] for media in media_channels]
        plt.bar(cluster_positions + i * bar_width, impacts, width=bar_width, label=cluster)
    plt.xlabel('Media Channel', fontsize=14)
    plt.ylabel('Impact (Coefficient Value)', fontsize=14)
    plt.title('Impact of Media Channels on Different Clusters', fontsize=16)
    plt.xticks(cluster_positions + bar_width * (n_clusters - 1) / 2, media_channels)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    st.empty()
    st.write("The main aim of running a market mix model is to maximise sales. To achieve that, one of the contributing factors would be to focus on those media channels and customer segments that are experiencing a positive impact, keeping in mind the focus segments derived from the previous phase.")
    st.write("The insights can be stated as –")
    st.write("1. Newspaper positively impacts customer segments 2,3,4 with negative impacts on 1 and 5.")
    st.write("2. Radio positively impacts customer segments 1,2,3,4 with negative impact on 5.")
    st.write("3. Social Media positively impacts segments 1,4 with negative impacts on 2,3 and 5.")
    st.write("4. TV positively impacts customer segments 1,2,3,4,5.")
    
def mmm_analysis(df_main):
    st.title("Media Data Analysis")
    st.empty()
    st.write("The MMM model is based on Bayesian Regression, which incorporates prior knowledge or beliefs into the modelling process, allowing for more robust and flexible estimation of marketing channel effects. It adapts to data uncertainty and complexity by updating beliefs with incoming data, offering probabilistic insights into the effectiveness of marketing spends across different channels. This approach provides a deeper understanding of marketing dynamics and decision-making under uncertainty, enhancing strategic planning and optimization in MMM.")
    st.write("A python framework called LightweightMMM has been used to perform Bayesian Modelling, and allows users to choose from 3 different approaches to demonstrate a lagged effect of media channels on sales – Ad stock, Carryover and Hill Ad stock. The choice of approach depends on the specific business use case and requirement. For our model, we have applied the ‘Hill Ad stock’ approach.")
    st.empty()
    if st.button("Run Analysis"):
        st.write("Please wait while the model runs in the background. Estimated time is about 1-2 minutes.")
        st.empty()
        mdsp_cols,control_vars, sales_cols, df_main = load_and_process_data(df_main)
        mmm, target_scaler, cost_scaler, n_media_channels, extra_features, extra_features_scaler, media_scaler, SEED, media_data = fit_mmm_model(df_main)
        media_contribution, roi_hat = mmm.get_posterior_metrics(target_scaler=target_scaler, cost_scaler=cost_scaler)  
        baseline_plot = plot.plot_media_baseline_contribution_area_plot(media_mix_model=mmm,
                                                            target_scaler=target_scaler,
                                                            fig_size=(30,10),
                                                            channel_names = mdsp_cols
                                                            )
        st.empty()
        st.write("The media channel attribution visualisation depicts contribution over a period of time, with the baseline contribution indicative of a baseline level of sales that is expected without any additional media impact.")
        st.empty()
        st.pyplot(baseline_plot)
        media_contri = plot.plot_bars_media_metrics(metric=media_contribution, metric_name="Media Contribution Percentage", channel_names=mdsp_cols)
        st.empty()
        st.write("Media contribution refers to the quantified impact that different media channels have on a specific marketing objective, such as sales, brand awareness, or customer engagement. It essentially measures how much each channel—be it television, online advertising, social media etc.—contributes to the overall effectiveness of a marketing campaign or strategy. For our project scenario, the specific marketing objective is sales.")
        st.write("From the visualisation below, we can interpret the contribution of each media channel towards the overall sales(baseline metric) , with TV being the highest contributor at roughly 60%. It is followed by Radio at around 20%, Social Media at around 8% and Newspaper at an estimate of 2% contribution.")
        st.empty()
        st.pyplot(media_contri)
        roi_plot = plot.plot_bars_media_metrics(metric=roi_hat, metric_name="ROI hat", channel_names=mdsp_cols)
        st.write("Return on Investment (ROI) is a metric that measures the efficiency of various marketing investments. It calculates the return generated from different marketing channels or tactics relative to their costs. ROI is expressed as a percentage or ratio, providing a straightforward way to compare the effectiveness of different marketing efforts.")
        st.empty()
        st.pyplot(roi_plot)
        st.empty()
        st.write("TV commands the largest share of sales impact, indicating that it is potentially the most effective channel for driving sales volume. On the other hand, Radio, while perhaps not contributing as much to sales volume directly as TV, offers the best ROI. While TV might be essential for visibility and volume, the higher ROI of Radio suggests that increasing the budget allocation there could yield proportionally greater returns. It is followed by TV, Social Media and Newspaper. This understanding is particularly useful in the budget optimisation process.")
        st.empty()
        response_curves = plot.plot_response_curves(media_mix_model=mmm, target_scaler=target_scaler, seed=SEED)
        st.pyplot(response_curves)
        st.empty()
        st.write("Marketers make use of marketing response curves and marginal revenue curves that shows the maximum amount required to allocate to a marketing channel before revenue diminishes. The information the response curves provide are translated into budget allocations for each of the marketing channels. This ensures that the maximum ROI is achieved.")

    
def optimisation(df_main):
    st.title("Budget Optimisation and Allocation")
    st.empty()
    st.write("The next step after understanding the current trends and data through media contribution estimates, ROI estimates and response curves is to perform optimisation. SLSQS, a gradient based optimisation algorithm is used to perform a maximisation task, which gives an optimal channel wise budget allocation that will maximise the sales. The predicted sales value after performing the suggested budget allocation is also generated as output.")
    st.write("The optimisation is run to give the estimated budget for maximising sales over a period of time. The period of time can be specified in days, weeks, months or years. Since our input time series data is weekly data, we will take the number of time periods as a weekly input.")
    st.empty()
    st.image("optimisation.png", caption = "Gradient based MMM optimisation")
    st.empty()
    n_time_periods = st.number_input("Enter the number of time periods (weeks):", min_value=1, step=1)
    if st.button("Run Optimization"):  
        st.write("Please wait while the optimisation is in progress. Estimated time is about 1-2 minutes.")
        mdsp_cols,control_vars, sales_cols, df_main = load_and_process_data(df_main)
        mmm, target_scaler, cost_scaler, n_media_channels, extra_features, extra_features_scaler, media_scaler, SEED, media_data = fit_mmm_model(df_main)
        media_data = df_main[mdsp_cols].to_numpy()
        prices = jnp.ones(mmm.n_media_channels)
        budget = jnp.sum(jnp.dot(prices, media_data.mean(axis=0)))* n_time_periods
        solution, kpi_without_optim, previous_media_allocation = optimize_media.find_optimal_budgets(
                n_time_periods=n_time_periods,
                media_mix_model=mmm,
                extra_features=extra_features_scaler.transform(extra_features)[:n_time_periods],
                budget=budget,
                prices=prices,
                media_scaler=media_scaler,
                target_scaler=target_scaler,
                seed=SEED)
        optimal_buget_allocation = prices * solution.x
        previous_budget_allocation = prices * previous_media_allocation
        fig = plot.plot_pre_post_budget_allocation_comparison(media_mix_model=mmm, 
                                                            kpi_with_optim=solution['fun'], 
                                                            kpi_without_optim=kpi_without_optim,
                                                            optimal_buget_allocation=optimal_buget_allocation, 
                                                            previous_budget_allocation=previous_media_allocation, 
                                                            figure_size=(10,10),
                                                            channel_names=mdsp_cols)
        st.pyplot(fig)
        st.empty()
        st.write("The optimisation output suggests that increasing the budget allocated towards TV and Radio channels and decresing the budget for Newspaper and Social Media would lead to a significant maximisation in sales over a period of 20 weeks. The sales value is in thousands.")
        

def insights():
    st.title("Overall Model Insights")
    st.empty()
    st.write("The insights derived from the model outputs and analysis suggest a strategic adjustment of marketing budgets and efforts across different media channels to better target specific customer segments that are most responsive. Here's an elaboration on the recommended actions")
    st.empty()
    st.image("insights.png", caption = "Overall Insights")
    st.empty()
    st.write("1. Media Channel - Newspaper")
    st.write("The recommendation to decrease the budget for newspaper advertising is likely due to a lower return on investment or less effectiveness compared to other channels. However, there’s still some value in using this channel for reaching customer segments 2 and 3, which may respond more positively to newspaper ads than other segments. The focus here would be on precision targeting within the newspaper medium to maximize impact where it remains relevant.")    
    st.empty()
    st.write("2. Media Channel - TV")
    st.write("TV advertising is shown to be effective and thus warrants an increased budget. This channel's broad reach and influence are particularly significant for customer segments 2, 3, and 5. These segments may have demonstrated a strong engagement with TV ads or a higher likelihood of conversion following exposure to such ads. By boosting investment in TV, the brand aims to capitalize on this channel's strengths to reach and influence these key segments more profoundly.")
    st.empty()
    st.write("3. Media Channel - Radio")
    st.write("Similar to TV, there is a suggestion to increase the budget for radio advertising. This medium is effective for targeted marketing towards customer segments 2 and 3, which implies these segments may consume radio content frequently or respond well to radio campaigns. Increasing radio spend should focus on times or programs that these segments are most engaged with.")
    st.empty()
    st.write("4. Media Channel - Social Media")
    st.write("The recommendation to decrease the social media budget indicates that this channel is not performing well in terms of impacting the key customer segments 2, 3, and 5. It’s possible these segments are either not as active or receptive on social media, or the social media campaigns are not resonating with them. Given that all focus segments are experiencing a negative impact, efforts here should be minimal and highly strategic. Focusing on segment 5, which is least negatively impacted, suggests trying to refine the approach on social media to this group, possibly by understanding and leveraging the specific aspects of social media that do engage them.")


def app():
    df_main = load_and_process_data("MMM_Data.csv")
    page = st.sidebar.selectbox("Navigation Bar", ["Introduction", "Customer Segmentation","Media Channel Impact", "Media Data Analysis", "Budget Optimisation and Allocation", "MMM Insights"])

    if page == "Introduction":
        st.title("Applications of Customer Segmentation and Bayesian Modelling in Market Mix Models")
        st.empty()
        st.write("In the digital age, where data drives decisions, the sanctity and accuracy of data have become paramount. However, with the advent of stringent data privacy regulations and the increasing use of ad blockers by consumers, the gap between captured data and actual user behaviour has widened, posing a significant challenge for marketers.")
        st.write("This discrepancy, which can be as high as 30% in conversion attribution, undermines the effectiveness of tracking tools and leaves marketers navigating in the dark, unable to gauge the real impact of their campaigns. This misalignment not only obscures the visibility of campaign performance but also leads to inefficient budget allocation across various online mediums, culminating in substantial financial waste and missed opportunities.")
        st.write("Market Mix Modeling (MMM) emerges as a robust alternative in this scenario, offering a way to circumvent the limitations posed by data privacy issues and the inherent shortcomings of attribution models. MMM can guide strategic decisions on budget allocation, channel optimization, and marketing mix strategy by identifying the incremental impact of each marketing input on sales. An implementation of Bayesian Models along with understanding customer segment level impact of media channels can help generate more granular insights and drive business decisions effectively.")
        st.write("An overall workflow of the project along with specific modules is depicted through the diagram below - ")
        st.empty()
        st.image("workflow.png", caption = "Project Workflow")
        st.empty()
        st.write("The project is essentially divided into two phases and four modules within the phases. Each tab in the navigation bar represents one module, followed by deriving the overall insights after running the complete model.")
    elif page == "Customer Segmentation":
        clusters()

    elif page == "Media Channel Impact":
        impact()
        
    elif page == "Media Data Analysis":
        mmm_analysis(df_main)
        
    elif page == "Budget Optimisation and Allocation":
        optimisation(df_main)
    elif page == "MMM Insights":
        insights()

if __name__ == "__main__":
    app()

