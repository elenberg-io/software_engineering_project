import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
import altair as alt


st.set_page_config(
    page_title="Data Analysis App",
    page_icon="🧊",
    initial_sidebar_state="expanded"
)

class DataUploader:
    """
    Class to upload tabular data
    """
    def upload_data(self):
        return st.file_uploader("Select a file with tabular data in csv or xlsx format:", type=["csv", "xlsx"])

    def read_data(self, uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
            
        except Exception as e:
            st.error(f"There's been an error loading the data: {e}")
            return None


class DataViewer:
    """
    Class to view the dataframe data and its information
    """
    def view_data(self, df):
        st.write("Data uploaded:")
        st.dataframe(df)
    
    def view_table_info(self, df):
        st.write(f"Number of samples: {df.shape[0]}")
        st.write(f"Number of features: {df.shape[1] - 1}")


class DataVisualizer:
    """
    Class to visualize data
    """
    def __init__(self, df):
        self.df = df
        self.features = df.iloc[:, :-1]
        self.labels = df.iloc[:, -1]

    def pca_reduction(self):
        pca = PCA(n_components=2)
        components = pca.fit_transform(self.features)
        self.plot_2d(components, "PCA with 2 components")

    def tsne_reduction(self):
        tsne = TSNE(n_components=2, random_state=26)
        components = tsne.fit_transform(self.features)
        self.plot_2d(components, "t-SNE")

    def plot_2d(self, components, title):
        plt.figure(figsize=(10, 8))
        plt.scatter(components[:, 0], components[:, 1], c=self.labels, cmap='plasma')
        plt.colorbar()
        plt.title(f"{title} 2D Visualization")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        st.pyplot(plt)

    def eda_charts(self):
        st.write("Scatter Plot")
        x_axis = st.selectbox("Select X-axis feature", self.df.columns[:-1], index=0)
        y_axis = st.selectbox("Select Y-axis feature", self.df.columns[:-1], index=1)
        st.write(f"Scatter plot of {x_axis} vs {y_axis}")
        st.scatter_chart(self.df, x=x_axis, y=y_axis, color="#ffaa00")

        st.write("Histogram")
        column = st.selectbox("Select feature for histogram", self.df.columns[:-1])
        num_bins = st.slider("Select number of bins", min_value=10, max_value=30, value=10, step=5)
        hist = alt.Chart(self.df).mark_bar().encode(
            alt.X(column, bin=alt.Bin(maxbins=num_bins), title=column),
            y='count()',
            tooltip=[column, 'count()']
        ).interactive()
        st.altair_chart(hist, use_container_width=True)


class Classifier:
    def __init__(self, df):
        self.df = df
        self.features = df.iloc[:, :-1]
        self.labels = df.iloc[:, -1]

    def knn_classifier(self, n_neighbors):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=26)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)

    def random_forest_classifier(self, max_depth):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=26)
        tree = RandomForestClassifier(max_depth=max_depth)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True)

    def compare_classifiers(self):
        st.subheader("KNN Classifier")
        n_neighbors = st.slider("Select number of neighbors (k) for KNN", 1, 20, 5)
        knn_auc, knn_report = self.knn_classifier(n_neighbors)
        st.write(f"KNN Accuracy: {knn_auc:.2f}")
        st.write("KNN Classification Results:")
        st.json(knn_report)

        st.subheader("Random Forest Classifier")
        max_depth = st.slider("Select max depth for Random Forest", 5, 20, 5)
        tree_auc, tree_report = self.random_forest_classifier(max_depth)
        st.write(f"Random Forest Accuracy: {tree_auc:.2f}")
        st.write("Random Forest Classification Results:")
        st.json(tree_report)

        st.subheader("Comparison Summary")
        best_classifier = "KNN" if knn_auc > tree_auc else "Random Forest"
        st.write(f"The best classifier is: {best_classifier}")

class Clusterer:
    def __init__(self, df):
        self.df = df
        self.features = df.iloc[:, :-1]

    def kmeans_clustering(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=26)
        kmeans.fit(self.features)
        kmeans_labels = kmeans.fit_predict(self.features)
        kmeans_davies_bouldin = davies_bouldin_score(self.features, kmeans_labels)
        return silhouette_score(self.features, kmeans.labels_), kmeans_davies_bouldin

    def affinity_clustering(self, n_clusters):
        aff_clust = AffinityPropagation(damping=0.5)
        aff_clust.fit(self.features)
        aff_labels = aff_clust.fit_predict(self.features)
        aff_davies_bouldin = davies_bouldin_score(self.features, aff_labels)
        return silhouette_score(self.features, aff_clust.labels_), aff_davies_bouldin

    def compare_clustering(self):
        st.subheader("K-Means Clustering")
        n_clusters_kmeans = st.slider("Select number of clusters for K-Means", 2, 10, 3)
        kmeans_score, kmeans_davies_bouldin = self.kmeans_clustering(n_clusters_kmeans)
        st.write(f"K-Means Silhouette Score: {kmeans_score:.2f}, and Davies Bouldin metric: {round(kmeans_davies_bouldin, 2)}")

        st.subheader("Affinity Propagation Clustering")
        n_clusters_agg = st.slider("Select number of clusters for Affinity Propagation Clustering", 0.1, 1.0, 0.1)
        aff_score, aff_davies_bouldin = self.affinity_clustering(n_clusters_agg)
        st.write(f"Propagation Clustering Silhouette Score: {aff_score:.2f}, and Davies Bouldin metric: {round(aff_davies_bouldin, 2)}")

class InfoPresenter:
    def show_info(self):
        st.write("""
            This section of the application is dedicated to presenting information about the application, how it works, the development team, and the specific tasks each team member worked on.
        """)

        st.subheader("About the Application")
        st.markdown("""
        - **Functionality**: The application allows users to perform various data analysis and machine learning tasks.
        - **Features**: It includes functionalities for loading data, visualizing data, performing classification and clustering, and presenting information.
        """)
        
        st.subheader("How it Works")
        st.markdown("""
        - **Tab Navigation**: Users navigate through different tabs in the  app to access different functionalities.
        - **Data Loading**: Users can upload tabular data in csv or xlsx format in the "Load Data" tab.
        - **Data Visualization**: The "Visualize Data" tab offers options for visualizing high-dimensional data using techniques like PCA and t-SNE.
                    For EDA we have interactive scatter plot and histogram where the users picks the column(s) to explore and a parameter.  
        - **Machine Learning**: The "Classification" and "Clustering" tabs allow users to compare interactively the performance of the ML models 
                    KNN and Random Forest for classification and compare kmeans and affinity propagation for Clustering
                    selecting a parameterf for the models.
        - **Information Presentation**: The "Info" tab provides background information about the application and its development team.
        """)

        st.subheader("Development Team")
        st.markdown("""
        - **Team Members**:
            - Elena Vergou: P2006041
        - **Roles**:
            - Elena Vergou: Performed project planning, designed and implemented the application. 
        """)

class MainApplication:
    def __init__(self):
        self.data_uploader = DataUploader()
        self.data_frame_viewer = DataViewer()
        self.df = None

    def run(self):

        tabs = st.tabs(["Load Data", "Visualize Data", "Classification", "Clustering", "Info"])
        
        with tabs[0]:
            self.load_data_tab()

        with tabs[1]:
            self.visualize_data_tab()

        with tabs[2]:
            self.classification_tab()

        with tabs[3]:
            self.clustering_tab()

        with tabs[4]:
            self.info_tab()


    def load_data_tab(self):
        st.header("Load Data")
        uploaded_file = self.data_uploader.upload_data()
        
        if uploaded_file is not None:
            self.df = self.data_uploader.read_data(uploaded_file)
            if self.df is not None:
                self.data_frame_viewer.view_data(self.df)
                self.data_frame_viewer.view_table_info(self.df)

    def visualize_data_tab(self):
        if self.df is not None:
            visualizer = DataVisualizer(self.df)
            st.subheader("PCA Visualization")
            visualizer.pca_reduction()
            st.subheader("t-SNE Visualization")
            visualizer.tsne_reduction()
            st.subheader("Exploratory Data Analysis (EDA) Charts")
            visualizer.eda_charts()
        else:
            st.write("Load Data first using the first tab!")

    def classification_tab(self):
        if self.df is not None:
            classifier = Classifier(self.df)
            classifier.compare_classifiers()
        else:
            st.write("Please load data first!")

    def clustering_tab(self):
        if self.df is not None:
            clustering = Clusterer(self.df)
            clustering.compare_clustering()
        else:
            st.write("Please load data first!")

    def info_tab(self):
        info = InfoPresenter()
        info.show_info()


if __name__ == "__main__":
    app = MainApplication()
    app.run()