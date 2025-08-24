import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="E-commerce Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #b8d4e8;
    }
</style>
""", unsafe_allow_html=True)

# ADD THE MISSING CLASS DEFINITION HERE
class AdaptiveNeighborhoodCF:
    def __init__(self, n_neighbors=50):
        self.n_neighbors = n_neighbors
        self.similarity_matrix = None
        self.train_matrix = None
        self.approach = None

    def fit(self, train_matrix):
        self.train_matrix = train_matrix

        # Decide whether to use user-based or item-based approach
        n_users, n_items = train_matrix.shape

        if n_users <= n_items:
            self.approach = "user-based"
            self._fit_user_based()
        else:
            self.approach = "item-based"
            self._fit_item_based()

    def _fit_user_based(self):
        """Fit user-based collaborative filtering"""
        # Use only active users to reduce computation
        user_activity = (self.train_matrix > 0).sum(axis=1)
        top_users = user_activity.nlargest(min(500, len(user_activity))).index
        subset_matrix = self.train_matrix.loc[top_users]

        # Calculate user similarity
        user_sim = cosine_similarity(subset_matrix.values)
        np.fill_diagonal(user_sim, 0)

        self.similarity_matrix = pd.DataFrame(
            user_sim, index=subset_matrix.index, columns=subset_matrix.index
        )

    def _fit_item_based(self):
        """Fit item-based collaborative filtering"""
        # Use only popular items to reduce computation
        item_activity = (self.train_matrix > 0).sum(axis=0)
        top_items = item_activity.nlargest(min(800, len(item_activity))).index
        subset_matrix = self.train_matrix[top_items].T

        # Calculate item similarity
        item_sim = cosine_similarity(subset_matrix.values)
        np.fill_diagonal(item_sim, 0)

        self.similarity_matrix = pd.DataFrame(
            item_sim, index=subset_matrix.index, columns=subset_matrix.index
        )

    def recommend(self, user_id, n_recommendations=10):
        if user_id not in self.train_matrix.index:
            return []

        if self.approach == "user-based":
            return self._recommend_user_based(user_id, n_recommendations)
        else:
            return self._recommend_item_based(user_id, n_recommendations)

    def _recommend_user_based(self, user_id, n_recommendations):
        if user_id not in self.similarity_matrix.index:
            return []

        # Find similar users
        similar_users = self.similarity_matrix.loc[user_id].sort_values(ascending=False)
        user_items = set(self.train_matrix.loc[user_id][self.train_matrix.loc[user_id] > 0].index)

        recommendations = {}
        for similar_user, similarity in similar_users.head(self.n_neighbors).items():
            if similarity > 0.1:  # Minimum similarity threshold
                similar_user_items = self.train_matrix.loc[similar_user]
                for item_id, rating in similar_user_items[similar_user_items > 0].items():
                    if item_id not in user_items:
                        recommendations[item_id] = recommendations.get(item_id, 0) + similarity * rating

        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_recs[:n_recommendations]]

    def _recommend_item_based(self, user_id, n_recommendations):
        user_ratings = self.train_matrix.loc[user_id]
        user_items = user_ratings[user_ratings > 0]

        recommendations = {}
        for item_id, rating in user_items.items():
            if item_id in self.similarity_matrix.index:
                similar_items = self.similarity_matrix.loc[item_id].sort_values(ascending=False)
                for similar_item, similarity in similar_items.head(self.n_neighbors).items():
                    if similar_item not in user_items.index and similarity > 0.1:
                        recommendations[similar_item] = recommendations.get(similar_item, 0) + similarity * rating

        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_recs[:n_recommendations]]

class RecommendationApp:
    def __init__(self):
        self.model = None
        self.user_item_matrix = None
        self.item_metrics = None
        self.popular_items = None
        self.model_results = None
        
    def load_data(self):
        """Load the saved models and data"""
        try:
            # Load the best model (Adaptive Neighborhood)
            self.model = joblib.load('/Users/paulaobeng-bioh/Desktop/e-commerce-recommendaation-system-modeling/recommendation_model_1_neighborhood.pkl')
            
            # Load Streamlit data with error handling
            streamlit_data = joblib.load('/Users/paulaobeng-bioh/Desktop/e-commerce-recommendaation-system-modeling/streamlit_data.pkl')
            
            # Handle missing keys gracefully
            self.user_item_matrix = streamlit_data.get('user_item_matrix')
            self.item_metrics = streamlit_data.get('item_metrics')
            self.popular_items = streamlit_data.get('popular_items', [])
            self.model_results = streamlit_data.get('model_results')
            
            # Check if essential data is loaded
            if self.model is None or self.user_item_matrix is None:
                return False
                
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def display_metrics(self):
        """Display key metrics about the recommendation system"""
        if self.user_item_matrix is None:
            st.warning("User-item matrix data not available")
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", f"{self.user_item_matrix.shape[0]:,}")
        with col2:
            st.metric("Total Items", f"{self.user_item_matrix.shape[1]:,}")
        with col3:
            sparsity = (1 - (self.user_item_matrix > 0).sum().sum() / 
                       (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1])) * 100
            st.metric("Data Sparsity", f"{sparsity:.1f}%")
        with col4:
            avg_interactions = (self.user_item_matrix > 0).sum().sum() / self.user_item_matrix.shape[0]
            st.metric("Avg Interactions/User", f"{avg_interactions:.1f}")
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations for a specific user"""
        try:
            if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
                return [], "User not found in training data"
            
            recommendations = self.model.recommend(user_id, n_recommendations)
            return recommendations, "Success"
        except Exception as e:
            return [], f"Error generating recommendations: {e}"
    
    def get_popular_recommendations(self, n_recommendations=10):
        """Get popular item recommendations"""
        if not self.popular_items:
            # Fallback popular items if not available
            return [119736, 461686, 9877, 241555, 369158][:n_recommendations]
        return self.popular_items[:n_recommendations]
    
    def display_item_info(self, item_id):
        """Display information about a specific item"""
        if self.item_metrics is not None and item_id in self.item_metrics.index:
            item_info = self.item_metrics.loc[item_id]
            st.write(f"**Item {item_id} Statistics:**")
            st.write(f"- Total Views: {int(item_info.get('views', 0))}")
            st.write(f"- Total Purchases: {int(item_info.get('purchases', 0))}")
            st.write(f"- Conversion Rate: {item_info.get('overall_conversion', 0):.2%}")
        else:
            st.write("Item information not available")
    
    def run(self):
        """Main application runner"""
        st.title("🛒 E-commerce Recommendation System")
        st.markdown("### Powered by Adaptive Neighborhood Collaborative Filtering")
        
        # Load data
        if not self.load_data():
            st.error("Failed to load recommendation data. Please check the file paths.")
            st.info("Make sure these files exist in the same directory:")
            st.write("- recommendation_model_1_neighborhood.pkl")
            st.write("- streamlit_data.pkl")
            return
        
        # Display metrics
        self.display_metrics()
        
        st.divider()
        
        # Recommendation section
        st.subheader("🎯 Get Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # User input section
            user_input = st.text_input("Enter User ID:", value="153600")
            n_recommendations = st.slider("Number of Recommendations:", 1, 20, 10)
            
            if st.button("Get Recommendations", type="primary"):
                if user_input.strip():
                    try:
                        user_id = int(user_input)
                        recommendations, status = self.get_user_recommendations(user_id, n_recommendations)
                        
                        if recommendations:
                            st.success(f"✅ Found {len(recommendations)} recommendations for User {user_id}")
                            
                            # Display recommendations
                            for i, item_id in enumerate(recommendations, 1):
                                with st.expander(f"Recommendation #{i}: Item {item_id}"):
                                    self.display_item_info(item_id)
                        else:
                            st.warning(f"⚠️ {status}. Showing popular items instead:")
                            popular_recs = self.get_popular_recommendations(n_recommendations)
                            for i, item_id in enumerate(popular_recs, 1):
                                with st.expander(f"Popular Item #{i}: Item {item_id}"):
                                    self.display_item_info(item_id)
                    except ValueError:
                        st.error("Please enter a valid numeric User ID")
                else:
                    st.error("Please enter a User ID")
        
        with col2:
            # Popular items section
            st.subheader("🔥 Popular Items")
            popular_recs = self.get_popular_recommendations(5)
            for item_id in popular_recs:
                st.write(f"• Item {item_id}")
        
        st.divider()
        
        # Data insights section
        st.subheader("📈 Data Insights")
        
        tab1, tab2, tab3 = st.tabs(["User Activity", "Item Performance", "Model Details"])
        
        with tab1:
            if self.user_item_matrix is not None:
                st.write("**User Interaction Statistics:**")
                user_activity = (self.user_item_matrix > 0).sum(axis=1)
                
                # Get actual statistics
                most_active_user = user_activity.idxmax()
                max_interactions = user_activity.max()
                avg_interactions = user_activity.mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Most Active User", f"{most_active_user} ({max_interactions:.0f} interactions)")
                with col2:
                    st.metric("Average Interactions", f"{avg_interactions:.1f}")
                with col3:
                    st.metric("Total Users", f"{len(user_activity):,}")
                
                # Create meaningful interaction ranges
                interaction_ranges = [
                    '1', '2-5', '6-10', '11-20', '21-50', '51-100', '100+'
                ]
                
                range_counts = [
                    (user_activity == 1).sum(),
                    ((user_activity >= 2) & (user_activity <= 5)).sum(),
                    ((user_activity >= 6) & (user_activity <= 10)).sum(),
                    ((user_activity >= 11) & (user_activity <= 20)).sum(),
                    ((user_activity >= 21) & (user_activity <= 50)).sum(),
                    ((user_activity >= 51) & (user_activity <= 100)).sum(),
                    (user_activity > 100).sum()
                ]
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(interaction_ranges, range_counts, color='skyblue', alpha=0.8)
                
                ax.set_title('User Engagement Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Number of Interactions')
                ax.set_ylabel('Number of Users')
                
                # Add value labels on bars
                for bar, count in zip(bars, range_counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                    
            else:
                st.warning("User activity data not available")


        with tab2:
            if self.item_metrics is not None:
                st.write("**Top Performing Items:**")
                top_items = self.item_metrics.nlargest(10, 'views')[['views', 'purchases', 'overall_conversion']]
                st.dataframe(top_items.style.format({
                    'views': '{:,.0f}',
                    'purchases': '{:,.0f}',
                    'overall_conversion': '{:.2%}'
                }), use_container_width=True)
            else:
                st.warning("Item performance data not available")
        
        with tab3:
            st.write("**Model Information:**")
            st.write("**Algorithm:** Adaptive Neighborhood Collaborative Filtering")
            st.write("**Approach:** User-based collaborative filtering with cosine similarity")
            st.write("**Neighbors:** 30 most similar users")
            st.write("**Strengths:**")
            st.write("- Handles cold start problems well")
            st.write("- Adapts to user behavior patterns")
            st.write("- Works well with sparse data")
        
        # Footer
        st.divider()
        st.caption("Built with ❤️ using Streamlit | Recommendation System Analysis & Modeling")

# Run the app
if __name__ == "__main__":
    app = RecommendationApp()
    app.run()