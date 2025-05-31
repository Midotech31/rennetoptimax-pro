# RennetOptiMax_Pro_Enhanced.py - Comprehensive Protein Expression Optimization Platform
# -------------------------------------------------------------------
# Run with: streamlit run RennetOptiMax_Pro_Enhanced.py
# Requirements: pip install streamlit pandas numpy scikit-learn plotly joblib biopython streamlit-authenticator

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import json
import random
import base64
from datetime import datetime, timedelta
import joblib
import io
import re
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import hashlib
import uuid

# Set page configuration
st.set_page_config(
    page_title="RennetOptiMax Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories for saved data if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('users', exist_ok=True)

# ----------------------
# Authentication Configuration
# ----------------------

def create_default_auth_config():
    """Create default authentication configuration"""
    config = {
        'credentials': {
            'usernames': {
                'admin': {
                    'email': 'admin@rennetoptimax.com',
                    'name': 'Administrator',
                    'password': '$2b$12$XcUMHB.QGJ8MKY.Hf9m8Yung7cHJ3ZlY1J9oX.zVm4N2aX8hJ0nCu',  # hashed: admin123
                    'user_type': 'admin',
                    'subscription_status': 'lifetime',
                    'subscription_expiry': None,
                    'referral_code': 'ADMIN001',
                    'referrals_made': 0,
                    'free_access_features': ['all'],
                    'trial_used': False,
                    'trial_expiry': None
                },
                'demo_student': {
                    'email': 'student@example.com',
                    'name': 'Demo Student',
                    'password': '$2b$12$XcUMHB.QGJ8MKY.Hf9m8Yung7cHJ3ZlY1J9oX.zVm4N2aX8hJ0nCu',  # hashed: student123
                    'user_type': 'student',
                    'subscription_status': 'trial',
                    'subscription_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
                    'referral_code': 'STU001',
                    'referrals_made': 0,
                    'free_access_features': ['vectors', 'hosts'],
                    'trial_used': False,
                    'trial_expiry': (datetime.now() + timedelta(days=30)).isoformat()
                }
            }
        },
        'cookie': {
            'name': 'rennet_auth_cookie',
            'key': 'random_signature_key_123456789',
            'expiry_days': 30
        },
        'preauthorized': {
            'emails': []
        }
    }
    return config

def load_auth_config():
    """Load authentication configuration"""
    config_file = 'data/auth_config.yaml'
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = yaml.load(file, Loader=SafeLoader)
    else:
        config = create_default_auth_config()
        save_auth_config(config)
    return config

def save_auth_config(config):
    """Save authentication configuration"""
    with open('data/auth_config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def generate_referral_code():
    """Generate a unique referral code"""
    return 'REF' + str(uuid.uuid4()).upper()[:6]

def add_new_user(username, email, name, password, user_type='professional'):
    """Add a new user to the authentication system"""
    config = load_auth_config()
    
    # Hash the password
    hashed_password = stauth.Hasher([password]).generate()[0]
    
    # Create new user data
    new_user = {
        'email': email,
        'name': name,
        'password': hashed_password,
        'user_type': user_type,
        'subscription_status': 'trial',
        'subscription_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
        'referral_code': generate_referral_code(),
        'referrals_made': 0,
        'free_access_features': ['vectors', 'hosts'],
        'trial_used': False,
        'trial_expiry': (datetime.now() + timedelta(days=30)).isoformat()
    }
    
    config['credentials']['usernames'][username] = new_user
    save_auth_config(config)
    return True

def check_user_access(username, feature):
    """Check if user has access to a specific feature"""
    config = load_auth_config()
    user_data = config['credentials']['usernames'].get(username, {})
    
    # Admin has access to everything
    if user_data.get('user_type') == 'admin':
        return True
    
    # Check subscription status
    subscription_status = user_data.get('subscription_status', 'none')
    
    if subscription_status == 'lifetime':
        return True
    
    if subscription_status == 'active':
        expiry = user_data.get('subscription_expiry')
        if expiry and datetime.fromisoformat(expiry) > datetime.now():
            return True
    
    if subscription_status == 'trial':
        trial_expiry = user_data.get('trial_expiry')
        if trial_expiry and datetime.fromisoformat(trial_expiry) > datetime.now():
            return True
    
    # Check free access features
    free_features = user_data.get('free_access_features', [])
    if feature in free_features or 'all' in free_features:
        return True
    
    return False

# ----------------------
# Pricing and Plans Configuration
# ----------------------

PRICING_PLANS = {
    'student': {
        '1_month': 5,
        '6_months': 25,
        '1_year': 40
    },
    'academic': {
        '1_month': 7,
        '6_months': 35,
        '1_year': 60
    },
    'professional': {
        '1_month': 10,
        '6_months': 50,
        '1_year': 85
    }
}

# ----------------------
# Data Classes & Models (Previous code remains the same)
# ----------------------

class Vector:
    """Expression vector class"""
    def __init__(self, id, name, size, promoter, terminator, origin, selection_marker, tags, description, features):
        self.id = id
        self.name = name
        self.size = size
        self.promoter = promoter
        self.terminator = terminator
        self.origin = origin
        self.selection_marker = selection_marker
        self.tags = tags
        self.description = description
        self.features = features
        
    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'size': self.size, 'promoter': self.promoter,
            'terminator': self.terminator, 'origin': self.origin, 'selection_marker': self.selection_marker,
            'tags': self.tags, 'description': self.description, 'features': self.features
        }

class Host:
    """Host strain class"""
    def __init__(self, id, strain, species, genotype, description, features, limitations):
        self.id = id
        self.strain = strain
        self.species = species
        self.genotype = genotype
        self.description = description
        self.features = features
        self.limitations = limitations
        
    def to_dict(self):
        return {
            'id': self.id, 'strain': self.strain, 'species': self.species, 'genotype': self.genotype,
            'description': self.description, 'features': self.features, 'limitations': self.limitations
        }

class ExpressionOptimizer:
    """Class for ML-based expression optimization"""
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.numeric_features = ['temperature', 'induction_time', 'inducer_concentration', 'OD600_at_induction']
        self.categorical_features = ['vector_type', 'host_strain', 'media_composition']
        self.model_path = 'models/rennet_model.joblib'
        
    def generate_sample_data(self, n_samples=100):
        """Generate synthetic training data for the model"""
        np.random.seed(42)
        
        vectors = ['pET21a', 'pET28a', 'pET22b', 'pBAD', 'pUC19']
        hosts = ['BL21(DE3)', 'Rosetta(DE3)', 'C41(DE3)', 'BL21(DE3)pLysS', 'DH5α']
        media = ['LB', 'TB', 'M9', 'SOC', '2xYT']
        
        data = {
            'vector_type': np.random.choice(vectors, n_samples),
            'host_strain': np.random.choice(hosts, n_samples),
            'temperature': np.random.choice([16, 25, 30, 37], n_samples),
            'induction_time': np.random.choice([2, 3, 4, 5, 6], n_samples),
            'inducer_concentration': np.random.uniform(0.1, 1.0, n_samples).round(2),
            'OD600_at_induction': np.random.uniform(0.4, 0.8, n_samples).round(2),
            'media_composition': np.random.choice(media, n_samples)
        }
        
        # Create synthetic expression levels
        vector_weights = {'pET21a': 0.9, 'pET28a': 0.8, 'pET22b': 0.7, 'pBAD': 0.5, 'pUC19': 0.3}
        vector_effect = np.array([vector_weights[v] for v in data['vector_type']])
        
        host_weights = {'BL21(DE3)': 1.0, 'Rosetta(DE3)': 0.9, 'C41(DE3)': 0.8, 'BL21(DE3)pLysS': 0.7, 'DH5α': 0.5}
        host_effect = np.array([host_weights[h] for h in data['host_strain']])
        
        media_weights = {'LB': 0.7, 'TB': 1.0, 'M9': 0.5, 'SOC': 0.6, '2xYT': 0.8}
        media_effect = np.array([media_weights[m] for m in data['media_composition']])
        
        temp_effect = 1.0 - 0.1 * abs(np.array(data['temperature']) - 30) / 10
        
        expression = (
            0.35 * vector_effect + 0.25 * host_effect + 0.20 * media_effect +
            0.10 * temp_effect + 0.05 * data['induction_time'] / 6 +
            0.05 * data['inducer_concentration'] + np.random.normal(0, 0.1, n_samples)
        )
        
        min_expr = min(expression)
        max_expr = max(expression)
        expression = 100 * (expression - min_expr) / (max_expr - min_expr)
        data['expression_level'] = expression.round(2)
        
        return pd.DataFrame(data)
    
    def train_model(self, data=None):
        """Train a machine learning model on expression data"""
        if data is None:
            data = self.generate_sample_data()
            
        X = data.drop('expression_level', axis=1)
        y = data['expression_level']
        
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        self.save_model()
        
        return {'train_score': train_score, 'test_score': test_score, 'model': self.model}
    
    def save_model(self):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'features': {'numeric': self.numeric_features, 'categorical': self.categorical_features}
        }
        joblib.dump(model_data, self.model_path)
    
    def load_model(self):
        """Load model from disk or train a new one if not found"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.preprocessor = model_data['preprocessor']
                return True
            else:
                self.train_model()
                return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.train_model()
            return False
    
    def predict_expression(self, conditions):
        """Predict expression level for given conditions"""
        if self.model is None or self.preprocessor is None:
            self.load_model()
            
        try:
            input_df = pd.DataFrame([conditions])
            X_input = self.preprocessor.transform(input_df)
            prediction = self.model.predict(X_input)[0]
            return prediction
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return 50.0

# ----------------------
# Database Functions (Previous code remains the same but shortened for space)
# ----------------------

def load_vectors():
    """Load vector database or create default vectors if not exists"""
    vector_file = 'data/vectors.json'
    
    if os.path.exists(vector_file):
        with open(vector_file, 'r') as f:
            vector_data = json.load(f)
            vectors = [Vector(**v) for v in vector_data]
    else:
        # Create default vectors (shortened for space)
        vectors = [
            Vector(1, "pET21a", 5443, "T7", "T7", "pBR322", "Ampicillin", ["His-tag", "C-terminal"],
                   "High-level expression vector with C-terminal His-tag",
                   {"cloning_sites": ["NdeI", "XhoI", "BamHI", "EcoRI"], "tag_location": "C-terminal", "induction": "IPTG"}),
            Vector(2, "pET28a", 5369, "T7", "T7", "pBR322", "Kanamycin", ["His-tag", "N-terminal", "T7-tag"],
                   "High-level expression vector with N-terminal His-tag",
                   {"cloning_sites": ["NdeI", "XhoI", "BamHI", "EcoRI"], "tag_location": "N-terminal", "induction": "IPTG"})
        ]
        save_vectors(vectors)
        
    return vectors

def save_vectors(vectors):
    """Save vectors to JSON file"""
    vector_data = [v.to_dict() for v in vectors]
    with open('data/vectors.json', 'w') as f:
        json.dump(vector_data, f, indent=2)

def load_hosts():
    """Load host database or create default hosts if not exists"""
    host_file = 'data/hosts.json'
    
    if os.path.exists(host_file):
        with open(host_file, 'r') as f:
            host_data = json.load(f)
            hosts = [Host(**h) for h in host_data]
    else:
        # Create default hosts (shortened for space)
        hosts = [
            Host(1, "BL21(DE3)", "E. coli", "F– ompT gal dcm lon hsdSB(rB–mB–) λ(DE3 [lacI lacUV5-T7p07 ind1 sam7 nin5]) [malB+]K-12(λS)",
                 "Standard expression strain with T7 RNA polymerase", ["T7 expression", "Protease deficient", "General purpose"],
                 ["Not suitable for toxic proteins", "No rare codon support"]),
            Host(2, "Rosetta(DE3)", "E. coli", "F- ompT hsdSB(rB- mB-) gal dcm (DE3) pRARE (CamR)",
                 "Enhanced expression of proteins containing rare codons", ["T7 expression", "Rare codon optimization", "Protease deficient"],
                 ["Additional antibiotic (chloramphenicol) required"])
        ]
        save_hosts(hosts)
        
    return hosts

def save_hosts(hosts):
    """Save hosts to JSON file"""
    host_data = [h.to_dict() for h in hosts]
    with open('data/hosts.json', 'w') as f:
        json.dump(host_data, f, indent=2)

# ----------------------
# UI Components
# ----------------------

def init_session_state():
    """Initialize session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'selected_vector' not in st.session_state:
        st.session_state.selected_vector = None
    if 'selected_host' not in st.session_state:
        st.session_state.selected_host = None
    if 'expression_parameters' not in st.session_state:
        st.session_state.expression_parameters = {
            'temperature': 30, 'induction_time': 4, 'inducer_concentration': 0.5,
            'OD600_at_induction': 0.6, 'media_composition': 'LB'
        }
    if 'protein_sequence' not in st.session_state:
        st.session_state.protein_sequence = ""
    if 'sequence_analysis' not in st.session_state:
        st.session_state.sequence_analysis = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None

def show_header():
    """Display the application header"""
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/test-tube.png", width=80)
    
    with col2:
        st.title("RennetOptiMax Pro")
        st.markdown("### 🧬 AI-Powered Protein Expression Optimization Platform")
    
    with col3:
        if st.session_state.get("authentication_status"):
            if st.button("🏠 Dashboard", key="dashboard_btn"):
                st.session_state.page = 'dashboard'
                st.experimental_rerun()

def show_product_banner():
    """Display product promotion banner"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 30px; 
                border-radius: 10px; 
                margin: 20px 0; 
                text-align: center;">
        <h2>🎯 NeoRen Chymosin Powder</h2>
        <h3>Premium Sustainable Rennet for Modern Cheese Production</h3>
        <p>✅ 100% Animal-Free • ✅ Superior Performance • ✅ Cost-Effective & Scalable</p>
        <div style="margin: 20px 0;">
            <button style="background: #ff6b6b; 
                          color: white; 
                          border: none; 
                          padding: 15px 30px; 
                          border-radius: 5px; 
                          font-size: 18px; 
                          cursor: pointer;">
                🛒 Buy 500g & Get 1 Year Free Access
            </button>
        </div>
        <p><small>Sustainable rennet production through advanced genetic engineering</small></p>
    </div>
    """, unsafe_allow_html=True)

def show_access_plans():
    """Display access plans and pricing"""
    st.markdown("## 🎁 Unlock Full Power of RennetOptiMax Pro")
    st.markdown("*Choose the access method that fits you best:*")
    
    # Free Access
    with st.expander("🆓 Free Access for All", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ 1-month free trial** with all features")
            st.markdown("**✅ Lifetime access** to 2 core features")
        with col2:
            st.markdown("- Vector Selection Tool")
            st.markdown("- Host Strain Database")
    
    # Referral Rewards
    with st.expander("🔗 Referral Reward Program"):
        st.markdown("**Share your referral link and earn rewards!**")
        if st.session_state.get("authentication_status"):
            config = load_auth_config()
            username = st.session_state["username"]
            referral_code = config['credentials']['usernames'][username]['referral_code']
            st.code(f"Your Referral Code: {referral_code}")
            st.markdown("**🎁 If 1 person buys via your link → 6 months free access**")
        else:
            st.info("Login to get your personal referral code!")
    
    # Product Purchase Bonus
    with st.expander("🛒 Product Purchase Bonus"):
        st.markdown("**Buy 500g of NeoRen Chymosin Powder → Get 1 year full access instantly**")
        st.markdown("- Premium sustainable rennet")
        st.markdown("- Immediate platform access")
        st.markdown("- Technical support included")
    
    # Subscription Plans
    with st.expander("💳 Subscription Plans (Post-Trial)"):
        st.markdown("**Affordable pricing tailored to different user profiles:**")
        
        pricing_df = pd.DataFrame({
            'User Type': ['🎓 Student', '🧪 Academic', '🧑‍💼 Professional'],
            '1 Month': ['$5', '$7', '$10'],
            '6 Months': ['$25', '$35', '$50'],
            '1 Year': ['$40', '$60', '$85']
        })
        
        st.table(pricing_df)
        st.markdown("**✅ All subscriptions include:** Full access, technical support, and updates")
    
    # Loyalty Program
    with st.expander("💎 Loyalty Program"):
        st.markdown("**Long-term subscribers and highly active users may qualify for:**")
        st.markdown("🌟 **Lifetime unlimited access** — at no extra cost")
        st.markdown("- Based on usage patterns and subscription history")
        st.markdown("- Exclusive benefits for power users")

def show_signup_form():
    """Display user registration form"""
    st.markdown("## 📝 Create Your Account")
    
    with st.form("signup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username*")
            email = st.text_input("Email*")
            name = st.text_input("Full Name*")
        
        with col2:
            user_type = st.selectbox("User Type", ["student", "academic", "professional"])
            password = st.text_input("Password*", type="password")
            confirm_password = st.text_input("Confirm Password*", type="password")
        
        referral_code = st.text_input("Referral Code (Optional)", help="Enter a referral code if you have one")
        
        terms_accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy*")
        
        submitted = st.form_submit_button("🚀 Create Account & Start Free Trial")
        
        if submitted:
            if not all([username, email, name, password]):
                st.error("Please fill in all required fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif not terms_accepted:
                st.error("Please accept the Terms of Service.")
            else:
                try:
                    if add_new_user(username, email, name, password, user_type):
                        st.success("Account created successfully! Please login to continue.")
                        st.balloons()
                        if st.button("Go to Login"):
                            st.session_state.show_signup = False
                            st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error creating account: {e}")

def show_user_dashboard():
    """Display user dashboard"""
    if not st.session_state.get("authentication_status"):
        st.error("Please login to access the dashboard.")
        return
    
    config = load_auth_config()
    username = st.session_state["username"]
    user_data = config['credentials']['usernames'][username]
    
    st.markdown(f"## 👋 Welcome, {user_data['name']}!")
    
    # Account Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Account Type", user_data['user_type'].title())
    
    with col2:
        status = user_data['subscription_status']
        st.metric("📊 Status", status.title())
    
    with col3:
        referrals = user_data.get('referrals_made', 0)
        st.metric("🔗 Referrals Made", referrals)
    
    with col4:
        if user_data['subscription_expiry']:
            expiry = datetime.fromisoformat(user_data['subscription_expiry'])
            days_left = (expiry - datetime.now()).days
            st.metric("⏰ Days Left", max(0, days_left))
        else:
            st.metric("⏰ Access", "Lifetime")
    
    # Referral Section
    st.markdown("### 🔗 Your Referral Program")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        referral_code = user_data['referral_code']
        st.code(f"Your Referral Code: {referral_code}")
        st.markdown("Share this code to earn 6 months free access when someone purchases our product!")
    
    with col2:
        if st.button("📧 Share via Email"):
            st.info("Email sharing functionality would be implemented here")
        if st.button("📱 Share on Social"):
            st.info("Social sharing functionality would be implemented here")
    
    # Subscription Management
    st.markdown("### 💳 Subscription Management")
    
    if user_data['subscription_status'] == 'trial':
        st.info("You're currently on a free trial. Upgrade to continue using all features after trial expires.")
        
        user_type = st.selectbox("Select Plan Type", ["student", "academic", "professional"], 
                                index=["student", "academic", "professional"].index(user_data['user_type']))
        
        plan_duration = st.selectbox("Select Duration", ["1_month", "6_months", "1_year"])
        
        price = PRICING_PLANS[user_type][plan_duration]
        st.markdown(f"**Price: ${price}**")
        
        if st.button("💳 Upgrade Subscription"):
            st.success("Subscription upgrade functionality would be implemented here with payment gateway integration.")
    
    elif user_data['subscription_status'] == 'active':
        st.success("Your subscription is active!")
        if st.button("🔄 Renew Subscription"):
            st.info("Subscription renewal functionality would be implemented here.")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧬 Start Optimization", use_container_width=True):
            st.session_state.page = 'vectors'
            st.experimental_rerun()
    
    with col2:
        if st.button("📊 View Results", use_container_width=True):
            st.session_state.page = 'results'
            st.experimental_rerun()
    
    with col3:
        if st.button("❓ Get Support", use_container_width=True):
            st.info("Support functionality would be implemented here.")

def show_navigation():
    """Display navigation sidebar"""
    st.sidebar.title("🧬 RennetOptiMax Pro")
    
    # Authentication status
    if st.session_state.get("authentication_status"):
        config = load_auth_config()
        username = st.session_state["username"]
        user_data = config['credentials']['usernames'][username]
        
        st.sidebar.success(f"👋 Welcome, {user_data['name']}")
        
        # Navigation based on access level
        pages = {
            'dashboard': "🏠 Dashboard",
            'home': "🌟 Home"
        }
        
        # Check access for each feature
        if check_user_access(username, 'vectors') or check_user_access(username, 'all'):
            pages['vectors'] = "1. 🧬 Select Vector"
        
        if check_user_access(username, 'hosts') or check_user_access(username, 'all'):
            pages['hosts'] = "2. 🦠 Select Host"
        
        if check_user_access(username, 'sequence') or check_user_access(username, 'all'):
            pages['sequence'] = "3. 🔬 Analyze Sequence"
        
        if check_user_access(username, 'parameters') or check_user_access(username, 'all'):
            pages['parameters'] = "4. ⚙️ Set Parameters"
        
        if check_user_access(username, 'optimize') or check_user_access(username, 'all'):
            pages['optimize'] = "5. 🎯 Optimize Expression"
        
        if check_user_access(username, 'results') or check_user_access(username, 'all'):
            pages['results'] = "6. 📊 View Results"
        
        # Display navigation buttons
        for page_id, page_name in pages.items():
            if st.sidebar.button(page_name, key=f"nav_{page_id}", use_container_width=True,
                                type="primary" if st.session_state.page == page_id else "secondary"):
                st.session_state.page = page_id
                st.experimental_rerun()
        
        # Account management
        st.sidebar.divider()
        st.sidebar.subheader("Account")
        
        if st.sidebar.button("⚙️ Account Settings", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.experimental_rerun()
        
        # Logout
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith(('authentication', 'name', 'username')):
                    del st.session_state[key]
            st.experimental_rerun()
    
    else:
        # Not authenticated - show only home
        if st.sidebar.button("🌟 Home", use_container_width=True):
            st.session_state.page = 'home'
            st.experimental_rerun()
    
    # About section
    st.sidebar.divider()
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "RennetOptiMax Pro: AI-powered protein expression optimization "
        "for sustainable rennet production."
    )
    st.sidebar.caption("Version 2.0.0 | © 2025 NeoRen")

def show_home_page():
    """Display enhanced home page"""
    st.markdown("## 🌟 Welcome to RennetOptiMax Pro")
    
    # Product Banner
    show_product_banner()
    
    # Main Description
    st.markdown("""
    ### 🧬 AI-Powered Protein Expression Optimization Platform
    
    This cutting-edge platform revolutionizes recombinant protein expression optimization, 
    specifically designed for **sustainable rennet (chymosin) production** using advanced AI and machine learning.
    
    **🔬 Why Choose RennetOptiMax Pro?**
    - **AI-Driven Optimization**: Machine learning algorithms predict optimal expression conditions
    - **Comprehensive Database**: Extensive collection of vectors, hosts, and protocols
    - **Real-Time Analysis**: Instant protein sequence analysis and recommendations
    - **Cost-Effective**: Reduce experimental costs by 60% through predictive modeling
    - **Sustainable Focus**: Supporting eco-friendly biotechnology solutions
    """)
    
    # Access Plans Section
    show_access_plans()
    
    # Features Overview
    st.markdown("### 🚀 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🧬 Vector Selection**
        - 50+ expression vectors
        - Advanced filtering system
        - Compatibility predictions
        - Performance analytics
        """)
    
    with col2:
        st.markdown("""
        **🦠 Host Optimization**
        - E. coli strain database
        - Specialized strain recommendations
        - Growth condition optimization
        - Yield predictions
        """)
    
    with col3:
        st.markdown("""
        **🎯 AI Predictions**
        - Expression level forecasting
        - Parameter optimization
        - Protocol generation
        - Success probability scoring
        """)
    
    # Call to Action
    st.markdown("### 🚀 Get Started Today!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🆓 Start Free Trial", use_container_width=True, type="primary"):
            st.session_state.show_signup = True
            st.experimental_rerun()
    
    with col2:
        if st.button("🔑 Login", use_container_width=True):
            st.session_state.show_login = True
            st.experimental_rerun()
    
    with col3:
        if st.button("🛒 Buy Product + Access", use_container_width=True):
            st.info("Redirecting to product purchase page...")

def show_vectors_page():
    """Display vector selection page with access control"""
    username = st.session_state.get("username")
    
    if not username or not check_user_access(username, 'vectors'):
        st.error("🔒 This feature requires a subscription. Please upgrade your account.")
        st.info("You can access this feature with any subscription plan or during your free trial.")
        return
    
    st.markdown("## 🧬 Expression Vector Selection")
    
    # Load vector database
    vectors = load_vectors()
    
    # Filter controls
    st.markdown("### 🔍 Filter Vectors")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        promoter_filter = st.selectbox("Promoter Type", ["All"] + sorted(set(v.promoter for v in vectors)))
    
    with col2:
        selection_filter = st.selectbox("Selection Marker", ["All"] + sorted(set(v.selection_marker for v in vectors)))
    
    with col3:
        all_tags = set()
        for v in vectors:
            all_tags.update(v.tags)
        tag_filter = st.selectbox("Tag Type", ["All"] + sorted(all_tags))
    
    # Apply filters
    filtered_vectors = vectors
    if promoter_filter != "All":
        filtered_vectors = [v for v in filtered_vectors if v.promoter == promoter_filter]
    if selection_filter != "All":
        filtered_vectors = [v for v in filtered_vectors if v.selection_marker == selection_filter]
    if tag_filter != "All":
        filtered_vectors = [v for v in filtered_vectors if tag_filter in v.tags]
    
    # Display vectors
    st.markdown("### 🧬 Available Vectors")
    
    if not filtered_vectors:
        st.warning("No vectors match the selected filters. Please adjust your criteria.")
    else:
        for i in range(0, len(filtered_vectors), 3):
            row_vectors = filtered_vectors[i:i+3]
            cols = st.columns(3)
            
            for j, vector in enumerate(row_vectors):
                with cols[j]:
                    selected = st.session_state.selected_vector and st.session_state.selected_vector.id == vector.id
                    
                    st.markdown(f"""
                    <div style="border: {'2px solid #1976d2' if selected else '1px solid #e0e0e0'}; 
                                border-radius: 10px; 
                                padding: 15px; 
                                height: 300px;
                                background-color: {'#f9f9f9' if selected else 'white'}">
                        <h4 style="color: {'#1976d2' if selected else 'black'}; margin-top: 0;">{vector.name}</h4>
                        <p><b>Size:</b> {vector.size} bp</p>
                        <p><b>Promoter:</b> {vector.promoter}</p>
                        <p><b>Selection:</b> {vector.selection_marker}</p>
                        <p><b>Tags:</b> {', '.join(vector.tags)}</p>
                        <p><b>Description:</b> {vector.description}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    btn_label = "✅ Selected" if selected else "Select Vector"
                    if st.button(btn_label, key=f"select_vector_{vector.id}",
                                use_container_width=True,
                                type="primary" if selected else "secondary"):
                        st.session_state.selected_vector = vector
                        st.experimental_rerun()

def show_hosts_page():
    """Display host selection page with access control"""
    username = st.session_state.get("username")
    
    if not username or not check_user_access(username, 'hosts'):
        st.error("🔒 This feature requires a subscription. Please upgrade your account.")
        st.info("You can access this feature with any subscription plan or during your free trial.")
        return
    
    st.markdown("## 🦠 Host Strain Selection")
    
    # Load host database
    hosts = load_hosts()
    
    # Similar implementation as vectors page...
    st.markdown("### 🦠 Available Host Strains")
    
    for i in range(0, len(hosts), 2):
        row_hosts = hosts[i:i+2]
        cols = st.columns(2)
        
        for j, host in enumerate(row_hosts):
            with cols[j]:
                selected = st.session_state.selected_host and st.session_state.selected_host.id == host.id
                
                st.markdown(f"""
                <div style="border: {'2px solid #1976d2' if selected else '1px solid #e0e0e0'}; 
                            border-radius: 10px; 
                            padding: 15px; 
                            background-color: {'#f9f9f9' if selected else 'white'}">
                    <h4 style="color: {'#1976d2' if selected else 'black'}; margin-top: 0;">{host.strain}</h4>
                    <p><b>Species:</b> {host.species}</p>
                    <p><b>Description:</b> {host.description}</p>
                    <p><b>Features:</b> {', '.join(host.features)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                btn_label = "✅ Selected" if selected else "Select Host"
                if st.button(btn_label, key=f"select_host_{host.id}",
                            use_container_width=True,
                            type="primary" if selected else "secondary"):
                    st.session_state.selected_host = host
                    st.experimental_rerun()

def show_restricted_feature(feature_name):
    """Show message for restricted features"""
    st.error(f"🔒 {feature_name} requires a subscription. Please upgrade your account.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Upgrade Now", use_container_width=True, type="primary"):
            st.session_state.page = 'dashboard'
            st.experimental_rerun()
    
    with col2:
        if st.button("🆓 Start Free Trial", use_container_width=True):
            st.session_state.show_signup = True
            st.experimental_rerun()

# ----------------------
# Main Application
# ----------------------

def main():
    # Page styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3, h4 {
        color: #1976d2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Load authentication config
    config = load_auth_config()
    
    # Create authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    
    # Show header
    show_header()
    
    # Handle authentication
    if not st.session_state.get("authentication_status"):
        # Show signup form if requested
        if st.session_state.get("show_signup", False):
            show_signup_form()
            if st.button("← Back to Home"):
                st.session_state.show_signup = False
                st.experimental_rerun()
            return
        
        # Show login form if requested
        if st.session_state.get("show_login", False):
            try:
                name, authentication_status, username = authenticator.login('Login', 'main')
                
                if authentication_status == False:
                    st.error('Username/password is incorrect')
                elif authentication_status == None:
                    st.warning('Please enter your username and password')
                
                if st.button("← Back to Home"):
                    st.session_state.show_login = False
                    st.experimental_rerun()
                
                return
                
            except Exception as e:
                st.error(f"Login error: {e}")
                return
        
        # Show home page for non-authenticated users
        show_navigation()
        show_home_page()
        return
    
    # User is authenticated
    try:
        name, authentication_status, username = authenticator.login('Login', 'sidebar')
    except:
        pass
    
    # Show navigation
    show_navigation()
    
    # Display the appropriate page based on session state
    current_page = st.session_state.page
    username = st.session_state.get("username")
    
    if current_page == 'home':
        show_home_page()
    elif current_page == 'dashboard':
        show_user_dashboard()
    elif current_page == 'vectors':
        show_vectors_page()
    elif current_page == 'hosts':
        show_hosts_page()
    elif current_page == 'sequence':
        if check_user_access(username, 'sequence'):
            st.markdown("## 🔬 Protein Sequence Analysis")
            st.info("Sequence analysis functionality will be implemented here.")
        else:
            show_restricted_feature("Sequence Analysis")
    elif current_page == 'parameters':
        if check_user_access(username, 'parameters'):
            st.markdown("## ⚙️ Expression Parameters Configuration")
            st.info("Parameters configuration functionality will be implemented here.")
        else:
            show_restricted_feature("Parameters Configuration")
    elif current_page == 'optimize':
        if check_user_access(username, 'optimize'):
            st.markdown("## 🎯 Expression Optimization")
            st.info("Optimization functionality will be implemented here.")
        else:
            show_restricted_feature("Expression Optimization")
    elif current_page == 'results':
        if check_user_access(username, 'results'):
            st.markdown("## 📊 Optimization Results")
            st.info("Results visualization functionality will be implemented here.")
        else:
            show_restricted_feature("Results Visualization")

# Run the application
if __name__ == "__main__":
    main()
