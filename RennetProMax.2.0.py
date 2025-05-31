# RennetOptiMax_Pro_Complete.py - Version Complète et Fonctionnelle
# -------------------------------------------------------------------
# Run with: streamlit run RennetOptiMax_Pro_Complete.py
# Requirements: pip install streamlit pandas numpy scikit-learn plotly joblib biopython

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
import os
import json
import random
import base64
from datetime import datetime, timedelta
import joblib
import io
import re
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

# NeoRen Product Link
NEOREN_WEBSITE = "https://neoren.mystrikingly.com/"

# ----------------------
# Authentication System
# ----------------------

def hash_password(password):
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """Verify a password against its hash"""
    return hash_password(password) == hashed_password

def create_default_users():
    """Create default user database"""
    users = {
        'admin': {
            'password': hash_password('admin123'),
            'name': 'Administrator',
            'email': 'admin@rennetoptimax.com',
            'user_type': 'admin',
            'subscription_status': 'lifetime',
            'subscription_expiry': None,
            'referral_code': 'ADMIN001',
            'trial_expiry': None
        },
        'demo_user': {
            'password': hash_password('demo123'),
            'name': 'Demo User',
            'email': 'demo@example.com',
            'user_type': 'professional',
            'subscription_status': 'trial',
            'subscription_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
            'referral_code': 'DEMO001',
            'trial_expiry': (datetime.now() + timedelta(days=30)).isoformat()
        }
    }
    return users

def load_users():
    """Load users from file or create default"""
    users_file = 'data/users.json'
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            return json.load(f)
    else:
        users = create_default_users()
        save_users(users)
        return users

def save_users(users):
    """Save users to file"""
    with open('data/users.json', 'w') as f:
        json.dump(users, f, indent=2)

def authenticate_user(username, password):
    """Authenticate a user"""
    users = load_users()
    if username in users:
        user_data = users[username]
        if verify_password(password, user_data['password']):
            return True, user_data
    return False, None

def check_user_access(username, feature):
    """Check if user has access to a specific feature"""
    users = load_users()
    user_data = users.get(username, {})
    
    # Admin has access to everything
    if user_data.get('user_type') == 'admin':
        return True
    
    # Check subscription status
    subscription_status = user_data.get('subscription_status', 'none')
    
    if subscription_status == 'lifetime':
        return True
    
    if subscription_status == 'trial':
        trial_expiry = user_data.get('trial_expiry')
        if trial_expiry and datetime.fromisoformat(trial_expiry) > datetime.now():
            return True
    
    # Free features
    free_features = ['vectors', 'hosts']
    if feature in free_features:
        return True
    
    return False

# ----------------------
# Data Classes & Models
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
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            self.load_model()
            
        importances = self.model.feature_importances_
        feature_names = self.numeric_features + [
            f"{cat}_{val}" for cat in self.categorical_features 
            for val in ['value1', 'value2', 'value3', 'value4', 'value5']
        ]
        
        # Trim to match importances length
        feature_names = feature_names[:len(importances)]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        return importance_df.sort_values('Importance', ascending=False)
    
    def suggest_optimal_conditions(self, vector_name, host_strain, protein_properties, n_suggestions=5):
        """Suggest optimal conditions based on protein properties"""
        if self.model is None:
            self.load_model()
            
        # Generate parameter combinations
        param_ranges = {
            'temperature': [16, 25, 30, 37],
            'induction_time': [2, 4, 6],
            'inducer_concentration': [0.1, 0.5, 1.0],
            'OD600_at_induction': [0.4, 0.6, 0.8],
            'media_composition': ['LB', 'TB', 'M9', '2xYT']
        }
        
        # Adjust based on protein properties
        if protein_properties.get('has_disulfide_bonds', False):
            param_ranges['temperature'] = [16, 25, 30]
            
        if protein_properties.get('is_membrane_protein', False):
            param_ranges['temperature'] = [16, 25]
            param_ranges['inducer_concentration'] = [0.1, 0.2, 0.5]
        
        # Generate combinations
        import itertools
        suggestions = []
        
        for temp in param_ranges['temperature'][:2]:  # Limit combinations
            for media in param_ranges['media_composition'][:2]:
                for inducer in param_ranges['inducer_concentration'][:2]:
                    params = {
                        'vector_type': vector_name,
                        'host_strain': host_strain,
                        'temperature': temp,
                        'induction_time': 4,
                        'inducer_concentration': inducer,
                        'OD600_at_induction': 0.6,
                        'media_composition': media
                    }
                    
                    expression = self.predict_expression(params)
                    
                    suggestions.append({
                        'parameters': params,
                        'predicted_expression': float(expression),
                        'additives': []
                    })
        
        # Sort by expression level
        suggestions.sort(key=lambda x: x['predicted_expression'], reverse=True)
        
        return suggestions[:n_suggestions]

# ----------------------
# Database Functions
# ----------------------

def load_vectors():
    """Load vector database or create default vectors if not exists"""
    vector_file = 'data/vectors.json'
    
    if os.path.exists(vector_file):
        with open(vector_file, 'r') as f:
            vector_data = json.load(f)
            vectors = [Vector(**v) for v in vector_data]
    else:
        vectors = [
            Vector(1, "pET21a", 5443, "T7", "T7", "pBR322", "Ampicillin", ["His-tag", "C-terminal"],
                   "High-level expression vector with C-terminal His-tag",
                   {"cloning_sites": ["NdeI", "XhoI", "BamHI", "EcoRI"], "tag_location": "C-terminal", "induction": "IPTG"}),
            Vector(2, "pET28a", 5369, "T7", "T7", "pBR322", "Kanamycin", ["His-tag", "N-terminal", "T7-tag"],
                   "High-level expression vector with N-terminal His-tag",
                   {"cloning_sites": ["NdeI", "XhoI", "BamHI", "EcoRI"], "tag_location": "N-terminal", "induction": "IPTG"}),
            Vector(3, "pET22b", 5493, "T7", "T7", "pBR322", "Ampicillin", ["His-tag", "C-terminal", "pelB"],
                   "Periplasmic expression vector with pelB signal sequence",
                   {"cloning_sites": ["NdeI", "XhoI", "BamHI", "EcoRI"], "tag_location": "C-terminal", "induction": "IPTG"}),
            Vector(4, "pBAD", 4102, "araBAD", "rrnB", "pBR322", "Ampicillin", ["His-tag", "C-terminal"],
                   "Arabinose-inducible expression vector",
                   {"cloning_sites": ["NcoI", "HindIII", "XhoI"], "tag_location": "C-terminal", "induction": "Arabinose"}),
            Vector(5, "pMAL-c5X", 5677, "tac", "lambda t0", "pBR322", "Ampicillin", ["MBP", "N-terminal"],
                   "MBP fusion vector for improved solubility",
                   {"cloning_sites": ["NdeI", "EcoRI", "BamHI", "SalI"], "tag_location": "N-terminal", "induction": "IPTG"}),
            Vector(6, "pGEX-6P-1", 4984, "tac", "lambda t0", "pBR322", "Ampicillin", ["GST", "N-terminal"],
                   "GST fusion vector for improved solubility",
                   {"cloning_sites": ["BamHI", "EcoRI", "SalI", "NotI"], "tag_location": "N-terminal", "induction": "IPTG"})
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
        hosts = [
            Host(1, "BL21(DE3)", "E. coli", "F– ompT gal dcm lon hsdSB(rB–mB–) λ(DE3 [lacI lacUV5-T7p07 ind1 sam7 nin5]) [malB+]K-12(λS)",
                 "Standard expression strain with T7 RNA polymerase", ["T7 expression", "Protease deficient", "General purpose"],
                 ["Not suitable for toxic proteins", "No rare codon support"]),
            Host(2, "Rosetta(DE3)", "E. coli", "F- ompT hsdSB(rB- mB-) gal dcm (DE3) pRARE (CamR)",
                 "Enhanced expression of proteins containing rare codons", ["T7 expression", "Rare codon optimization", "Protease deficient"],
                 ["Additional antibiotic (chloramphenicol) required"]),
            Host(3, "BL21(DE3)pLysS", "E. coli", "F– ompT gal dcm lon hsdSB(rB–mB–) λ(DE3) pLysS(cmR)",
                 "Reduced basal expression, good for toxic proteins", ["T7 expression", "Reduced leaky expression", "Toxic protein compatible"],
                 ["Lower overall expression", "Additional antibiotic required"]),
            Host(4, "C41(DE3)", "E. coli", "F– ompT gal dcm hsdSB(rB- mB-) (DE3)",
                 "Optimized for membrane proteins and toxic proteins", ["Membrane protein expression", "Toxic protein compatible", "T7 expression"],
                 ["Lower expression of soluble proteins"]),
            Host(5, "SHuffle T7", "E. coli", "Enhanced disulfide bond formation strain",
                 "Enhanced disulfide bond formation in cytoplasm", ["Disulfide bond formation", "T7 expression", "Oxidizing cytoplasmic environment"],
                 ["Slower growth", "Lower overall yield"]),
            Host(6, "ArcticExpress(DE3)", "E. coli", "Cold expression strain with chaperones",
                 "Cold-adapted chaperonins for low temperature expression", ["Low temperature expression", "Cold-adapted chaperones", "T7 expression"],
                 ["Gentamicin resistance", "Slower growth"])
        ]
        save_hosts(hosts)
        
    return hosts

def save_hosts(hosts):
    """Save hosts to JSON file"""
    host_data = [h.to_dict() for h in hosts]
    with open('data/hosts.json', 'w') as f:
        json.dump(host_data, f, indent=2)

# ----------------------
# Sequence Analysis Functions
# ----------------------

def analyze_protein_sequence(sequence):
    """Analyze a protein sequence for expression optimization"""
    # Clean sequence
    sequence = re.sub(r'[^A-Za-z]', '', sequence.upper())
    
    if not sequence:
        return {"error": "Invalid sequence. Please provide a valid protein sequence."}
    
    try:
        # Basic properties
        mol_weight = len(sequence) * 110 / 1000  # Rough estimation in kDa
        
        # Amino acid composition
        aa_count = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            aa_count[aa] = sequence.count(aa)
        
        total_aa = len(sequence)
        aa_percent = {aa: count/total_aa for aa, count in aa_count.items()}
        
        # Hydrophobicity (simple estimation)
        hydrophobic_aas = 'AILMFWYV'
        hydrophobicity = sum(1 for aa in sequence if aa in hydrophobic_aas) / total_aa
        
        # Charged residues
        charged_aas = 'DEKR'
        charged_residues = sum(1 for aa in sequence if aa in charged_aas) / total_aa
        
        # Cysteine analysis
        cys_count = aa_count.get('C', 0)
        has_disulfide_potential = cys_count >= 2
        
        # Instability prediction (simplified)
        instability_index = abs(hydrophobicity - 0.5) * 100 + charged_residues * 50
        is_stable = instability_index < 40
        
        # Issues and recommendations
        issues = []
        recommendations = []
        
        if has_disulfide_potential:
            issues.append(f"Multiple cysteines detected ({cys_count} cysteines)")
            recommendations.append("Consider using SHuffle T7 strain for disulfide bond formation")
        
        is_hydrophobic = hydrophobicity > 0.4
        if is_hydrophobic:
            issues.append("Highly hydrophobic protein (potential membrane protein)")
            recommendations.append("Consider using C41(DE3) strain for membrane proteins")
            recommendations.append("Lower expression temperature (16-25°C)")
        
        if not is_stable:
            issues.append(f"Potentially unstable protein (instability index: {instability_index:.1f})")
            recommendations.append("Consider fusion tags like MBP or GST to improve stability")
            recommendations.append("Lower expression temperature (16-25°C)")
        
        if mol_weight > 70:
            issues.append(f"Large protein ({mol_weight:.1f} kDa)")
            recommendations.append("Consider co-expression with chaperones")
            recommendations.append("Lower expression temperature (16-25°C)")
        
        # Rare codons
        rare_aas = []
        rare_threshold = 0.05
        rare_codons = {'R': 'Arginine', 'C': 'Cysteine', 'I': 'Isoleucine', 'L': 'Leucine', 'P': 'Proline'}
        
        for aa, name in rare_codons.items():
            if aa_percent.get(aa, 0) > rare_threshold:
                rare_aas.append(name)
        
        if rare_aas:
            issues.append(f"High content of amino acids with rare codons: {', '.join(rare_aas)}")
            recommendations.append("Consider using Rosetta(DE3) strain for rare codon optimization")
        
        return {
            "sequence_length": total_aa,
            "molecular_weight": round(mol_weight, 2),
            "instability_index": round(instability_index, 2),
            "is_stable": is_stable,
            "hydrophobicity": round(hydrophobicity, 3),
            "charged_residues": round(charged_residues, 3),
            "cysteine_count": cys_count,
            "has_disulfide_potential": has_disulfide_potential,
            "is_hydrophobic": is_hydrophobic,
            "issues": issues,
            "recommendations": recommendations,
            "protein_properties": {
                "size": mol_weight,
                "has_disulfide_bonds": has_disulfide_potential,
                "is_membrane_protein": is_hydrophobic,
                "is_toxic": not is_stable
            }
        }
    except Exception as e:
        return {"error": f"Error analyzing sequence: {str(e)}"}

# ----------------------
# UI Components
# ----------------------

def init_session_state():
    """Initialize session state variables"""
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'selected_vector' not in st.session_state:
        st.session_state.selected_vector = None
    if 'selected_host' not in st.session_state:
        st.session_state.selected_host = None
    if 'protein_sequence' not in st.session_state:
        st.session_state.protein_sequence = ""
    if 'sequence_analysis' not in st.session_state:
        st.session_state.sequence_analysis = None
    if 'expression_parameters' not in st.session_state:
        st.session_state.expression_parameters = {
            'temperature': 30,
            'induction_time': 4,
            'inducer_concentration': 0.5,
            'OD600_at_induction': 0.6,
            'media_composition': 'LB'
        }
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False

def show_header():
    """Display the application header"""
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        st.image("https://img.icons8.com/fluency/96/test-tube.png", width=80)
    
    with col2:
        st.title("RennetOptiMax Pro")
        st.markdown("### 🧬 AI-Powered Protein Expression Optimization Platform")
    
    with col3:
        if st.session_state.authenticated:
            if st.button("🏠 Dashboard", key="dashboard_btn"):
                st.session_state.page = 'dashboard'
                st.rerun()

def show_product_banner():
    """Display product promotion banner with NeoRen link"""
    st.markdown(f"""
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
            <a href="{NEOREN_WEBSITE}" target="_blank" style="background: #ff6b6b; 
                          color: white; 
                          border: none; 
                          padding: 15px 30px; 
                          border-radius: 5px; 
                          font-size: 18px; 
                          text-decoration: none;
                          display: inline-block;">
                🛒 Buy 500g & Get 1 Year Free Access
            </a>
        </div>
        <p><small>Sustainable rennet production through advanced genetic engineering</small></p>
    </div>
    """, unsafe_allow_html=True)

def show_login_form():
    """Display login form"""
    st.markdown("## 🔐 Login to Your Account")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            login_button = st.form_submit_button("🔐 Login", use_container_width=True, type="primary")
        
        with col2:
            demo_button = st.form_submit_button("🎮 Demo Login", use_container_width=True)
        
        if login_button:
            if username and password:
                authenticated, user_data = authenticate_user(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_name = user_data['name']
                    st.session_state.show_login = False
                    st.success(f"Welcome back, {user_data['name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Please enter both username and password")
        
        if demo_button:
            # Demo login with admin credentials
            st.session_state.authenticated = True
            st.session_state.username = 'admin'
            st.session_state.user_name = 'Administrator'
            st.session_state.show_login = False
            st.success("Demo login successful!")
            st.rerun()

def show_navigation():
    """Display navigation sidebar"""
    st.sidebar.title("🧬 RennetOptiMax Pro")
    
    # Authentication status
    if st.session_state.authenticated:
        users = load_users()
        user_data = users.get(st.session_state.username, {})
        
        st.sidebar.success(f"👋 Welcome, {user_data.get('name', 'User')}")
        
        # Navigation pages
        pages = {
            'dashboard': "🏠 Dashboard",
            'home': "🌟 Home",
            'vectors': "1. 🧬 Select Vector",
            'hosts': "2. 🦠 Select Host",
            'sequence': "3. 🔬 Analyze Sequence",
            'parameters': "4. ⚙️ Set Parameters",
            'optimize': "5. 🎯 Optimize Expression",
            'results': "6. 📊 View Results"
        }
        
        # Display navigation buttons
        for page_id, page_name in pages.items():
            if st.sidebar.button(page_name, key=f"nav_{page_id}", use_container_width=True,
                                type="primary" if st.session_state.page == page_id else "secondary"):
                st.session_state.page = page_id
                st.rerun()
        
        # Account management
        st.sidebar.divider()
        st.sidebar.subheader("Account")
        
        if st.sidebar.button("⚙️ Account Settings", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
        
        # Logout
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_name = None
            st.session_state.page = 'home'
            st.rerun()
    
    else:
        # Not authenticated - show only home
        if st.sidebar.button("🌟 Home", use_container_width=True):
            st.session_state.page = 'home'
            st.rerun()
    
    # About section
    st.sidebar.divider()
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "RennetOptiMax Pro: AI-powered protein expression optimization "
        "for sustainable rennet production."
    )
    
    # Link to NeoRen website in sidebar
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin: 10px 0;">
        <a href="{NEOREN_WEBSITE}" target="_blank" style="background: #4CAF50; 
                  color: white; 
                  padding: 8px 16px; 
                  border-radius: 5px; 
                  text-decoration: none; 
                  display: inline-block;
                  font-size: 12px;">
            🌐 Visit NeoRen
        </a>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Features Overview
    st.markdown("### 🚀 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🧬 Vector Selection**
        - 6+ expression vectors
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
        if st.button("🆓 Demo Login", use_container_width=True, type="primary"):
            st.session_state.authenticated = True
            st.session_state.username = 'admin'
            st.session_state.user_name = 'Administrator'
            st.rerun()
    
    with col2:
        if st.button("🔑 Login", use_container_width=True):
            st.session_state.show_login = True
            st.rerun()
    
    with col3:
        st.link_button("🛒 Buy Product + Access", NEOREN_WEBSITE, use_container_width=True)

def show_dashboard():
    """Display user dashboard"""
    if not st.session_state.authenticated:
        st.error("Please login to access the dashboard.")
        return
    
    users = load_users()
    user_data = users.get(st.session_state.username, {})
    
    st.markdown(f"## 👋 Welcome, {user_data.get('name', 'User')}!")
    
    # Account Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Account Type", user_data.get('user_type', 'Unknown').title())
    
    with col2:
        status = user_data.get('subscription_status', 'None')
        st.metric("📊 Status", status.title())
    
    with col3:
        referral_code = user_data.get('referral_code', 'N/A')
        st.metric("🔗 Referral Code", referral_code)
    
    with col4:
        if user_data.get('subscription_expiry'):
            try:
                expiry = datetime.fromisoformat(user_data['subscription_expiry'])
                days_left = (expiry - datetime.now()).days
                st.metric("⏰ Days Left", max(0, days_left))
            except:
                st.metric("⏰ Access", "Active")
        else:
            st.metric("⏰ Access", "Lifetime")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧬 Start Optimization", use_container_width=True):
            st.session_state.page = 'vectors'
            st.rerun()
    
    with col2:
        if st.button("📊 View Results", use_container_width=True):
            st.session_state.page = 'results'
            st.rerun()
    
    with col3:
        st.link_button("🛒 Buy NeoRen Product", NEOREN_WEBSITE, use_container_width=True)

def show_vectors_page():
    """Display vector selection page"""
    username = st.session_state.username
    
    if not username or not check_user_access(username, 'vectors'):
        st.error("🔒 This feature requires a subscription. Please upgrade your account.")
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
                if j < len(cols):
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
                            st.rerun()

def show_hosts_page():
    """Display host selection page"""
    username = st.session_state.username
    
    if not username or not check_user_access(username, 'hosts'):
        st.error("🔒 This feature requires a subscription. Please upgrade your account.")
        return
    
    st.markdown("## 🦠 Host Strain Selection")
    
    # Load host database
    hosts = load_hosts()
    
    st.markdown("### 🦠 Available Host Strains")
    
    for i in range(0, len(hosts), 2):
        row_hosts = hosts[i:i+2]
        cols = st.columns(2)
        
        for j, host in enumerate(row_hosts):
            if j < len(cols):
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
                        st.rerun()

def show_sequence_page():
    """Display sequence analysis page"""
    username = st.session_state.username
    
    if not username or not check_user_access(username, 'sequence'):
        st.error("🔒 This feature requires a subscription. Please upgrade your account.")
        return
    
    st.markdown("## 🔬 Protein Sequence Analysis")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["Input Sequence", "Sample Sequences"])
    
    with tab1:
        sequence_text = st.text_area(
            "Enter protein sequence (amino acids):",
            value=st.session_state.protein_sequence,
            height=200,
            help="Paste your protein sequence in single letter amino acid format (A, C, D, etc.)"
        )
        
        if st.button("Analyze Sequence", key="analyze_btn", use_container_width=True):
            if not sequence_text:
                st.error("Please enter a protein sequence.")
            else:
                st.session_state.protein_sequence = sequence_text
                st.session_state.sequence_analysis = analyze_protein_sequence(sequence_text)
                st.rerun()
    
    with tab2:
        sample_sequences = {
            "Rennet (Chymosin)": "MEMKFLIFVLTILVLPVFGNLLVYAPFDEEPQQPWQVLSLRYNTKETCEKLVLLDLNQAPLPWHVTVQEDGRCLGGHLEAHQLYCNVTKSEHFRLATHLNDVVLAPTFCQESIENDSKLVLLDVDLPLSHFQLSAAPGTTLEASPNFISHYGIQHLCPNDIYPAGNCSEEGMDLRVTVSSTMDPNQLFTLQISRPWIVIGSDCPLDGLDCEPGYPCDFHPKYGQDGTVPFLVYEAYKSWKQTGVEILQTYCIYPSVVSPHCTSPTSSEPAPQDTVSLTIINHEIPYSQEALVRFENGSKNFRLGEHYLKACGETAYVWHEARKTNRFQVESFKESNTYLMHNLLDKYNCNVGFMPAYGFDQIIEGEEIVLRHSGEFAFSPETPASYTCVNEIFLRPTSNAYLKAQSCWAIPLFNSVPSTLMYMKYCGWSTANPDEIEIGSNSSHYKRTFGQNLDSSDKLNFTDMAGEVISVAITKSQGEKSDHHHHHHHSRSAAGRLEHHHHHH",
            "GFP": "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
            "Small Test Protein": "MGSSHHHHHHSSGLVPRGSHMQCVVLVTLLCFAACSAVCEPRCEPRCEPRCNNGCPAFCQCLYNGCPVLGAEESPTIVKGKDMCSPCGKNGPKACEAEKSKCNGGHCPFAKPCKKGCKGRCQYNYPDKKGFGSCPFVENVPYTIKVGSCPFNFNTFANKCRFGYQMGTLCPFEDPHSKPCTDGMTPTMCPEDCESGLRYSTCPFNYQPNDKLEWPRCPTGYRTTDKACPDGMPSQVCPSAQTTTAPAAKQSPAAKQSPAAKQSPAAKQSPAAAK"
        }
        
        selected_sample = st.selectbox(
            "Select a sample protein:",
            list(sample_sequences.keys())
        )
        
        st.text_area("Sample sequence:", value=sample_sequences[selected_sample], height=150)
        
        if st.button("Use This Sample", key="use_sample_btn", use_container_width=True):
            st.session_state.protein_sequence = sample_sequences[selected_sample]
            st.session_state.sequence_analysis = analyze_protein_sequence(sample_sequences[selected_sample])
            st.rerun()
    
    # Display analysis results if available
    if st.session_state.sequence_analysis:
        st.markdown("### 🔬 Sequence Analysis Results")
        
        analysis = st.session_state.sequence_analysis
        
        if 'error' in analysis:
            st.error(analysis['error'])
        else:
            # Create columns for basic properties
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Length", f"{analysis['sequence_length']} aa")
            
            with col2:
                st.metric("Molecular Weight", f"{analysis['molecular_weight']} kDa")
            
            with col3:
                st.metric("Hydrophobicity", f"{analysis['hydrophobicity']:.3f}")
            
            with col4:
                stability = "Stable" if analysis['is_stable'] else "Unstable"
                st.metric("Stability", stability, 
                          delta="Good" if analysis['is_stable'] else "Concern", 
                          delta_color="normal" if analysis['is_stable'] else "inverse")
            
            # Potential issues
            if analysis['issues']:
                st.subheader("⚠️ Potential Expression Issues")
                for issue in analysis['issues']:
                    st.warning(issue)
            
            # Recommendations
            if analysis['recommendations']:
                st.subheader("💡 Recommendations")
                for recommendation in analysis['recommendations']:
                    st.info(recommendation)
            
            # Special properties
            special_props = []
            
            if analysis['has_disulfide_potential']:
                special_props.append("🔗 Has potential disulfide bonds (" + str(analysis['cysteine_count']) + " cysteines)")
            
            if analysis['is_hydrophobic']:
                special_props.append("💧 Hydrophobic (possible membrane protein)")
            
            if special_props:
                st.subheader("🔍 Special Properties")
                for prop in special_props:
                    st.info(prop)

def show_parameters_page():
    """Display expression parameters configuration page"""
    username = st.session_state.username
    
    if not username or not check_user_access(username, 'parameters'):
        st.error("🔒 This feature requires a subscription. Please upgrade your account.")
        return
    
    st.markdown("## ⚙️ Expression Parameters Configuration")
    
    # Check if vector and host are selected
    if not st.session_state.selected_vector or not st.session_state.selected_host:
        st.warning("Please select a vector and host before configuring expression parameters.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select Vector", use_container_width=True):
                st.session_state.page = 'vectors'
                st.rerun()
        with col2:
            if st.button("Select Host", use_container_width=True):
                st.session_state.page = 'hosts'
                st.rerun()
        
        return
    
    # Show current selections
    st.markdown("### 📋 Current Selections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Vector:** {st.session_state.selected_vector.name}
        - Promoter: {st.session_state.selected_vector.promoter}
        - Selection: {st.session_state.selected_vector.selection_marker}
        - Tags: {', '.join(st.session_state.selected_vector.tags)}
        """)
    
    with col2:
        st.markdown(f"""
        **Host:** {st.session_state.selected_host.strain}
        - Species: {st.session_state.selected_host.species}
        - Features: {', '.join(st.session_state.selected_host.features)}
        """)
    
    st.divider()
    
    # Parameters configuration
    st.markdown("### ⚙️ Configure Expression Parameters")
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "Temperature (°C)",
            min_value=16,
            max_value=37,
            value=st.session_state.expression_parameters['temperature'],
            step=1,
            help="Lower temperatures often result in better folding but slower expression"
        )
        
        induction_time = st.slider(
            "Induction Time (hours)",
            min_value=2,
            max_value=16,
            value=st.session_state.expression_parameters['induction_time'],
            step=1,
            help="Longer induction times may increase yield for well-behaved proteins"
        )
        
        inducer_concentration = st.slider(
            "Inducer Concentration (mM)",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.expression_parameters['inducer_concentration'],
            step=0.1,
            help="IPTG concentration for T7 or tac promoters, arabinose for pBAD"
        )
    
    with col2:
        od600 = st.slider(
            "OD600 at Induction",
            min_value=0.4,
            max_value=1.0,
            value=st.session_state.expression_parameters['OD600_at_induction'],
            step=0.1,
            help="Cell density at which to induce protein expression"
        )
        
        media_options = ["LB", "TB", "M9", "2xYT", "SOC"]
        media = st.selectbox(
            "Media Composition",
            options=media_options,
            index=media_options.index(st.session_state.expression_parameters['media_composition']) 
            if st.session_state.expression_parameters['media_composition'] in media_options else 0,
            help="Rich media (TB, 2xYT) gives higher yields; minimal media (M9) allows better control"
        )
    
    # Auto-suggest parameters based on protein properties
    st.divider()
    st.markdown("### 🤖 Auto-Suggest Parameters")
    
    if st.session_state.sequence_analysis and 'protein_properties' in st.session_state.sequence_analysis:
        protein_properties = st.session_state.sequence_analysis['protein_properties']
        
        if st.button("Suggest Optimal Parameters Based on Protein Properties", use_container_width=True):
            # Get customized parameters based on protein properties
            suggested_params = st.session_state.expression_parameters.copy()
            
            if protein_properties.get('has_disulfide_bonds', False):
                suggested_params['temperature'] = 25
                st.info("🔗 Disulfide bonds detected: Lowered temperature to 25°C")
            
            if protein_properties.get('is_membrane_protein', False):
                suggested_params['temperature'] = 16
                suggested_params['inducer_concentration'] = 0.2
                st.info("💧 Membrane protein detected: Lowered temperature and inducer concentration")
            
            if protein_properties.get('size', 0) > 70:
                suggested_params['induction_time'] = 8
                st.info("📏 Large protein detected: Extended induction time to 8 hours")
                
            if protein_properties.get('is_toxic', False):
                suggested_params['OD600_at_induction'] = 0.8
                suggested_params['inducer_concentration'] = 0.2
                st.info("⚠️ Potentially toxic protein: Higher OD and lower inducer concentration")
            
            # Rich media generally better for expression
            suggested_params['media_composition'] = "TB"
            st.info("🧪 Recommended rich media (TB) for better expression")
            
            # Update session state
            st.session_state.expression_parameters = suggested_params
            
            st.success("Parameters updated based on protein analysis!")
            st.rerun()
    else:
        st.info("For parameter suggestions based on protein properties, please analyze your protein sequence first.")
        
        if st.button("Go to Sequence Analysis", use_container_width=True):
            st.session_state.page = 'sequence'
            st.rerun()
    
    # Save parameters
    if st.button("Save Parameters", type="primary", use_container_width=True):
        st.session_state.expression_parameters = {
            'temperature': temperature,
            'induction_time': induction_time,
            'inducer_concentration': inducer_concentration,
            'OD600_at_induction': od600,
            'media_composition': media
        }
        
        st.success("Expression parameters saved!")
        
        # Show next step button
        if st.button("Proceed to Optimization", use_container_width=True):
            st.session_state.page = 'optimize'
            st.rerun()

def show_optimization_page():
    """Display optimization page"""
    username = st.session_state.username
    
    if not username or not check_user_access(username, 'optimize'):
        st.error("🔒 This feature requires a subscription. Please upgrade your account.")
        return
    
    st.markdown("## 🎯 Expression Optimization")
    
    # Check if all required selections are made
    if not st.session_state.selected_vector or not st.session_state.selected_host:
        st.warning("Please select a vector and host before optimization.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select Vector", use_container_width=True):
                st.session_state.page = 'vectors'
                st.rerun()
        with col2:
            if st.button("Select Host", use_container_width=True):
                st.session_state.page = 'hosts'
                st.rerun()
        
        return
    
    # Optimization parameters
    st.markdown("### 📊 Optimization Parameters")
    
    # Show current selections in a table
    selection_data = [
        ["Vector", st.session_state.selected_vector.name],
        ["Host", st.session_state.selected_host.strain],
        ["Temperature", f"{st.session_state.expression_parameters['temperature']} °C"],
        ["Induction Time", f"{st.session_state.expression_parameters['induction_time']} hours"],
        ["Inducer Concentration", f"{st.session_state.expression_parameters['inducer_concentration']} mM"],
        ["OD600 at Induction", str(st.session_state.expression_parameters['OD600_at_induction'])],
        ["Media", st.session_state.expression_parameters['media_composition']]
    ]
    
    selection_df = pd.DataFrame(selection_data, columns=["Parameter", "Value"])
    st.table(selection_df)
    
    # Additional optimization settings
    st.markdown("### ⚙️ Additional Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_suggestions = st.slider("Number of Suggestions", min_value=1, max_value=10, value=5)
    
    with col2:
        if st.session_state.sequence_analysis and 'protein_properties' in st.session_state.sequence_analysis:
            protein_properties = st.session_state.sequence_analysis['protein_properties']
            
            is_membrane = st.checkbox("Membrane Protein", 
                                     value=protein_properties.get('is_membrane_protein', False))
            has_disulfide = st.checkbox("Contains Disulfide Bonds", 
                                       value=protein_properties.get('has_disulfide_bonds', False))
            is_toxic = st.checkbox("Potentially Toxic to Host", 
                                  value=protein_properties.get('is_toxic', False))
        else:
            is_membrane = st.checkbox("Membrane Protein", value=False)
            has_disulfide = st.checkbox("Contains Disulfide Bonds", value=False)
            is_toxic = st.checkbox("Potentially Toxic to Host", value=False)
    
    # Create protein properties dictionary
    protein_properties = {
        'size': 300,  # Default size
        'has_disulfide_bonds': has_disulfide,
        'is_membrane_protein': is_membrane,
        'is_toxic': is_toxic
    }
    
    # If sequence analysis available, use actual properties
    if st.session_state.sequence_analysis and 'protein_properties' in st.session_state.sequence_analysis:
        protein_properties['size'] = st.session_state.sequence_analysis['protein_properties'].get('size', 300)
    
    # Run optimization
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        with st.spinner("Running optimization..."):
            # Initialize optimizer
            optimizer = ExpressionOptimizer()
            optimizer.load_model()
            
            # Current conditions
            current_conditions = {
                'vector_type': st.session_state.selected_vector.name,
                'host_strain': st.session_state.selected_host.strain,
                'temperature': st.session_state.expression_parameters['temperature'],
                'induction_time': st.session_state.expression_parameters['induction_time'],
                'inducer_concentration': st.session_state.expression_parameters['inducer_concentration'],
                'OD600_at_induction': st.session_state.expression_parameters['OD600_at_induction'],
                'media_composition': st.session_state.expression_parameters['media_composition']
            }
            
            # Predict expression with current conditions
            current_expression = optimizer.predict_expression(current_conditions)
            
            # Get optimization suggestions
            suggestions = optimizer.suggest_optimal_conditions(
                vector_name=st.session_state.selected_vector.name,
                host_strain=st.session_state.selected_host.strain,
                protein_properties=protein_properties,
                n_suggestions=n_suggestions
            )
            
            # Store results
            st.session_state.optimization_results = {
                'current_conditions': current_conditions,
                'current_expression': current_expression,
                'suggestions': suggestions,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'vector': st.session_state.selected_vector.to_dict(),
                'host': st.session_state.selected_host.to_dict()
            }
            
            # Go to results page
            st.session_state.page = 'results'
            st.rerun()

def show_results_page():
    """Display optimization results page"""
    username = st.session_state.username
    
    if not username or not check_user_access(username, 'results'):
        st.error("🔒 This feature requires a subscription. Please upgrade your account.")
        return
    
    st.markdown("## 📊 Optimization Results")
    
    if not st.session_state.optimization_results:
        st.warning("No optimization results available. Please run optimization first.")
        
        if st.button("Go to Optimization", use_container_width=True):
            st.session_state.page = 'optimize'
            st.rerun()
        
        return
    
    results = st.session_state.optimization_results
    
    # Display optimization time
    st.markdown(f"*Optimization completed at: {results['timestamp']}*")
    
    # Current vs Optimal conditions
    st.markdown("### 📈 Expression Prediction")
    
    # Calculate improvement
    current_expr = results['current_expression']
    best_suggestion = results['suggestions'][0] if results['suggestions'] else None
    
    if best_suggestion:
        best_expr = best_suggestion['predicted_expression']
        improvement = best_expr - current_expr
        improvement_percent = (improvement / current_expr) * 100 if current_expr > 0 else 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Parameters",
                f"{current_expr:.1f}%",
                help="Predicted expression with your selected parameters"
            )
        
        with col2:
            st.metric(
                "Optimal Parameters",
                f"{best_expr:.1f}%",
                delta=f"+{improvement:.1f}%",
                help="Predicted expression with optimized parameters"
            )
        
        with col3:
            st.metric(
                "Improvement",
                f"{improvement_percent:.1f}%",
                help="Percentage improvement over current parameters"
            )
        
        # Create visualization
        st.markdown("### 📊 Visualization of Results")
        
        # Prepare data for chart
        expressions = [results['current_expression']] + [s['predicted_expression'] for s in results['suggestions']]
        labels = ['Current'] + [f'Option {i+1}' for i in range(len(results['suggestions']))]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=expressions,
                marker_color=['#1976d2'] + ['#4caf50'] * len(results['suggestions']),
                text=[f"{e:.1f}%" for e in expressions],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Predicted Expression Levels",
            xaxis_title="Parameters",
            yaxis_title="Predicted Expression (%)",
            yaxis_range=[0, max(expressions) * 1.1],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display suggested conditions
        st.markdown("### 💡 Suggested Expression Conditions")
        
        # Create tabs for different suggestions
        suggestion_tabs = st.tabs([f"Option {i+1}" for i in range(len(results['suggestions']))])
        
        for i, (tab, suggestion) in enumerate(zip(suggestion_tabs, results['suggestions'])):
            with tab:
                params = suggestion['parameters']
                
                # Create two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🧪 Expression Parameters")
                    
                    st.markdown(f"""
                    - **Vector:** {params['vector_type']}
                    - **Host:** {params['host_strain']}
                    - **Temperature:** {params['temperature']} °C
                    - **Induction Time:** {params['induction_time']} hours
                    - **Inducer Concentration:** {params['inducer_concentration']} mM
                    - **OD600 at Induction:** {params['OD600_at_induction']}
                    - **Media:** {params['media_composition']}
                    """)
                
                with col2:
                    st.markdown("#### 📊 Predicted Performance")
                    
                    st.markdown(f"""
                    - **Predicted Expression:** {suggestion['predicted_expression']:.1f}%
                    - **Improvement:** {(suggestion['predicted_expression'] - current_expr):.1f}%
                    """)
                    
                    if 'additives' in suggestion and suggestion['additives']:
                        st.markdown("#### 🧬 Recommended Additives")
                        
                        for additive in suggestion['additives']:
                            st.info(additive)
        
        # Generate expression protocol
        st.markdown("### 📋 Expression Protocol")
        
        # Use the best suggestion for the protocol
        best_params = best_suggestion['parameters']
        
        # Create protocol text
        protocol = f"""
        # Optimized Expression Protocol for {results['vector']['name']} in {results['host']['strain']}
        
        ## Materials
        
        - {results['host']['strain']} competent cells
        - {results['vector']['name']} vector containing your gene of interest
        - {best_params['media_composition']} medium
        - {results['vector']['selection_marker']} antibiotic
        - Inducer: {"IPTG" if best_params['vector_type'].startswith(('pET', 'pGEX', 'pMAL')) else "Arabinose"}
        - Sterile culture flasks
        - Incubator with shaking capability
        
        ## Procedure
        
        1. **Transformation**
           - Transform {results['host']['strain']} cells with your {results['vector']['name']} construct
           - Plate on {results['vector']['selection_marker']} agar and incubate overnight at 37°C
        
        2. **Starter Culture**
           - Pick a single colony and inoculate 5 mL {best_params['media_composition']} medium with {results['vector']['selection_marker']}
           - Incubate overnight at 37°C with shaking (200 rpm)
        
        3. **Expression Culture**
           - Dilute the starter culture 1:100 into fresh {best_params['media_composition']} with {results['vector']['selection_marker']}
           - Grow at 37°C with shaking until OD600 reaches {best_params['OD600_at_induction']}
        
        4. **Induction**
           - Add {"IPTG" if best_params['vector_type'].startswith(('pET', 'pGEX', 'pMAL')) else "Arabinose"} to a final concentration of {best_params['inducer_concentration']} mM
           - Continue incubation at {best_params['temperature']}°C for {best_params['induction_time']} hours
        
        5. **Harvesting**
           - Collect cells by centrifugation at 4,000g for 15 minutes at 4°C
           - Discard supernatant and freeze cell pellet at -20°C or proceed directly to lysis
        
        6. **Cell Lysis**
           - Resuspend the cell pellet in lysis buffer appropriate for your purification method
           - Lyse cells by sonication or other preferred method
           - Clarify lysate by centrifugation at 15,000g for 30 minutes at 4°C
        
        7. **Purification**
           - Proceed with purification appropriate for your tag and protein
        """
        
        # Display protocol in a code block
        st.code(protocol, language="markdown")
        
        # Download button for protocol
        protocol_download = protocol.replace('#', '').strip()
        
        st.download_button(
            label="📄 Download Protocol as Text",
            data=protocol_download,
            file_name=f"expression_protocol_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # Option to save results
        if st.button("💾 Save Results to File", use_container_width=True):
            # Save results to JSON
            results_file = f"results/optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Prepare results for saving
            save_results = {
                'current_conditions': results['current_conditions'],
                'current_expression': float(results['current_expression']),
                'suggestions': [{
                    'parameters': s['parameters'],
                    'predicted_expression': float(s['predicted_expression']),
                    'additives': s.get('additives', [])
                } for s in results['suggestions']],
                'timestamp': results['timestamp'],
                'vector': results['vector'],
                'host': results['host']
            }
            
            # Save to file
            with open(results_file, 'w') as f:
                json.dump(save_results, f, indent=2)
            
            st.success(f"Results saved to {results_file}")

def show_restricted_feature(feature_name):
    """Show message for restricted features"""
    st.error(f"🔒 {feature_name} requires a subscription. Please upgrade your account.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Upgrade Now", use_container_width=True, type="primary"):
            st.session_state.page = 'dashboard'
            st.rerun()
    
    with col2:
        if st.button("🎮 Demo Login", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.username = 'admin'
            st.session_state.user_name = 'Administrator'
            st.rerun()
    
    with col3:
        st.link_button("🛒 Buy NeoRen Product", NEOREN_WEBSITE, use_container_width=True)

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
    
    # Show header
    show_header()
    
    # Show navigation
    show_navigation()
    
    # Handle authentication flows
    if st.session_state.show_login:
        show_login_form()
        if st.button("← Back to Home"):
            st.session_state.show_login = False
            st.rerun()
        return
    
    # Display the appropriate page based on session state
    current_page = st.session_state.page
    username = st.session_state.username
    
    if current_page == 'home':
        show_home_page()
    elif current_page == 'dashboard':
        show_dashboard()
    elif current_page == 'vectors':
        show_vectors_page()
    elif current_page == 'hosts':
        show_hosts_page()
    elif current_page == 'sequence':
        show_sequence_page()
    elif current_page == 'parameters':
        show_parameters_page()
    elif current_page == 'optimize':
        show_optimization_page()
    elif current_page == 'results':
        show_results_page()

# Run the application
if __name__ == "__main__":
    main()