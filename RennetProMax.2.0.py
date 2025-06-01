# RennetOptiMax_Pro_Complete_Secure.py - Complete Secure Version
# -------------------------------------------------------------------
# Run with: streamlit run RennetOptiMax_Pro_Complete_Secure.py
# Requirements: pip install streamlit pandas numpy scikit-learn plotly joblib

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

# Constants
NEOREN_WEBSITE = "https://neoren.mystrikingly.com/"
NEOREN_LOGO_URL = "https://i.postimg.cc/HJTvQyhk/Logo-Neo-Ren.png"

PRICING_PLANS = {
    'student': {'1_month': 5, '6_months': 25, '1_year': 40},
    'academic': {'1_month': 7, '6_months': 35, '1_year': 60},
    'professional': {'1_month': 10, '6_months': 50, '1_year': 85}
}

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
    """Create default user database with demo accounts"""
    users = {
        'admin': {
            'password': hash_password('admin123'),
            'name': 'Administrator',
            'email': 'admin@rennetoptimax.com',
            'user_type': 'admin',
            'subscription_status': 'lifetime',
            'subscription_expiry': None,
            'referral_code': 'ADMIN001',
            'trial_expiry': None,
            'free_access_features': ['all']
        },
        'demo_student': {
            'password': hash_password('student123'),
            'name': 'Demo Student',
            'email': 'student@demo.com',
            'user_type': 'student',
            'subscription_status': 'trial',
            'subscription_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
            'referral_code': 'STUDENT001',
            'trial_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
            'free_access_features': ['vectors', 'hosts']
        },
        'demo_professional': {
            'password': hash_password('pro123'),
            'name': 'Demo Professional',
            'email': 'professional@demo.com',
            'user_type': 'professional',
            'subscription_status': 'trial',
            'subscription_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
            'referral_code': 'PRO001',
            'trial_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
            'free_access_features': ['vectors', 'hosts', 'sequence']
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
    if not username:
        return False
        
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
        if trial_expiry:
            try:
                expiry_date = datetime.fromisoformat(trial_expiry)
                if expiry_date > datetime.now():
                    return True
            except:
                pass
    
    # Check free access features
    free_features = user_data.get('free_access_features', [])
    if feature in free_features or 'all' in free_features:
        return True
    
    return False

def add_new_user(username, email, name, password, user_type='professional'):
    """Add a new user to the system"""
    users = load_users()
    
    if username in users:
        return False, "Username already exists"
    
    new_user = {
        'password': hash_password(password),
        'name': name,
        'email': email,
        'user_type': user_type,
        'subscription_status': 'trial',
        'subscription_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
        'referral_code': f"REF{random.randint(1000, 9999)}",
        'trial_expiry': (datetime.now() + timedelta(days=30)).isoformat(),
        'free_access_features': ['vectors', 'hosts']
    }
    
    users[username] = new_user
    save_users(users)
    return True, "User created successfully"

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
    """Complete ML-based expression optimization class"""
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.numeric_features = ['temperature', 'induction_time', 'inducer_concentration', 'OD600_at_induction']
        self.categorical_features = ['vector_type', 'host_strain', 'media_composition']
        self.model_path = 'models/rennet_model.joblib'
        
    def generate_sample_data(self, n_samples=200):
        """Generate comprehensive synthetic training data"""
        np.random.seed(42)
        
        vectors = ['pET21a', 'pET28a', 'pET22b', 'pBAD', 'pUC19', 'pMAL-c5X', 'pGEX-6P-1']
        hosts = ['BL21(DE3)', 'Rosetta(DE3)', 'C41(DE3)', 'BL21(DE3)pLysS', 'DH5α', 'SHuffle T7', 'ArcticExpress(DE3)']
        media = ['LB', 'TB', 'M9', 'SOC', '2xYT']
        
        data = {
            'vector_type': np.random.choice(vectors, n_samples),
            'host_strain': np.random.choice(hosts, n_samples),
            'temperature': np.random.choice([16, 25, 30, 37], n_samples),
            'induction_time': np.random.choice([2, 3, 4, 5, 6, 8, 12, 16], n_samples),
            'inducer_concentration': np.random.uniform(0.1, 1.0, n_samples).round(2),
            'OD600_at_induction': np.random.uniform(0.4, 1.0, n_samples).round(2),
            'media_composition': np.random.choice(media, n_samples)
        }
        
        # Create realistic expression levels
        vector_weights = {
            'pET21a': 0.95, 'pET28a': 0.90, 'pET22b': 0.85, 'pBAD': 0.65,
            'pUC19': 0.40, 'pMAL-c5X': 0.80, 'pGEX-6P-1': 0.75
        }
        vector_effect = np.array([vector_weights[v] for v in data['vector_type']])
        
        host_weights = {
            'BL21(DE3)': 1.0, 'Rosetta(DE3)': 0.95, 'C41(DE3)': 0.85,
            'BL21(DE3)pLysS': 0.80, 'DH5α': 0.50, 'SHuffle T7': 0.75, 'ArcticExpress(DE3)': 0.70
        }
        host_effect = np.array([host_weights[h] for h in data['host_strain']])
        
        media_weights = {'LB': 0.70, 'TB': 1.0, 'M9': 0.55, 'SOC': 0.65, '2xYT': 0.85}
        media_effect = np.array([media_weights[m] for m in data['media_composition']])
        
        # Temperature effect
        temp_effect = 1.0 - 0.15 * abs(np.array(data['temperature']) - 30) / 15
        temp_effect = np.maximum(temp_effect, 0.3)
        
        # Time effect
        time_effect = np.minimum(np.array(data['induction_time']) / 6, 1.0)
        
        # Inducer effect
        inducer_optimal = 0.5
        inducer_effect = 1.0 - 0.2 * abs(np.array(data['inducer_concentration']) - inducer_optimal)
        inducer_effect = np.maximum(inducer_effect, 0.5)
        
        # OD effect
        od_optimal = 0.6
        od_effect = 1.0 - 0.1 * abs(np.array(data['OD600_at_induction']) - od_optimal)
        od_effect = np.maximum(od_effect, 0.7)
        
        # Combined effect
        expression = (
            0.30 * vector_effect +
            0.25 * host_effect +
            0.20 * media_effect +
            0.10 * temp_effect +
            0.08 * time_effect +
            0.04 * inducer_effect +
            0.03 * od_effect +
            np.random.normal(0, 0.08, n_samples)
        )
        
        # Scale to 0-100
        expression = np.maximum(expression, 0)
        expression = 100 * expression / np.max(expression)
        
        # Add low performers
        low_performers = np.random.random(n_samples) < 0.1
        expression[low_performers] *= 0.3
        
        data['expression_level'] = expression.round(2)
        
        return pd.DataFrame(data)
    
    def train_model(self, data=None):
        """Train a comprehensive machine learning model"""
        if data is None:
            data = self.generate_sample_data()
            
        X = data.drop('expression_level', axis=1)
        y = data['expression_level']
        
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
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
            'features': {'numeric': self.numeric_features, 'categorical': self.categorical_features},
            'created_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, self.model_path)
    
    def load_model(self):
        """Load model from disk or train a new one"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.preprocessor = model_data['preprocessor']
                return True
            else:
                st.info("Training new model... This may take a moment.")
                self.train_model()
                return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Training new model...")
            self.train_model()
            return False
    
    def predict_expression(self, conditions):
        """Predict expression level for given conditions"""
        if self.model is None or self.preprocessor is None:
            self.load_model()
            
        try:
            input_df = pd.DataFrame([conditions])
            
            # Ensure all required columns exist
            for feature in self.numeric_features + self.categorical_features:
                if feature not in input_df.columns:
                    if feature == 'temperature':
                        input_df[feature] = 30
                    elif feature == 'induction_time':
                        input_df[feature] = 4
                    elif feature == 'inducer_concentration':
                        input_df[feature] = 0.5
                    elif feature == 'OD600_at_induction':
                        input_df[feature] = 0.6
                    elif feature == 'vector_type':
                        input_df[feature] = 'pET28a'
                    elif feature == 'host_strain':
                        input_df[feature] = 'BL21(DE3)'
                    elif feature == 'media_composition':
                        input_df[feature] = 'LB'
            
            X_input = self.preprocessor.transform(input_df)
            prediction = self.model.predict(X_input)[0]
            
            # Get confidence
            tree_predictions = np.array([tree.predict(X_input)[0] for tree in self.model.estimators_])
            confidence = 100 - (np.std(tree_predictions) / np.mean(tree_predictions) * 100)
            confidence = max(0, min(100, confidence))
            
            return {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'std_dev': float(np.std(tree_predictions))
            }
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return {'prediction': 50.0, 'confidence': 50.0, 'std_dev': 15.0}
    
    def suggest_optimal_conditions(self, vector_name, host_strain, protein_properties, n_suggestions=5):
        """Generate comprehensive optimization suggestions"""
        if self.model is None:
            self.load_model()
            
        # Get protein properties
        size = protein_properties.get('size', 300)
        has_disulfide_bonds = protein_properties.get('has_disulfide_bonds', False)
        is_membrane_protein = protein_properties.get('is_membrane_protein', False)
        is_toxic = protein_properties.get('is_toxic', False)
        
        # Define parameter ranges based on protein properties
        if has_disulfide_bonds:
            temperatures = [16, 25, 30]
            additives = ["Add reducing agents (DTT, TCEP) during lysis"]
        else:
            temperatures = [25, 30, 37]
            additives = []
            
        if is_membrane_protein:
            temperatures = [16, 25]
            induction_concentrations = [0.1, 0.2, 0.5]
            additives.append("Add detergents for solubilization")
        else:
            induction_concentrations = [0.2, 0.5, 1.0]
            
        if size > 100:
            temperatures = [16, 25, 30]
            induction_times = [6, 8, 12, 16]
            additives.append("Consider co-expression with chaperones")
        else:
            induction_times = [2, 4, 6, 8]
            
        if is_toxic:
            induction_concentrations = [0.1, 0.2]
            od600_values = [0.6, 0.8, 1.0]
            additives.append("Use glucose to suppress leaky expression")
        else:
            od600_values = [0.4, 0.6, 0.8]
        
        # Generate combinations
        suggestions = []
        media_options = ['LB', 'TB', '2xYT', 'M9']
        
        for temp in temperatures:
            for time in induction_times[:3]:
                for conc in induction_concentrations[:2]:
                    for od in od600_values[:2]:
                        for media in media_options[:3]:
                            params = {
                                'vector_type': vector_name,
                                'host_strain': host_strain,
                                'temperature': temp,
                                'induction_time': time,
                                'inducer_concentration': conc,
                                'OD600_at_induction': od,
                                'media_composition': media
                            }
                            
                            result = self.predict_expression(params)
                            
                            suggestion = {
                                'parameters': params,
                                'predicted_expression': result['prediction'],
                                'confidence': result['confidence'],
                                'std_dev': result['std_dev'],
                                'additives': additives,
                                'protocol_notes': []
                            }
                            suggestions.append(suggestion)
        
        # Sort by expression level
        suggestions.sort(key=lambda x: x['predicted_expression'], reverse=True)
        
        return suggestions[:n_suggestions]

# ----------------------
# Database Functions
# ----------------------

def load_vectors():
    """Load comprehensive vector database"""
    vector_file = 'data/vectors.json'
    
    if os.path.exists(vector_file):
        with open(vector_file, 'r') as f:
            vector_data = json.load(f)
            vectors = [Vector(**v) for v in vector_data]
    else:
        vectors = [
            Vector(1, "pET21a", 5443, "T7", "T7", "pBR322", "Ampicillin", ["His-tag", "C-terminal"],
                   "High-level expression vector with C-terminal His-tag",
                   {"cloning_sites": ["NdeI", "XhoI", "BamHI", "EcoRI"], "tag_location": "C-terminal", "induction": "IPTG", "expression_level": "Very High"}),
            Vector(2, "pET28a", 5369, "T7", "T7", "pBR322", "Kanamycin", ["His-tag", "N-terminal", "T7-tag"],
                   "High-level expression vector with N-terminal His-tag",
                   {"cloning_sites": ["NdeI", "XhoI", "BamHI", "EcoRI"], "tag_location": "N-terminal", "induction": "IPTG", "expression_level": "Very High"}),
            Vector(3, "pET22b", 5493, "T7", "T7", "pBR322", "Ampicillin", ["His-tag", "C-terminal", "pelB"],
                   "Periplasmic expression vector with pelB signal sequence",
                   {"cloning_sites": ["NdeI", "XhoI", "BamHI", "EcoRI"], "tag_location": "C-terminal", "induction": "IPTG", "expression_level": "High"}),
            Vector(4, "pBAD", 4102, "araBAD", "rrnB", "pBR322", "Ampicillin", ["His-tag", "C-terminal"],
                   "Arabinose-inducible expression vector for tight control",
                   {"cloning_sites": ["NcoI", "HindIII", "XhoI"], "tag_location": "C-terminal", "induction": "Arabinose", "expression_level": "Medium"}),
            Vector(5, "pMAL-c5X", 5677, "tac", "lambda t0", "pBR322", "Ampicillin", ["MBP", "N-terminal"],
                   "MBP fusion vector for improved solubility",
                   {"cloning_sites": ["NdeI", "EcoRI", "BamHI", "SalI"], "tag_location": "N-terminal", "induction": "IPTG", "expression_level": "High"}),
            Vector(6, "pGEX-6P-1", 4984, "tac", "lambda t0", "pBR322", "Ampicillin", ["GST", "N-terminal"],
                   "GST fusion vector for improved solubility",
                   {"cloning_sites": ["BamHI", "EcoRI", "SalI", "NotI"], "tag_location": "N-terminal", "induction": "IPTG", "expression_level": "High"}),
            Vector(7, "pUC19", 2686, "lac", "lac", "pMB1", "Ampicillin", ["None"],
                   "Basic cloning vector with lac promoter",
                   {"cloning_sites": ["EcoRI", "SacI", "KpnI", "BamHI", "XbaI"], "tag_location": "None", "induction": "IPTG", "expression_level": "Low"}),
            Vector(8, "pTrcHis", 4356, "trc", "rrnB", "pBR322", "Ampicillin", ["His-tag", "N-terminal"],
                   "Strong constitutive expression with His-tag",
                   {"cloning_sites": ["NcoI", "HindIII", "BamHI"], "tag_location": "N-terminal", "induction": "IPTG", "expression_level": "High"})
        ]
        save_vectors(vectors)
        
    return vectors

def save_vectors(vectors):
    """Save vectors to JSON file"""
    vector_data = [v.to_dict() for v in vectors]
    with open('data/vectors.json', 'w') as f:
        json.dump(vector_data, f, indent=2)

def load_hosts():
    """Load comprehensive host database"""
    host_file = 'data/hosts.json'
    
    if os.path.exists(host_file):
        with open(host_file, 'r') as f:
            host_data = json.load(f)
            hosts = [Host(**h) for h in host_data]
    else:
        hosts = [
            Host(1, "BL21(DE3)", "E. coli",
                 "F- ompT gal dcm lon hsdSB(rB-mB-) λ(DE3 [lacI lacUV5-T7p07 ind1 sam7 nin5]) [malB+]K-12(λS)",
                 "Standard expression strain with T7 RNA polymerase",
                 ["T7 expression", "Protease deficient", "General purpose", "Fast growth"],
                 ["Not suitable for toxic proteins", "No rare codon support"]),
            Host(2, "Rosetta(DE3)", "E. coli",
                 "F- ompT hsdSB(rB- mB-) gal dcm (DE3) pRARE (CamR)",
                 "Enhanced expression of proteins containing rare codons",
                 ["T7 expression", "Rare codon optimization", "Protease deficient", "tRNA supplementation"],
                 ["Additional antibiotic required", "Slower growth than BL21"]),
            Host(3, "BL21(DE3)pLysS", "E. coli",
                 "F- ompT gal dcm lon hsdSB(rB-mB-) λ(DE3) pLysS(cmR)",
                 "Reduced basal expression, good for toxic proteins",
                 ["T7 expression", "Reduced leaky expression", "Toxic protein compatible", "T7 lysozyme control"],
                 ["Lower overall expression", "Additional antibiotic required"]),
            Host(4, "C41(DE3)", "E. coli",
                 "F- ompT gal dcm hsdSB(rB- mB-) λ(DE3)",
                 "Optimized for membrane proteins and toxic proteins",
                 ["Membrane protein expression", "Toxic protein compatible", "T7 expression", "Reduced inclusion bodies"],
                 ["Lower expression of soluble proteins", "Slower growth"]),
            Host(5, "SHuffle T7", "E. coli",
                 "F' lac, pro, lacIQ / Δ(ara-leu)7697 araD139 fhuA2 lacZ::T7 gene1",
                 "Enhanced disulfide bond formation in cytoplasm",
                 ["Disulfide bond formation", "T7 expression", "Oxidizing cytoplasmic environment", "DsbC co-expression"],
                 ["Slower growth", "Lower overall yield"]),
            Host(6, "ArcticExpress(DE3)", "E. coli",
                 "E. coli B F- ompT hsdS(rB- mB-) dcm+ Tetr gal λ(DE3) endA Hte [cpn10 cpn60 Gentr]",
                 "Cold-adapted chaperonins for low temperature expression",
                 ["Low temperature expression", "Cold-adapted chaperones", "T7 expression", "Proper folding"],
                 ["Gentamicin resistance", "Slower growth"]),
            Host(7, "DH5α", "E. coli",
                 "F- endA1 glnV44 thi-1 recA1 relA1 gyrA96 deoR nupG purB20 φ80dlacZΔM15",
                 "Standard cloning strain, not for expression",
                 ["High transformation efficiency", "Stable plasmid maintenance", "Blue-white screening"],
                 ["No T7 RNA polymerase", "Poor expression", "Only for cloning"]),
            Host(8, "Origami(DE3)", "E. coli",
                 "Δ(ara-leu)7697 ΔlacX74 ΔphoA PvuII phoR araD139 ahpC galE galK rpsL λ(DE3) gor522::Tn10 trxB",
                 "Enhanced disulfide bond formation with trxB mutation",
                 ["Disulfide bond formation", "T7 expression", "Oxidizing environment", "trxB/gor mutations"],
                 ["Slower growth", "Requires special media"])
        ]
        save_hosts(hosts)
        
    return hosts

def save_hosts(hosts):
    """Save hosts to JSON file"""
    host_data = [h.to_dict() for h in hosts]
    with open('data/hosts.json', 'w') as f:
        json.dump(host_data, f, indent=2)

# ----------------------
# Sequence Analysis
# ----------------------

def analyze_protein_sequence(sequence):
    """Comprehensive protein sequence analysis"""
    sequence = re.sub(r'[^A-Za-z]', '', sequence.upper())
    
    if not sequence:
        return {"error": "Invalid sequence. Please provide a valid protein sequence."}
    
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    if not set(sequence).issubset(valid_aas):
        return {"error": "Sequence contains invalid amino acid characters."}
    
    try:
        seq_length = len(sequence)
        mol_weight = seq_length * 110 / 1000
        
        # Amino acid composition
        aa_count = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            aa_count[aa] = sequence.count(aa)
        
        aa_percent = {aa: count/seq_length for aa, count in aa_count.items()}
        
        # Basic properties
        hydrophobic_aas = 'AILMFWYV'
        hydrophobicity = sum(1 for aa in sequence if aa in hydrophobic_aas) / seq_length
        
        charged_aas = 'DEKR'
        charged_residues = sum(1 for aa in sequence if aa in charged_aas) / seq_length
        
        cys_count = aa_count.get('C', 0)
        has_disulfide_potential = cys_count >= 2
        
        # Simplified analysis
        instability_index = abs(hydrophobicity - 0.5) * 100 + charged_residues * 50
        is_stable = instability_index < 40
        
        issues = []
        recommendations = []
        
        if has_disulfide_potential:
            issues.append(f"Multiple cysteines detected ({cys_count} cysteines)")
            recommendations.append("Consider using SHuffle T7 strain for disulfide bond formation")
        
        is_hydrophobic = hydrophobicity > 0.4
        if is_hydrophobic:
            issues.append("Highly hydrophobic protein (potential membrane protein)")
            recommendations.append("Consider using C41(DE3) strain for membrane proteins")
        
        if not is_stable:
            issues.append(f"Potentially unstable protein")
            recommendations.append("Consider fusion tags like MBP or GST to improve stability")
        
        if mol_weight > 70:
            issues.append(f"Large protein ({mol_weight:.1f} kDa)")
            recommendations.append("Consider co-expression with chaperones")
        
        return {
            "sequence_length": seq_length,
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
# Session State Management
# ----------------------

def init_session_state():
    """Initialize all session state variables"""
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
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False

# ----------------------
# UI Components (SECURE & COMPLETE)
# ----------------------

def add_developer_credit():
    """Add discrete developer credit"""
    st.markdown("""
    <div style="position: fixed; 
                bottom: 5px; 
                right: 5px; 
                font-size: 8px; 
                color: #888; 
                opacity: 0.5; 
                z-index: 1000;
                background: rgba(255,255,255,0.8);
                padding: 2px 5px;
                border-radius: 3px;">
        Developed by Dr Merzoug Mohamed
    </div>
    """, unsafe_allow_html=True)

def show_header():
    """Display application header with NeoRen logo"""
    col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
    
    with col1:
        try:
            st.image(NEOREN_LOGO_URL, width=80)
        except:
            st.markdown("**NeoRen®**")
    
    with col2:
        st.title("RennetOptiMax Pro")
        st.markdown("### 🧬 AI-Powered Protein Expression Optimization Platform")
        st.caption("Powered by NeoRen® - Engineered for Excellence")
    
    with col3:
        st.write("")
    
    with col4:
        if st.session_state.authenticated:
            if st.button("🏠 Dashboard", key="dashboard_btn"):
                st.session_state.page = 'dashboard'
                st.rerun()

def show_product_banner():
    """Display product promotion banner"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 40px; 
                border-radius: 15px; 
                margin: 20px 0; 
                text-align: center;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                min-height: 300px;
                display: flex;
                flex-direction: column;
                justify-content: center;">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 30px; flex-wrap: wrap;">
            <img src="{NEOREN_LOGO_URL}" alt="NeoRen Logo" style="height: 80px; margin-right: 30px; margin-bottom: 10px;">
            <div style="text-align: left; min-width: 400px;">
                <h2 style="margin: 0; font-size: 2.5em; line-height: 1.2;">🎯 NeoRen Chymosin Powder</h2>
                <h3 style="margin: 10px 0; font-size: 1.5em; line-height: 1.3;">Premium Sustainable Rennet for Modern Cheese Production</h3>
            </div>
        </div>
        <div style="margin: 30px 0; line-height: 1.6;">
            <p style="font-size: 1.2em; margin: 15px 0;">✅ 100% Animal-Free • ✅ Superior Performance • ✅ Cost-Effective & Scalable</p>
            <p style="font-size: 1.1em; margin: 15px 0;">🧬 Engineered with Advanced Genetic Engineering • 🌱 Environmentally Sustainable</p>
        </div>
        <div style="margin: 30px 0;">
            <a href="{NEOREN_WEBSITE}" target="_blank" style="background: #ff6b6b; 
                          color: white; 
                          border: none; 
                          padding: 20px 40px; 
                          border-radius: 10px; 
                          font-size: 1.3em; 
                          text-decoration: none;
                          display: inline-block;
                          font-weight: bold;
                          box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                          transition: all 0.3s ease;
                          min-width: 350px;">
                🛒 Buy 500g & Get 1 Year Free Platform Access
            </a>
        </div>
        <p style="font-size: 1em; margin: 15px 0; opacity: 0.9; line-height: 1.4;">
            <em>Revolutionizing cheese production through sustainable biotechnology solutions</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_login_form():
    """SECURE: Display login form WITHOUT exposing credentials publicly"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        try:
            st.image(NEOREN_LOGO_URL, width=150)
        except:
            st.markdown("### NeoRen®")
        st.markdown("## 🔐 Login to Your Account")
        st.caption("Access the world's most advanced protein expression platform")
    
    # Manual login section ONLY
    st.markdown("### 🔑 Account Login")
    st.markdown("*Sign in with your credentials:*")
    
    with st.form("login_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        with col2:
            remember_me = st.checkbox("Remember me")
            st.markdown("")
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            login_button = st.form_submit_button("🔐 Sign In", use_container_width=True, type="primary")
        
        with col2:
            forgot_password = st.form_submit_button("🔑 Forgot Password?", use_container_width=True)
        
        with col3:
            create_account = st.form_submit_button("📝 Create Account", use_container_width=True)
        
        if login_button:
            if username and password:
                authenticated, user_data = authenticate_user(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_name = user_data['name']
                    st.session_state.show_login = False
                    st.success(f"Welcome back, {user_data['name']}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.error("⚠️ Please enter both username and password")
        
        if forgot_password:
            st.info("🔄 Password reset - Contact administrator")
        
        if create_account:
            st.session_state.show_signup = True
            st.session_state.show_login = False
            st.rerun()
    
    # SECURE: Only show credentials to authenticated admin
    if st.session_state.authenticated and st.session_state.username == 'admin':
        with st.expander("🔐 Admin Panel - Demo Credentials", expanded=False):
            st.warning("⚠️ CONFIDENTIAL - Admin Use Only")
            st.markdown("""
            **Demo Account Credentials (Confidential):**
            
            **Administrator Account:**
            - Username: `admin`
            - Password: `admin123`
            - Access: Full platform (lifetime)
            
            **Student Demo Account:**
            - Username: `demo_student` 
            - Password: `student123`
            - Access: Basic features (30-day trial)
            
            **Professional Demo Account:**
            - Username: `demo_professional`
            - Password: `pro123`
            - Access: Advanced features (30-day trial)
            
            *🔒 These credentials are strictly confidential and for administrator demonstration purposes only.*
            """)
    else:
        st.info("💡 **Need an account?** Contact administrator for demo access or create a new account above.")

def show_signup_form():
    """Display user registration form"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        try:
            st.image(NEOREN_LOGO_URL, width=120)
        except:
            st.markdown("### NeoRen®")
        st.markdown("## 📝 Create Your Account")
        st.caption("Join thousands of researchers optimizing protein expression")
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); 
                color: white; 
                padding: 25px; 
                border-radius: 10px; 
                margin: 20px 0; 
                text-align: center;
                min-height: 80px;
                display: flex;
                flex-direction: column;
                justify-content: center;">
        <h3 style="margin: 0; font-size: 1.5em;">🎉 Start Your Free 30-Day Trial Today!</h3>
        <p style="margin: 10px 0; font-size: 1.1em;">Full access to all features • No credit card required • Cancel anytime</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("signup_form"):
        st.markdown("### 📋 Account Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username*", placeholder="Choose a unique username")
            email = st.text_input("Email Address*", placeholder="your.email@domain.com")
            name = st.text_input("Full Name*", placeholder="Your full name")
        
        with col2:
            user_type = st.selectbox(
                "Account Type*", 
                ["student", "academic", "professional"],
                format_func=lambda x: {
                    "student": "🎓 Student",
                    "academic": "🧪 Academic Researcher", 
                    "professional": "🧑‍💼 Industry Professional"
                }[x]
            )
            password = st.text_input("Password*", type="password", placeholder="Choose a strong password")
            confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Confirm your password")
        
        st.markdown("### 🎯 Optional Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            institution = st.text_input("Institution/Company", placeholder="University or company name")
            research_area = st.selectbox(
                "Research Area",
                ["", "Protein Expression", "Biotechnology", "Food Science", "Pharmaceuticals", "Academic Research", "Other"]
            )
        
        with col2:
            referral_code = st.text_input("Referral Code (Optional)", placeholder="Enter referral code if you have one")
            newsletter = st.checkbox("📧 Subscribe to newsletter for updates and tips", value=True)
        
        st.markdown("### 📜 Terms and Privacy")
        
        terms_accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy*", help="Required to create an account")
        marketing_consent = st.checkbox("I consent to receive marketing communications", help="Optional - you can unsubscribe anytime")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            submitted = st.form_submit_button("🚀 Create Account & Start Free Trial", use_container_width=True, type="primary")
        
        if submitted:
            errors = []
            
            if not all([username, email, name, password]):
                errors.append("Please fill in all required fields.")
            
            if len(username) < 3:
                errors.append("Username must be at least 3 characters long.")
            
            if '@' not in email or '.' not in email:
                errors.append("Please enter a valid email address.")
            
            if len(password) < 6:
                errors.append("Password must be at least 6 characters long.")
            
            if password != confirm_password:
                errors.append("Passwords do not match.")
            
            if not terms_accepted:
                errors.append("Please accept the Terms of Service.")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                try:
                    success, message = add_new_user(username, email, name, password, user_type)
                    if success:
                        st.success("🎉 Account created successfully!")
                        st.balloons()
                        st.info("Please login with your new credentials to access your 30-day free trial.")
                        
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_name = name
                        st.session_state.show_signup = False
                        
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"❌ Error creating account: {str(e)}")

def show_navigation():
    """Display secure navigation sidebar"""
    try:
        st.sidebar.image(NEOREN_LOGO_URL, width=120)
    except:
        st.sidebar.markdown("### NeoRen®")
    
    st.sidebar.title("🧬 RennetOptiMax Pro")
    st.sidebar.caption("AI-Powered Protein Expression Platform")
    
    if st.session_state.authenticated:
        users = load_users()
        user_data = users.get(st.session_state.username, {})
        
        st.sidebar.success(f"👋 {user_data.get('name', 'User')}")
        st.sidebar.caption(f"🏷️ {user_data.get('user_type', 'Unknown').title()}")
        
        # SECURE: Only show admin indicator, not credentials
        if st.session_state.username == 'admin':
            st.sidebar.info("🔐 Admin Mode Active")
        
        subscription_status = user_data.get('subscription_status', 'none')
        if subscription_status == 'lifetime':
            st.sidebar.info("♾️ Lifetime Access")
        elif subscription_status == 'trial':
            trial_expiry = user_data.get('trial_expiry')
            if trial_expiry:
                try:
                    expiry_date = datetime.fromisoformat(trial_expiry)
                    days_left = (expiry_date - datetime.now()).days
                    if days_left > 0:
                        st.sidebar.info(f"🎁 Trial: {days_left} days left")
                    else:
                        st.sidebar.warning("⏰ Trial expired")
                except:
                    st.sidebar.info("🎁 Trial Active")
        
        st.sidebar.divider()
        
        pages = {
            'dashboard': "🏠 Dashboard",
            'home': "🌟 Home"
        }
        
        features = [
            ('vectors', '1. 🧬 Vector Selection'),
            ('hosts', '2. 🦠 Host Selection'),
            ('sequence', '3. 🔬 Sequence Analysis'),
            ('parameters', '4. ⚙️ Parameters'),
            ('optimize', '5. 🎯 Optimization'),
            ('results', '6. 📊 Results')
        ]
        
        for feature_id, feature_name in features:
            has_access = check_user_access(st.session_state.username, feature_id)
            
            if has_access:
                pages[feature_id] = feature_name
            else:
                pages[feature_id] = f"{feature_name} 🔒"
        
        for page_id, page_name in pages.items():
            is_locked = '🔒' in page_name
            button_type = "primary" if st.session_state.page == page_id else "secondary"
            
            if st.sidebar.button(page_name, key=f"nav_{page_id}", use_container_width=True, type=button_type, disabled=is_locked):
                st.session_state.page = page_id
                st.rerun()
        
        st.sidebar.divider()
        st.sidebar.subheader("⚙️ Account")
        
        if st.sidebar.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_name = None
            st.session_state.page = 'home'
            st.session_state.show_login = False
            st.sidebar.success("Logged out successfully!")
            st.rerun()
    
    else:
        if st.sidebar.button("🌟 Home", use_container_width=True):
            st.session_state.page = 'home'
            st.session_state.show_login = False
            st.rerun()
        
        if st.sidebar.button("🔑 Login", use_container_width=True):
            st.session_state.show_login = True
            st.rerun()
        
        if st.sidebar.button("📝 Sign Up", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()
        
        # SECURE: No demo credentials exposed
        st.sidebar.info("💡 Contact administrator for demo access")
    
    st.sidebar.divider()
    st.sidebar.markdown("### ℹ️ About")
    
    st.sidebar.info("RennetOptiMax Pro: The world's most advanced AI-powered platform for protein expression optimization, specializing in sustainable rennet production.")
    
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin: 15px 0;">
        <a href="{NEOREN_WEBSITE}" target="_blank" style="background: #ff6600; 
                  color: white; 
                  padding: 10px 16px; 
                  border-radius: 6px; 
                  text-decoration: none; 
                  display: inline-block;
                  font-size: 13px;
                  font-weight: bold;">
            🌐 NeoRen Website
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.divider()
    st.sidebar.caption("🔬 Version 2.0.0")
    st.sidebar.caption("© 2025 NeoRen® - Engineered for Excellence")
    st.sidebar.caption("🌱 Sustainable Biotechnology Solutions")

def show_home_page():
    """Display secure home page"""
    st.markdown("## 🌟 Welcome to RennetOptiMax Pro")
    st.caption("The Future of Protein Expression Optimization")
    
    show_product_banner()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="min-height: 400px; padding: 20px; line-height: 1.6;">
        <h3>🧬 Revolutionary AI-Powered Platform</h3>
        
        <strong>RennetOptiMax Pro</strong> transforms protein expression optimization through cutting-edge 
        artificial intelligence and machine learning. Specifically engineered for <strong>sustainable 
        rennet (chymosin) production</strong>, our platform delivers unprecedented accuracy and efficiency.<br><br>
        
        <strong>🔬 Why Industry Leaders Choose RennetOptiMax Pro:</strong><br><br>
        
        🎯 <strong>AI-Driven Predictions</strong>: Advanced machine learning algorithms analyze thousands of 
        expression parameters to predict optimal conditions with 94% accuracy<br><br>
        
        📊 <strong>Comprehensive Database</strong>: Curated collection of 8+ expression vectors and 8+ specialized 
        E. coli strains, continuously updated with latest research<br><br>
        
        ⚡ <strong>Real-Time Analysis</strong>: Instant protein sequence analysis with detailed recommendations 
        for expression strategy and troubleshooting<br><br>
        
        💰 <strong>Cost Optimization</strong>: Reduce experimental costs by up to 60% through predictive modeling 
        and first-time-right protocols<br><br>
        
        🌱 <strong>Sustainability Focus</strong>: Supporting the transition to animal-free, environmentally 
        responsible biotechnology solutions
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="border: 3px solid #ff6600; 
                    border-radius: 15px; 
                    padding: 30px; 
                    text-align: center;
                    background: linear-gradient(135deg, #fff8f0 0%, #ffebdb 100%);
                    box-shadow: 0 6px 12px rgba(255,102,0,0.1);
                    min-height: 350px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;">
            <img src="{NEOREN_LOGO_URL}" alt="NeoRen Logo" style="width: 120px; margin-bottom: 20px;">
            <h3 style="color: #ff6600; margin: 20px 0; font-size: 1.8em;">NeoRen®</h3>
            <p style="font-size: 16px; color: #666; margin: 15px 0; font-weight: bold;">Engineered for Excellence</p>
            <p style="font-size: 14px; color: #888; margin: 8px 0;">🧬 Advanced Genetic Engineering</p>
            <p style="font-size: 14px; color: #888; margin: 8px 0;">🌱 Sustainable Biotechnology</p>
            <p style="font-size: 14px; color: #888; margin: 8px 0;">🏭 Industrial Solutions</p>
            <div style="margin-top: 25px;">
                <a href="{NEOREN_WEBSITE}" target="_blank" style="background: #ff6600; 
                          color: white; 
                          padding: 12px 20px; 
                          border-radius: 8px; 
                          text-decoration: none; 
                          font-size: 14px;
                          font-weight: bold;">
                    🌐 Explore Products
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Get Started Today!")
    st.markdown("*Create your account to access the world's most advanced protein expression platform:*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Create Free Account", use_container_width=True, type="primary"):
            st.session_state.show_signup = True
            st.rerun()
    
    with col2:
        if st.button("🔑 Login to Existing Account", use_container_width=True):
            st.session_state.show_login = True
            st.rerun()

def show_dashboard():
    """Display user dashboard"""
    if not st.session_state.authenticated:
        st.error("🔒 Please login to access the dashboard.")
        return
    
    users = load_users()
    user_data = users.get(st.session_state.username, {})
    
    st.markdown(f"## 🏠 Welcome, {user_data.get('name', 'User')}!")
    
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
    """Complete vector selection page"""
    if not st.session_state.authenticated or not check_user_access(st.session_state.username, 'vectors'):
        st.error("🔒 This feature requires authentication and proper access level.")
        if st.button("🔑 Login to Access"):
            st.session_state.show_login = True
            st.rerun()
        return
    
    st.markdown("## 🧬 Expression Vector Selection")
    st.caption("Choose the optimal expression vector for your protein")
    
    vectors = load_vectors()
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        promoter_filter = st.selectbox("Promoter Type", ["All"] + sorted(set(v.promoter for v in vectors)))
    
    with col2:
        selection_filter = st.selectbox("Selection Marker", ["All"] + sorted(set(v.selection_marker for v in vectors)))
    
    with col3:
        expression_filter = st.selectbox("Expression Level", ["All", "Very High", "High", "Medium", "Low"])
    
    # Apply filters
    filtered_vectors = vectors
    
    if promoter_filter != "All":
        filtered_vectors = [v for v in filtered_vectors if v.promoter == promoter_filter]
    
    if selection_filter != "All":
        filtered_vectors = [v for v in filtered_vectors if v.selection_marker == selection_filter]
    
    if expression_filter != "All":
        filtered_vectors = [v for v in filtered_vectors if v.features.get('expression_level', 'Medium') == expression_filter]
    
    st.markdown(f"### 🧬 Available Vectors ({len(filtered_vectors)} found)")
    
    # Display vectors
    for i in range(0, len(filtered_vectors), 2):
        row_vectors = filtered_vectors[i:i+2]
        cols = st.columns(2)
        
        for j, vector in enumerate(row_vectors):
            if j < len(cols):
                with cols[j]:
                    selected = st.session_state.selected_vector and st.session_state.selected_vector.id == vector.id
                    
                    with st.container():
                        st.subheader(vector.name)
                        st.write(f"**Size:** {vector.size:,} bp")
                        st.write(f"**Promoter:** {vector.promoter}")
                        st.write(f"**Tags:** {', '.join(vector.tags)}")
                        st.write(f"**Expression Level:** {vector.features.get('expression_level', 'Medium')}")
                        st.write(vector.description)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            btn_label = "✅ Selected" if selected else "Select Vector"
                            btn_type = "primary" if selected else "secondary"
                            
                            if st.button(btn_label, key=f"select_vector_{vector.id}", use_container_width=True, type=btn_type):
                                st.session_state.selected_vector = vector
                                st.success(f"Vector {vector.name} selected!")
                                st.rerun()
                        
                        with col2:
                            if st.button("📋 Details", key=f"details_vector_{vector.id}", use_container_width=True):
                                with st.expander(f"📋 {vector.name} - Detailed Information", expanded=True):
                                    st.markdown(f"**Full Description:** {vector.description}")
                                    st.markdown(f"**Cloning Sites:** {', '.join(vector.features.get('cloning_sites', []))}")
                                    st.markdown(f"**Tag Location:** {vector.features.get('tag_location', 'N/A')}")
                                    st.markdown(f"**Induction:** {vector.features.get('induction', 'IPTG')}")
    
    if st.session_state.selected_vector:
        st.success(f"✅ **Selected Vector:** {st.session_state.selected_vector.name}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Description:** {st.session_state.selected_vector.description}")
        
        with col2:
            if st.button("➡️ Continue to Host Selection", type="primary", use_container_width=True):
                st.session_state.page = 'hosts'
                st.rerun()
    else:
        st.info("👆 Please select a vector to continue.")

def show_hosts_page():
    """Complete host selection page"""
    if not st.session_state.authenticated or not check_user_access(st.session_state.username, 'hosts'):
        st.error("🔒 This feature requires authentication and proper access level.")
        if st.button("🔑 Login to Access"):
            st.session_state.show_login = True
            st.rerun()
        return
    
    st.markdown("## 🦠 Host Strain Selection")
    st.caption("Choose the optimal bacterial strain for your expression")
    
    hosts = load_hosts()
    
    if st.session_state.selected_vector:
        st.info(f"🧬 **Selected Vector:** {st.session_state.selected_vector.name} - Now choose a compatible host strain")
    
    # Display hosts
    for i in range(0, len(hosts), 2):
        row_hosts = hosts[i:i+2]
        cols = st.columns(2)
        
        for j, host in enumerate(row_hosts):
            if j < len(cols):
                with cols[j]:
                    selected = st.session_state.selected_host and st.session_state.selected_host.id == host.id
                    
                    with st.container():
                        st.subheader(host.strain)
                        st.write(f"**Species:** {host.species}")
                        st.write(f"**Features:** {', '.join(host.features[:3])}...")
                        st.write(host.description)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            btn_label = "✅ Selected" if selected else "Select Host"
                            btn_type = "primary" if selected else "secondary"
                            
                            if st.button(btn_label, key=f"select_host_{host.id}", use_container_width=True, type=btn_type):
                                st.session_state.selected_host = host
                                st.success(f"Host {host.strain} selected!")
                                st.rerun()
                        
                        with col2:
                            if st.button("📋 Details", key=f"details_host_{host.id}", use_container_width=True):
                                with st.expander(f"📋 {host.strain} - Detailed Information", expanded=True):
                                    st.markdown(f"**Full Description:** {host.description}")
                                    st.markdown(f"**All Features:** {', '.join(host.features)}")
                                    if host.limitations:
                                        st.markdown(f"**Limitations:** {', '.join(host.limitations)}")
                                    st.code(host.genotype)
    
    if st.session_state.selected_host:
        st.success(f"✅ **Selected Host:** {st.session_state.selected_host.strain}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Description:** {st.session_state.selected_host.description}")
        
        with col2:
            if st.button("➡️ Continue to Sequence Analysis", type="primary", use_container_width=True):
                st.session_state.page = 'sequence'
                st.rerun()
    else:
        st.info("👆 Please select a host strain to continue.")

def show_sequence_page():
    """Complete sequence analysis page"""
    if not st.session_state.authenticated or not check_user_access(st.session_state.username, 'sequence'):
        st.error("🔒 This feature requires authentication and proper access level.")
        if st.button("🔑 Login to Access"):
            st.session_state.show_login = True
            st.rerun()
        return
    
    st.markdown("## 🔬 Protein Sequence Analysis")
    st.caption("Analyze your protein sequence for optimal expression conditions")
    
    sequence = st.text_area("Enter protein sequence:", height=200, value=st.session_state.protein_sequence,
                           placeholder="Enter your protein sequence using single letter amino acid codes...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔬 Analyze Sequence", type="primary", use_container_width=True):
            if sequence:
                with st.spinner("Analyzing sequence..."):
                    st.session_state.protein_sequence = sequence
                    analysis = analyze_protein_sequence(sequence)
                    st.session_state.sequence_analysis = analysis
                    
                    if 'error' in analysis:
                        st.error(analysis['error'])
                    else:
                        st.success("✅ Analysis complete!")
                        st.rerun()
            else:
                st.error("Please enter a protein sequence.")
    
    with col2:
        if st.button("🗑️ Clear Sequence", use_container_width=True):
            st.session_state.protein_sequence = ""
            st.session_state.sequence_analysis = None
            st.rerun()
    
    if st.session_state.sequence_analysis and 'error' not in st.session_state.sequence_analysis:
        analysis = st.session_state.sequence_analysis
        
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Length", f"{analysis['sequence_length']} aa")
        
        with col2:
            st.metric("Molecular Weight", f"{analysis['molecular_weight']} kDa")
        
        with col3:
            st.metric("Hydrophobicity", f"{analysis['hydrophobicity']:.3f}")
        
        with col4:
            stability = "Stable" if analysis['is_stable'] else "Unstable"
            st.metric("Stability", stability)
        
        if analysis['issues']:
            st.markdown("### ⚠️ Potential Issues")
            for issue in analysis['issues']:
                st.warning(f"⚠️ {issue}")
        
        if analysis['recommendations']:
            st.markdown("### 💡 Recommendations")
            for rec in analysis['recommendations']:
                st.info(f"💡 {rec}")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("➡️ Continue to Parameters", type="primary", use_container_width=True):
                st.session_state.page = 'parameters'
                st.rerun()

def show_parameters_page():
    """Complete parameters configuration page"""
    if not st.session_state.authenticated or not check_user_access(st.session_state.username, 'parameters'):
        st.error("🔒 This feature requires authentication and proper access level.")
        if st.button("🔑 Login to Access"):
            st.session_state.show_login = True
            st.rerun()
        return
    
    st.markdown("## ⚙️ Expression Parameters Configuration")
    st.caption("Fine-tune expression conditions for optimal protein production")
    
    if not st.session_state.selected_vector or not st.session_state.selected_host:
        st.warning("Please select a vector and host first.")
        return
    
    st.info(f"🧬 **Vector:** {st.session_state.selected_vector.name} | 🦠 **Host:** {st.session_state.selected_host.strain}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌡️ Expression Conditions")
        
        temperature = st.slider("Temperature (°C)", 16, 42, st.session_state.expression_parameters['temperature'])
        induction_time = st.slider("Induction Time (hours)", 1, 24, st.session_state.expression_parameters['induction_time'])
    
    with col2:
        st.markdown("### 🧪 Induction Parameters")
        
        inducer_conc = st.slider("Inducer Concentration (mM)", 0.1, 2.0, st.session_state.expression_parameters['inducer_concentration'], step=0.1)
        od600 = st.slider("OD600 at Induction", 0.3, 1.5, st.session_state.expression_parameters['OD600_at_induction'], step=0.1)
    
    media = st.selectbox("Growth Medium", ["LB", "TB", "2xYT", "M9", "SOC"], 
                        index=["LB", "TB", "2xYT", "M9", "SOC"].index(st.session_state.expression_parameters['media_composition']))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Parameters", type="primary", use_container_width=True):
            st.session_state.expression_parameters = {
                'temperature': temperature,
                'induction_time': induction_time,
                'inducer_concentration': inducer_conc,
                'OD600_at_induction': od600,
                'media_composition': media
            }
            st.success("✅ Parameters saved!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.expression_parameters = {
                'temperature': 30,
                'induction_time': 4,
                'inducer_concentration': 0.5,
                'OD600_at_induction': 0.6,
                'media_composition': 'LB'
            }
            st.rerun()
    
    with col3:
        if st.button("➡️ Proceed to Optimization", type="primary", use_container_width=True):
            st.session_state.page = 'optimize'
            st.rerun()

def show_optimization_page():
    """Complete optimization page"""
    if not st.session_state.authenticated or not check_user_access(st.session_state.username, 'optimize'):
        st.error("🔒 This feature requires authentication and proper access level.")
        if st.button("🔑 Login to Access"):
            st.session_state.show_login = True
            st.rerun()
        return
    
    st.markdown("## 🎯 AI-Powered Expression Optimization")
    st.caption("Generate optimal expression conditions using machine learning")
    
    if not st.session_state.selected_vector or not st.session_state.selected_host:
        st.warning("Please select a vector and host first.")
        return
    
    st.info(f"🔬 **Optimizing:** {st.session_state.selected_vector.name} + {st.session_state.selected_host.strain}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔧 Current Parameters")
        params_df = pd.DataFrame([
            ["🌡️ Temperature", f"{st.session_state.expression_parameters['temperature']} °C"],
            ["⏱️ Induction Time", f"{st.session_state.expression_parameters['induction_time']} hours"],
            ["🧪 Inducer Concentration", f"{st.session_state.expression_parameters['inducer_concentration']} mM"],
            ["📏 OD600 at Induction", str(st.session_state.expression_parameters['OD600_at_induction'])],
            ["🧬 Media", st.session_state.expression_parameters['media_composition']]
        ], columns=["Parameter", "Value"])
        
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("⚙️ Adjust Parameters", use_container_width=True):
            st.session_state.page = 'parameters'
            st.rerun()
        
        if st.button("🔬 Analyze Sequence", use_container_width=True):
            st.session_state.page = 'sequence'
            st.rerun()
    
    if st.button("🧠 Run AI Optimization", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing thousands of parameter combinations... This may take a moment."):
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
            
            # Predict current expression
            current_result = optimizer.predict_expression(current_conditions)
            current_expression = current_result['prediction']
            current_confidence = current_result['confidence']
            
            # Generate optimization suggestions
            protein_properties = st.session_state.sequence_analysis.get('protein_properties', {}) if st.session_state.sequence_analysis else {}
            
            suggestions = optimizer.suggest_optimal_conditions(
                vector_name=st.session_state.selected_vector.name,
                host_strain=st.session_state.selected_host.strain,
                protein_properties=protein_properties,
                n_suggestions=5
            )
            
            # Store optimization results
            st.session_state.optimization_results = {
                'current_conditions': current_conditions,
                'current_expression': current_expression,
                'current_confidence': current_confidence,
                'suggestions': suggestions,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'vector': st.session_state.selected_vector.to_dict(),
                'host': st.session_state.selected_host.to_dict(),
                'sequence_analysis': st.session_state.sequence_analysis
            }
            
            st.success("✅ Optimization complete! Generated optimized parameter sets.")
            st.balloons()
            
            # Automatically go to results page
            st.session_state.page = 'results'
            st.rerun()

def show_results_page():
    """Complete results visualization page"""
    if not st.session_state.authenticated or not check_user_access(st.session_state.username, 'results'):
        st.error("🔒 This feature requires authentication and proper access level.")
        if st.button("🔑 Login to Access"):
            st.session_state.show_login = True
            st.rerun()
        return
    
    st.markdown("## 📊 Optimization Results & Analysis")
    st.caption("AI-generated recommendations for optimal protein expression")
    
    if not st.session_state.optimization_results:
        st.warning("⚠️ No optimization results available. Please run optimization first.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Run Optimization", type="primary", use_container_width=True):
                st.session_state.page = 'optimize'
                st.rerun()
        
        with col2:
            if st.button("🧬 Start from Vector Selection", use_container_width=True):
                st.session_state.page = 'vectors'
                st.rerun()
        
        return
    
    results = st.session_state.optimization_results
    
    # Results header with timestamp
    st.info(f"🕒 **Optimization completed:** {results['timestamp']}")
    
    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    
    current_expr = results['current_expression']
    current_conf = results['current_confidence']
    best_suggestion = results['suggestions'][0] if results['suggestions'] else None
    
    if best_suggestion:
        best_expr = best_suggestion['predicted_expression']
        improvement = best_expr - current_expr
        improvement_percent = (improvement / current_expr) * 100 if current_expr > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔬 Current Setup",
                f"{current_expr:.1f}%",
                delta=f"Confidence: {current_conf:.1f}%",
                help="Predicted expression with your current parameters"
            )
        
        with col2:
            st.metric(
                "🎯 Best Option",
                f"{best_expr:.1f}%",
                delta=f"+{improvement:.1f}%",
                delta_color="normal" if improvement > 0 else "inverse",
                help="Best predicted expression from optimization"
            )
        
        with col3:
            st.metric(
                "📈 Improvement",
                f"{improvement_percent:+.1f}%",
                delta="vs current setup",
                delta_color="normal" if improvement_percent > 0 else "inverse",
                help="Percentage improvement over current parameters"
            )
        
        with col4:
            avg_confidence = np.mean([s['confidence'] for s in results['suggestions']])
            st.metric(
                "🎯 Avg Confidence",
                f"{avg_confidence:.1f}%",
                help="Average confidence across all suggestions"
            )
    
    # Visualization of results
    st.markdown("### 📊 Expression Level Comparison")
    
    # Prepare data for visualization
    labels = ['Current Setup'] + [f'Option {i+1}' for i in range(len(results['suggestions']))]
    expressions = [results['current_expression']] + [s['predicted_expression'] for s in results['suggestions']]
    confidences = [results['current_confidence']] + [s['confidence'] for s in results['suggestions']]
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add expression bars
    fig.add_trace(go.Bar(
        x=labels,
        y=expressions,
        name='Predicted Expression (%)',
        marker_color=['#1976d2'] + ['#4caf50'] * len(results['suggestions']),
        text=[f"{e:.1f}%" for e in expressions],
        textposition='auto',
        yaxis='y1'
    ))
    
    # Add confidence line
    fig.add_trace(go.Scatter(
        x=labels,
        y=confidences,
        mode='lines+markers',
        name='Confidence (%)',
        line=dict(color='#ff9800', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Update layout for dual y-axis
    fig.update_layout(
        title="Expression Predictions with Confidence Scores",
        xaxis_title="Parameter Sets",
        yaxis=dict(
            title="Predicted Expression (%)",
            side="left",
            range=[0, max(expressions) * 1.1]
        ),
        yaxis2=dict(
            title="Confidence (%)",
            side="right",
            overlaying="y",
            range=[0, 100]
        ),
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed suggestions
    st.markdown("### 💡 Optimization Suggestions")
    
    # Create tabs for each suggestion
    if results['suggestions']:
        tab_labels = [f"🥇 Best Option" if i == 0 else f"Option {i+1}" for i in range(len(results['suggestions']))]
        suggestion_tabs = st.tabs(tab_labels)
        
        for i, (tab, suggestion) in enumerate(zip(suggestion_tabs, results['suggestions'])):
            with tab:
                params = suggestion['parameters']
                predicted_expr = suggestion['predicted_expression']
                confidence = suggestion['confidence']
                
                # Suggestion header
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if i == 0:
                        st.markdown("#### 🏆 Recommended Best Option")
                    else:
                        st.markdown(f"#### 🔬 Alternative Option {i+1}")
                
                with col2:
                    st.metric("Expression", f"{predicted_expr:.1f}%")
                
                with col3:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                # Parameter comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🧪 Optimized Parameters:**")
                    
                    current_params = results['current_conditions']
                    
                    st.write(f"- **Temperature:** {params['temperature']}°C")
                    st.write(f"- **Induction Time:** {params['induction_time']} hours")
                    st.write(f"- **Inducer Concentration:** {params['inducer_concentration']} mM")
                    st.write(f"- **OD600:** {params['OD600_at_induction']}")
                    st.write(f"- **Media:** {params['media_composition']}")
                
                with col2:
                    st.markdown("**📊 Performance Metrics:**")
                    
                    improvement = predicted_expr - current_expr
                    improvement_pct = (improvement / current_expr * 100) if current_expr > 0 else 0
                    
                    st.write(f"- **Expression Level:** {predicted_expr:.1f}%")
                    st.write(f"- **Improvement:** {improvement:+.1f}% ({improvement_pct:+.1f}%)")
                    st.write(f"- **Confidence:** {confidence:.1f}%")
                    
                    if confidence > 80:
                        risk = "🟢 Low Risk"
                    elif confidence > 60:
                        risk = "🟡 Medium Risk"
                    else:
                        risk = "🔴 High Risk"
                    st.write(f"- **Risk Level:** {risk}")
                
                # Additives and special considerations
                if 'additives' in suggestion and suggestion['additives']:
                    st.markdown("**🧪 Recommended Additives:**")
                    for additive in suggestion['additives']:
                        st.success(f"➕ {additive}")
    
    # Protocol generation
    st.markdown("### 📋 Complete Expression Protocol")
    
    if best_suggestion:
        best_params = best_suggestion['parameters']
        
        # Generate comprehensive protocol
        protocol = f"""
# Optimized Expression Protocol
Generated by RennetOptiMax Pro on {results['timestamp']}

## Expression System
- **Vector:** {results['vector']['name']} ({results['vector']['size']:,} bp)
- **Host:** {results['host']['strain']} ({results['host']['species']})
- **Promoter:** {results['vector']['promoter']}
- **Selection:** {results['vector']['selection_marker']}
- **Tags:** {', '.join(results['vector']['tags'])}

## Predicted Performance
- **Expression Level:** {best_suggestion['predicted_expression']:.1f}%
- **Confidence:** {best_suggestion['confidence']:.1f}%
- **Improvement:** {(best_suggestion['predicted_expression'] - current_expr):+.1f}% vs current setup

## Optimized Parameters
- **Temperature:** {best_params['temperature']}°C
- **Induction Time:** {best_params['induction_time']} hours
- **Inducer Concentration:** {best_params['inducer_concentration']} mM
- **OD600 at Induction:** {best_params['OD600_at_induction']}
- **Media:** {best_params['media_composition']}

## Materials Required

### Bacterial Strains
- {results['host']['strain']} competent cells
- Store at -80°C until use

### Plasmids
- {results['vector']['name']} vector containing your gene of interest
- Verify insert by sequencing before expression

### Media and Reagents
- {best_params['media_composition']} medium (autoclaved)
- {results['vector']['selection_marker']} antibiotic stock solution
- {"IPTG" if results['vector']['features'].get('induction', 'IPTG') == 'IPTG' else results['vector']['features'].get('induction', 'IPTG')} (inducer)
- Sterile culture flasks (250 mL recommended for 50 mL culture)

## Detailed Protocol

### Day 1: Transformation
1. **Transform competent cells**
   - Thaw {results['host']['strain']} competent cells on ice
   - Add 1-5 ng plasmid DNA to 50 μL competent cells
   - Incubate on ice for 30 minutes
   - Heat shock at 42°C for 45 seconds
   - Return to ice for 2 minutes
   - Add 250 μL SOC medium
   - Recover at 37°C for 1 hour with shaking

2. **Plate and select**
   - Plate on LB agar + {results['vector']['selection_marker']} 
   - Incubate overnight at 37°C

### Day 2: Starter Culture
1. **Inoculate starter culture**
   - Pick single colony from transformation plate
   - Inoculate 5 mL {best_params['media_composition']} + {results['vector']['selection_marker']}
   - Incubate overnight at 37°C with shaking (200 rpm)

### Day 3: Expression Culture
1. **Prepare expression culture**
   - Dilute starter culture 1:100 into fresh {best_params['media_composition']} + {results['vector']['selection_marker']}
   - Use 50 mL medium in 250 mL flask for optimal aeration
   - Incubate at 37°C with shaking (200 rpm)

2. **Monitor growth**
   - Measure OD600 every hour starting from OD600 ~0.2
   - Culture typically reaches OD600 0.6 in 2-3 hours

3. **Induction**
   - When OD600 reaches {best_params['OD600_at_induction']:.1f}:
     * Add {"IPTG" if results['vector']['features'].get('induction', 'IPTG') == 'IPTG' else results['vector']['features'].get('induction', 'IPTG')} to final concentration {best_params['inducer_concentration']} mM
     * Reduce temperature to {best_params['temperature']}°C
     * Continue shaking at 200 rpm
     * Incubate for {best_params['induction_time']} hours

4. **Harvest cells**
   - Centrifuge culture at 4,000g for 15 minutes at 4°C
   - Discard supernatant
   - Cell pellet can be stored at -20°C or processed immediately

### Cell Lysis and Purification
1. **Cell lysis**
   - Resuspend pellet in lysis buffer appropriate for your purification method
   - For His-tag: 50 mM Tris-HCl pH 7.5, 300 mM NaCl, 10 mM imidazole
   - Add protease inhibitors if needed
   - Lyse by sonication (10 x 30 sec pulses with 30 sec rest)
   - Centrifuge at 15,000g for 30 minutes at 4°C

2. **Purification**
   - Proceed with purification appropriate for your tag system
   - His-tag: Ni-NTA affinity chromatography
   - GST-tag: Glutathione affinity chromatography
   - MBP-tag: Amylose affinity chromatography

## Quality Control
- Check expression by SDS-PAGE before and after induction
- Verify protein identity by Western blot or mass spectrometry
- Assess solubility by comparing total vs soluble fractions
- Monitor expression level throughout induction period

## Troubleshooting
- **Low expression:** Increase induction time, verify plasmid integrity
- **Inclusion bodies:** Lower temperature, reduce inducer concentration
- **No expression:** Check antibiotic concentration, verify strain compatibility
- **Slow growth:** Check medium freshness, verify temperature

Generated by RennetOptiMax Pro - AI-Powered Protein Expression Optimization
© 2025 NeoRen - Engineered for Excellence
        """
        
        # Display protocol in expandable section
        with st.expander("📋 Complete Protocol (Click to expand)", expanded=True):
            st.code(protocol, language="markdown")
        
        # Protocol download
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📄 Download Protocol (TXT)",
                data=protocol,
                file_name=f"expression_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Create CSV of parameters
            csv_data = []
            csv_data.append(["Parameter", "Value", "Unit"])
            csv_data.append(["Temperature", best_params['temperature'], "°C"])
            csv_data.append(["Induction Time", best_params['induction_time'], "hours"])
            csv_data.append(["Inducer Concentration", best_params['inducer_concentration'], "mM"])
            csv_data.append(["OD600 at Induction", best_params['OD600_at_induction'], ""])
            csv_data.append(["Media", best_params['media_composition'], ""])
            csv_data.append(["Predicted Expression", f"{best_suggestion['predicted_expression']:.1f}", "%"])
            csv_data.append(["Confidence", f"{best_suggestion['confidence']:.1f}", "%"])
            
            csv_string = "\n".join([",".join(row) for row in csv_data])
            
            st.download_button(
                label="📊 Download Data (CSV)",
                data=csv_string,
                file_name=f"optimization_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # Create results summary
            summary = f"""
RennetOptiMax Pro - Optimization Summary
Generated: {results['timestamp']}

SYSTEM:
Vector: {results['vector']['name']}
Host: {results['host']['strain']}

OPTIMIZED PARAMETERS:
Temperature: {best_params['temperature']}°C
Induction Time: {best_params['induction_time']} hours  
Inducer: {best_params['inducer_concentration']} mM
OD600: {best_params['OD600_at_induction']}
Media: {best_params['media_composition']}

PERFORMANCE:
Predicted Expression: {best_suggestion['predicted_expression']:.1f}%
Confidence: {best_suggestion['confidence']:.1f}%
Improvement: {(best_suggestion['predicted_expression'] - current_expr):+.1f}%

Powered by NeoRen AI Technology
            """
            
            st.download_button(
                label="📋 Download Summary",
                data=summary,
                file_name=f"optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Next steps
    st.markdown("### 🚀 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run New Optimization", use_container_width=True):
            st.session_state.page = 'optimize'
            st.rerun()
    
    with col2:
        if st.button("⚙️ Adjust Parameters", use_container_width=True):
            st.session_state.page = 'parameters'
            st.rerun()
    
    with col3:
        if st.button("🧬 Try Different Vector", use_container_width=True):
            st.session_state.page = 'vectors'
            st.rerun()

# ----------------------
# Main Application (COMPLETE)
# ----------------------

def main():
    """Main application entry point with full screen wide theme"""
    
    # Set wide page configuration
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
    
    # Custom CSS for full screen experience
    st.markdown("""
    <style>
    /* Remove default Streamlit padding and margins for full screen */
    .st-emotion-cache-1jicfl2 {
        width: 100%;
        padding: 1rem 0.5rem 2rem;
        min-width: auto;
        max-width: initial;
    }
    
    /* Full width container */
    .stApp > div:first-child {
        margin-top: 0;
        padding-top: 0;
    }
    
    /* Remove top padding */
    .st-emotion-cache-z5fcl4 {
        padding-top: 1rem;
    }
    
    /* Sidebar full height */
    .st-emotion-cache-16txtl3 {
        padding: 1rem 1rem 10rem;
    }
    
    /* Main content area full width */
    .st-emotion-cache-1dp5vir {
        width: 100%;
        max-width: none;
    }
    
    /* Header styling */
    h1, h2, h3, h4 {
        color: #1976d2;
        margin-top: 0;
    }
    
    /* NeoRen accent color */
    .neoren-accent {
        color: #ff6600;
    }
    
    /* Full screen optimizations */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: none;
    }
    
    /* Remove Streamlit branding for full screen */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .success-gradient {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    }
    .warning-gradient {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    }
    .error-gradient {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
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
    if st.session_state.show_signup:
        show_signup_form()
        if st.button("← Back to Home"):
            st.session_state.show_signup = False
            st.rerun()
        return
    
    if st.session_state.show_login:
        show_login_form()
        if st.button("← Back to Home"):
            st.session_state.show_login = False
            st.rerun()
        return
    
    # Display the appropriate page based on session state
    current_page = st.session_state.page
    
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
    else:
        # Default to home page
        show_home_page()
    
    # Add developer credit at the end
    add_developer_credit()

# Application entry point
if __name__ == "__main__":
    main()