import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import newton
import plotly.express as px

class MulticomponentDistillation:
    def __init__(self):
        st.set_page_config(page_title="Multicomponent Distillation", layout="wide")
        st.title("🧪 Multicomponent Distillation Analysis")
        
    def validate_inputs(self, df, light_key, heavy_key):
        """Validation des données d'entrée"""
        errors = []
        
        # Vérification des clés
        if light_key == heavy_key:
            errors.append("Les clés légère et lourde doivent être différentes")
        
        # Vérification somme fractions molaires
        for col in ['xD', 'xW', 'xF']:
            total = df[col].sum()
            if not np.isclose(total, 1.0):
                errors.append(f"La somme des fractions molaires pour {col} doit être 1.0 (actuellement {total})")
        
        # Vérification coefficients Ki
        ki_columns = ['Ki alimentation', 'Ki Distillat', 'Ki résidu']
        for col in ki_columns:
            if (df[col] < 0).any():
                errors.append(f"{col} doit être ≥ 0")
        
        # Vérification des volatilités des clés
        light_key_alpha = df.loc[df['Component'] == light_key, 'Ki alimentation'].values[0]
        heavy_key_alpha = df.loc[df['Component'] == heavy_key, 'Ki alimentation'].values[0]
        
        if np.isclose(light_key_alpha, heavy_key_alpha):
            errors.append("Les volatilités des clés doivent être significativement différentes")
        
        return errors
    
    def fenske_method(self, df, light_key, heavy_key):
        """Calcul du nombre minimum d'étages selon Fenske"""
        try:
            xD_lk = df.loc[df['Component'] == light_key, 'xD'].values[0]
            xD_hk = df.loc[df['Component'] == heavy_key, 'xD'].values[0]
            xW_lk = df.loc[df['Component'] == light_key, 'xW'].values[0]
            xW_hk = df.loc[df['Component'] == heavy_key, 'xW'].values[0]
            
            # Ajouter des vérifications pour éviter les log(0)
            if xD_lk == 0 or xD_hk == 0 or xW_lk == 0 or xW_hk == 0:
                raise ValueError("Les fractions molaires des clés ne peuvent pas être nulles")
            
            alpha_lk = df.loc[df['Component'] == light_key, 'Ki alimentation'].values[0]
            alpha_hk = df.loc[df['Component'] == heavy_key, 'Ki alimentation'].values[0]
            
            alpha_mean = np.sqrt(alpha_lk * alpha_hk)
            
            N_min = np.log(xD_lk/xW_lk * xW_hk/xD_hk) / np.log(alpha_mean)
            return max(N_min, 1)  # Assurer un minimum de 1 étage
        
        except Exception as e:
            st.error(f"Erreur dans la méthode de Fenske : {e}")
            return 1
    
    def underwood_equation(self, theta, df, q, light_key, heavy_key):
        """Équation de Underwood avec contrainte sur θ"""
        try:
            alpha_lk = df.loc[df['Component'] == light_key, 'Ki alimentation'].values[0]
            alpha_hk = df.loc[df['Component'] == heavy_key, 'Ki alimentation'].values[0]
            
            # Vérifier que θ est entre α_lk et α_hk
            if theta < min(alpha_lk, alpha_hk) or theta > max(alpha_lk, alpha_hk):
                return float('inf')  # Valeur invalide
            
            return sum([zi * Ki / (Ki - theta) for zi, Ki in zip(df['xF'], df['Ki alimentation'])]) - (1 - q)
        
        except Exception as e:
            st.error(f"Erreur dans l'équation d'Underwood : {e}")
            return float('inf')
    
    def gilliland_method(self, N_min, theta, df, light_key, heavy_key):
        """Estimation du nombre réel d'étages selon Gilliland"""
        try:
            alpha_lk = df.loc[df['Component'] == light_key, 'Ki alimentation'].values[0]
            alpha_hk = df.loc[df['Component'] == heavy_key, 'Ki alimentation'].values[0]
            
            # Vérifier que N_min est positif
            if N_min <= 0:
                return 1.0
            
            # Vérifier que theta est dans la plage des volatilités
            if theta < min(alpha_lk, alpha_hk) or theta > max(alpha_lk, alpha_hk):
                return 1.0
            
            X = (N_min - 1) / (N_min + 1)
            Y = 1 - np.exp(((1 + 54.4 * X) / (11 + 117.2 * X)) * ((X - 1) / np.sqrt(X)))
            
            return max(Y, 0.1)  # Valeur minimale de 0.1
        
        except Exception as e:
            st.error(f"Erreur dans la méthode de Gilliland : {e}")
            return 1.0
    
    def create_input_dataframe(self, num_components):
        """Création du DataFrame d'entrée"""
        df = pd.DataFrame({
            'Component': [f'Component {i+1}' for i in range(num_components)],
            'xD': [1/num_components] * num_components,
            'xW': [1/num_components] * num_components,
            'xF': [1/num_components] * num_components,
            'Ki alimentation': [1.2 + 0.1*i for i in range(num_components)],  # Volatilités différentes
            'Ki Distillat': [1.2 + 0.1*i for i in range(num_components)],
            'Ki résidu': [1.2 + 0.1*i for i in range(num_components)]
        })
        return df
    
    def run(self):
        # Sélection du nombre de composants
        num_components = st.sidebar.slider("Nombre de composants", 2, 20, 3)
        
        # Création du DataFrame initial
        df = self.create_input_dataframe(num_components)
        
        # Tableau éditable
        edited_df = st.data_editor(df, num_rows="dynamic")
        
        # Sélection des clés
        st.sidebar.header("Sélection des clés")
        light_key = st.sidebar.selectbox("Clé légère", edited_df['Component'])
        heavy_key = st.sidebar.selectbox("Clé lourde", 
                                         edited_df[edited_df['Component'] != light_key]['Component'])
        
        # Qualité thermique de l'alimentation
        q = st.sidebar.slider("Qualité thermique de l'alimentation (q)", 0.0, 1.0, 0.5)
        
        # Bouton de calcul
        if st.button("Calculer"):
            # Validation des données
            errors = self.validate_inputs(edited_df, light_key, heavy_key)
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    # Calcul Fenske
                    N_min = abs(self.fenske_method(edited_df, light_key, heavy_key))
                    
                    # Calcul Underwood (theta)
                    alpha_lk = edited_df.loc[edited_df['Component'] == light_key, 'Ki alimentation'].values[0]
                    alpha_hk = edited_df.loc[edited_df['Component'] == heavy_key, 'Ki alimentation'].values[0]

                    theta = abs(newton(
                        self.underwood_equation, 
                        x0=(alpha_lk + alpha_hk) / 2,  # Initialisation entre les volatilités des clés
                        args=(edited_df, q, light_key, heavy_key)
                    ))
                    
                    # Calcul Gilliland
                    reflux_factor = abs(self.gilliland_method(N_min, theta, edited_df, light_key, heavy_key))
                    
                    # Calcul du reflux minimum (estimation simple)
                    R_min = 1 / (reflux_factor - 1) if reflux_factor > 1 else 1
                    
                    # Estimation du nombre réel d'étages (avec une règle simple)
                    N_reel = max(int(N_min * (1 + reflux_factor)), int(N_min))
                    
                    # Estimation de la position de l'étage d'alimentation
                    N_F = max(int(N_reel / 2), 1)  # Positionnement approximatif au milieu de la colonne
                    
                    # Affichage des résultats
                    st.header("Table récapitulative des résultats")
                    results_df = pd.DataFrame({
                        'Paramètre': [
                            'Nombre minimum d\'étages (Nmin)', 
                            'Reflux minimum (Rmin)', 
                            'Nombre réel d\'étages (N)', 
                            'Position de l\'étage d\'alimentation (NF)',
                            'Volatilité clé légère',
                            'Volatilité clé lourde',
                            'Theta (Underwood)'
                        ],
                        'Valeur': [
                            round(N_min, 2), 
                            round(R_min, 2), 
                            N_reel, 
                            N_F,
                            round(alpha_lk, 2),
                            round(alpha_hk, 2),
                            round(theta, 2)
                        ]
                    })
                    st.dataframe(results_df)
                    
                    # Graphique des résultats
                    fig = px.bar(results_df, x='Paramètre', y='Valeur', 
                                 title='Paramètres de Distillation')
                    st.plotly_chart(fig)
                
                except Exception as e:
                    st.error(f"Erreur de calcul globale : {e}")

# Exécution de l'application
if __name__ == "__main__":
    app = MulticomponentDistillation()
    app.run()