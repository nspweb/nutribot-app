import ast
from dotenv import load_dotenv
from tenacity import retry
load_dotenv()
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
import json
import os
from dataclasses import dataclass
import random
from datetime import datetime
import requests
import time

@dataclass
class UserProfile:
    weight: float
    height: float
    age: int
    gender: str
    bmi: float
    bmr: float
    target_energy_intake: float
    activity_level: str
    special_conditions: List[str]
    dietary_preferences: str

class NutritionDatabase:
    def __init__(self, food_csv_path: str = None, nutrition_excel_path: str = None):
        """
        Initialize database with custom dataset paths
        
        Args:
            food_csv_path: Path to 'merged_food_with_ingredients.csv'
            nutrition_excel_path: Path to 'Recommended Dietary Allowances...' Excel file
        """
        self.food_csv_path = food_csv_path
        self.nutrition_excel_path = nutrition_excel_path
        
        self.food_data = self._load_food_data()
        self.nutrition_requirements = self._load_nutrition_requirements()
    
    def _load_food_data(self) -> pd.DataFrame:
        """Load food data from CSV file or create sample data"""
        if self.food_csv_path and os.path.exists(self.food_csv_path):
            try:
                print(f"Loading food data from: {self.food_csv_path}")
                df = pd.read_csv(self.food_csv_path)
                
                # Validate required columns
                required_columns = ['id', 'cuisine', 'ingredients', 'matched_food', 'calories', 'proteins', 'fat', 'carbohydrate']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"Warning: Missing columns in CSV: {missing_columns}")
                    print(f"Available columns: {list(df.columns)}")
                
                # Add image column if not exists
                if 'image' not in df.columns:
                    df['image'] = 'https://via.placeholder.com/300x200?text=Food+Image'
                
                # Clean data
                df = df.dropna(subset=['matched_food', 'calories'])
                
                # Ensure numeric columns are numeric
                numeric_columns = ['calories', 'proteins', 'fat', 'carbohydrate']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                print(f"Successfully loaded {len(df)} food items from CSV")
                return df
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                print("Using sample data instead...")
                return self._create_sample_food_data()
        else:
            if self.food_csv_path:
                print(f"CSV file not found at: {self.food_csv_path}")
                print("Using sample data instead...")
            return self._create_sample_food_data()
    
    def _create_sample_food_data(self) -> pd.DataFrame:
        """Create comprehensive sample food data"""
        sample_data = [
            {
                'id': 25693,
                'cuisine': 'indonesian',
                'ingredients': "['nasi', 'ayam', 'sayuran', 'minyak goreng', 'bumbu rempah']",
                'matched_food': 'Nasi Ayam Sayur',
                'calories': 450.0,
                'proteins': 25.0,
                'fat': 15.0,
                'carbohydrate': 55.0,
                'image': 'https://via.placeholder.com/300x200?text=Nasi+Ayam'
            },
            {
                'id': 25694, 
                'cuisine': 'indonesian',
                'ingredients': "['ikan', 'nasi', 'sayuran hijau', 'tomat', 'bumbu bakar']",
                'matched_food': 'Ikan Bakar dengan Nasi',
                'calories': 380.0,
                'proteins': 30.0,
                'fat': 12.0,
                'carbohydrate': 45.0,
                'image': 'https://via.placeholder.com/300x200?text=Ikan+Bakar'
            },
            {
                'id': 25695,
                'cuisine': 'indonesian', 
                'ingredients': "['tempe', 'tahu', 'sayuran', 'nasi', 'kecap']",
                'matched_food': 'Tempe Tahu Sayur',
                'calories': 320.0,
                'proteins': 18.0,
                'fat': 10.0,
                'carbohydrate': 42.0,
                'image': 'https://via.placeholder.com/300x200?text=Tempe+Tahu'
            },
            {
                'id': 25696,
                'cuisine': 'indonesian',
                'ingredients': "['telur', 'nasi', 'sayuran', 'minyak', 'bawang']", 
                'matched_food': 'Telur Dadar Sayur',
                'calories': 350.0,
                'proteins': 15.0,
                'fat': 18.0,
                'carbohydrate': 35.0,
                'image': 'https://via.placeholder.com/300x200?text=Telur+Dadar'
            },
            {
                'id': 25697,
                'cuisine': 'international',
                'ingredients': "['buah-buahan', 'yogurt', 'madu', 'granola']",
                'matched_food': 'Fruit Yogurt Bowl',
                'calories': 180.0,
                'proteins': 8.0,
                'fat': 5.0,
                'carbohydrate': 28.0,
                'image': 'https://via.placeholder.com/300x200?text=Fruit+Bowl'
            },
            {
                'id': 25698,
                'cuisine': 'international',
                'ingredients': "['oatmeal', 'banana', 'almond', 'susu', 'madu']",
                'matched_food': 'Oatmeal Banana Almond',
                'calories': 320.0,
                'proteins': 12.0,
                'fat': 8.0,
                'carbohydrate': 54.0,
                'image': 'https://via.placeholder.com/300x200?text=Oatmeal'
            },
            {
                'id': 25699,
                'cuisine': 'international',
                'ingredients': "['telur', 'roti gandum', 'alpukat', 'tomat']",
                'matched_food': 'Scrambled Eggs Avocado Toast',
                'calories': 290.0,
                'proteins': 18.0,
                'fat': 14.0,
                'carbohydrate': 24.0,
                'image': 'https://via.placeholder.com/300x200?text=Eggs+Toast'
            },
            {
                'id': 25700,
                'cuisine': 'indonesian',
                'ingredients': "['kacang tanah', 'garam', 'cabai']",
                'matched_food': 'Kacang Tanah Rebus',
                'calories': 150.0,
                'proteins': 7.0,
                'fat': 12.0,
                'carbohydrate': 6.0,
                'image': 'https://via.placeholder.com/300x200?text=Kacang+Tanah'
            },
            {
                'id': 25701,
                'cuisine': 'indonesian',
                'ingredients': "['nasi', 'rendang', 'sayuran', 'sambal']",
                'matched_food': 'Nasi Rendang',
                'calories': 520.0,
                'proteins': 28.0,
                'fat': 22.0,
                'carbohydrate': 58.0,
                'image': 'https://via.placeholder.com/300x200?text=Nasi+Rendang'
            },
            {
                'id': 25702,
                'cuisine': 'indonesian',
                'ingredients': "['gado-gado', 'sayuran', 'tahu', 'tempe', 'bumbu kacang']",
                'matched_food': 'Gado-Gado Lengkap',
                'calories': 380.0,
                'proteins': 16.0,
                'fat': 18.0,
                'carbohydrate': 42.0,
                'image': 'https://via.placeholder.com/300x200?text=Gado+Gado'
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def _load_nutrition_requirements(self) -> pd.DataFrame:
        """Load nutrition requirements from Excel file or create sample data"""
        if self.nutrition_excel_path and os.path.exists(self.nutrition_excel_path):
            try:
                print(f"Loading nutrition requirements from: {self.nutrition_excel_path}")
                
                xl_file = pd.ExcelFile(self.nutrition_excel_path)
                print(f"Available sheets: {xl_file.sheet_names}")
                
                df = pd.read_excel(self.nutrition_excel_path, sheet_name=0)
                
                print(f"Excel columns: {list(df.columns)}")
                print(f"Successfully loaded {len(df)} nutrition requirement records")
                
                numeric_columns = ['Total Water (L/d)', 'Carbohydrate (g/d)', 'Total Fiber (g/d)', 
                                  'Fat (g/d)', 'Protein (g/d)']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
                
            except Exception as e:
                print(f"Error loading Excel file: {e}")
                print("Using sample nutrition data instead...")
                return self._create_sample_nutrition_data()
        else:
            if self.nutrition_excel_path:
                print(f"Excel file not found at: {self.nutrition_excel_path}")
                print("Using sample nutrition data instead...")
            return self._create_sample_nutrition_data()
    
    def _create_sample_nutrition_data(self) -> pd.DataFrame:
        """Create sample nutrition requirements data"""
        data = [
            {'Life Stage Group': 'Children', 'Age Group': '1–3 y', 'Total Water (L/d)': 1.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 19, 'Fat (g/d)': 30, 'Protein (g/d)': 13},
            {'Life Stage Group': 'Children', 'Age Group': '4–8 y', 'Total Water (L/d)': 1.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 25, 'Fat (g/d)': 35, 'Protein (g/d)': 19},
            {'Life Stage Group': 'Males', 'Age Group': '9–13 y', 'Total Water (L/d)': 2.4, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 31, 'Fat (g/d)': 40, 'Protein (g/d)': 34},
            {'Life Stage Group': 'Males', 'Age Group': '14–18 y', 'Total Water (L/d)': 3.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 38, 'Fat (g/d)': 55, 'Protein (g/d)': 52},
            {'Life Stage Group': 'Males', 'Age Group': '19–30 y', 'Total Water (L/d)': 3.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 38, 'Fat (g/d)': 65, 'Protein (g/d)': 56},
            {'Life Stage Group': 'Males', 'Age Group': '31–50 y', 'Total Water (L/d)': 3.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 38, 'Fat (g/d)': 65, 'Protein (g/d)': 56},
            {'Life Stage Group': 'Females', 'Age Group': '9–13 y', 'Total Water (L/d)': 2.1, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 26, 'Fat (g/d)': 35, 'Protein (g/d)': 34},
            {'Life Stage Group': 'Females', 'Age Group': '14–18 y', 'Total Water (L/d)': 2.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 26, 'Fat (g/d)': 45, 'Protein (g/d)': 46},
            {'Life Stage Group': 'Females', 'Age Group': '19–30 y', 'Total Water (L/d)': 2.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 25, 'Fat (g/d)': 55, 'Protein (g/d)': 46},
            {'Life Stage Group': 'Females', 'Age Group': '31–50 y', 'Total Water (L/d)': 2.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 25, 'Fat (g/d)': 55, 'Protein (g/d)': 46},
        ]
        
        df = pd.DataFrame(data)
        numeric_columns = ['Total Water (L/d)', 'Carbohydrate (g/d)', 'Total Fiber (g/d)', 
                          'Fat (g/d)', 'Protein (g/d)']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def get_nutrition_requirements(self, age: int, gender: str) -> Dict[str, float]:
        """Get nutrition requirements based on age and gender"""
        if age <= 3:
            age_group = '1–3 y'
            life_stage = 'Children'
        elif age <= 8:
            age_group = '4–8 y'
            life_stage = 'Children'
        elif age <= 13:
            age_group = '9–13 y'
            life_stage = gender.title() + 's'
        elif age <= 18:
            age_group = '14–18 y'
            life_stage = gender.title() + 's'
        elif age <= 30:
            age_group = '19–30 y'
            life_stage = gender.title() + 's'
        else:
            age_group = '31–50 y'
            life_stage = gender.title() + 's'
        
        mask = (self.nutrition_requirements['Age Group'] == age_group) & \
               (self.nutrition_requirements['Life Stage Group'] == life_stage)
        
        if not mask.any():
            return {
                'Total Water (L/d)': 2.5,
                'Carbohydrate (g/d)': 130.0,
                'Total Fiber (g/d)': 25.0,
                'Fat (g/d)': 65.0,
                'Protein (g/d)': 50.0
            }
        
        req = self.nutrition_requirements[mask].iloc[0]
        return {
            'Total Water (L/d)': float(req['Total Water (L/d)']),
            'Carbohydrate (g/d)': float(req['Carbohydrate (g/d)']),
            'Total Fiber (g/d)': float(req['Total Fiber (g/d)']),
            'Fat (g/d)': max(float(req['Fat (g/d)']), 20.0),
            'Protein (g/d)': float(req['Protein (g/d)'])
        }
    
    def search_foods(self, query: str, limit: int = 5) -> List[Dict]:
        """Search foods based on query"""
        query_lower = query.lower()
        
        filtered_foods = []
        for _, food in self.food_data.iterrows():
            try:
                ingredients = ast.literal_eval(food['ingredients'])
                if (query_lower in food['matched_food'].lower() or 
                    query_lower in food['ingredients'].lower() or
                    any(query_lower in ingredient.lower() for ingredient in ingredients)):
                    filtered_foods.append(food.to_dict())
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse ingredients for {food['matched_food']}")
                continue
        
        return filtered_foods[:limit]

class DetailedNutritionKnowledge:
    """Comprehensive nutrition knowledge base for detailed responses"""
    
    @staticmethod
    def get_bmi_interpretation(bmi: float) -> Dict[str, str]:
        """Get detailed BMI interpretation"""
        if bmi < 18.5:
            category = "Underweight"
            description = "BMI di bawah normal. Disarankan untuk meningkatkan asupan kalori dengan makanan bergizi."
            recommendations = [
                "Tingkatkan porsi makan secara bertahap",
                "Konsumsi makanan padat nutrisi seperti kacang-kacangan, alpukat",
                "Makan lebih sering dengan porsi kecil (5-6 kali sehari)",
                "Konsultasi dengan ahli gizi untuk program penambahan berat badan"
            ]
        elif 18.5 <= bmi < 25:
            category = "Normal"
            description = "BMI dalam rentang normal. Pertahankan pola makan sehat dan aktivitas fisik."
            recommendations = [
                "Pertahankan pola makan seimbang",
                "Lakukan olahraga teratur 150 menit/minggu",
                "Konsumsi buah dan sayur minimal 5 porsi/hari",
                "Minum air putih minimal 8 gelas/hari"
            ]
        elif 25 <= bmi < 30:
            category = "Overweight"
            description = "BMI di atas normal. Disarankan menurunkan berat badan untuk kesehatan optimal."
            recommendations = [
                "Kurangi asupan kalori 300-500 kcal/hari",
                "Tingkatkan aktivitas fisik menjadi 300 menit/minggu",
                "Batasi makanan olahan dan tinggi gula",
                "Fokus pada makanan tinggi serat dan protein"
            ]
        else:
            category = "Obese"
            description = "BMI menunjukkan obesitas. Perlu program penurunan berat badan yang terstruktur."
            recommendations = [
                "Konsultasi dengan dokter atau ahli gizi",
                "Buat rencana penurunan berat badan bertahap (0.5-1 kg/minggu)",
                "Gabungkan diet rendah kalori dengan olahraga",
                "Pertimbangkan dukungan kelompok atau program khusus"
            ]
        
        return {
            "category": category,
            "description": description,
            "recommendations": recommendations
        }
    
    @staticmethod
    def get_nutrient_functions() -> Dict[str, Dict[str, str]]:
        """Get detailed information about nutrient functions"""
        return {
            "protein": {
                "function": "Membangun dan memperbaiki jaringan tubuh, produksi enzim dan hormon",
                "sources": "Daging, ikan, telur, kacang-kacangan, tahu, tempe",
                "deficiency": "Kehilangan massa otot, penyembuhan lambat, sistem imun lemah",
                "excess": "Beban berlebih pada ginjal, dehidrasi"
            },
            "carbohydrate": {
                "function": "Sumber energi utama untuk otak dan otot",
                "sources": "Nasi, roti, pasta, buah-buahan, sayuran bertepung",
                "deficiency": "Kelelahan, kesulitan konsentrasi, hipoglikemia",
                "excess": "Penambahan berat badan, diabetes tipe 2"
            },
            "fat": {
                "function": "Penyimpanan energi, penyerapan vitamin larut lemak, produksi hormon",
                "sources": "Minyak, kacang-kacangan, alpukat, ikan berlemak",
                "deficiency": "Kulit kering, gangguan hormon, defisiensi vitamin A,D,E,K",
                "excess": "Obesitas, penyakit jantung, kolesterol tinggi"
            },
            "fiber": {
                "function": "Melancarkan pencernaan, mengontrol gula darah, menurunkan kolesterol",
                "sources": "Sayuran, buah-buahan, kacang-kacangan, biji-bijian utuh",
                "deficiency": "Sembelit, kolesterol tinggi, gula darah tidak stabil",
                "excess": "Kembung, gangguan penyerapan mineral"
            },
            "water": {
                "function": "Mengatur suhu tubuh, transportasi nutrisi, detoksifikasi",
                "sources": "Air putih, buah-buahan, sayuran, sup",
                "deficiency": "Dehidrasi, kelelahan, batu ginjal, sembelit",
                "excess": "Hiponatremia (jarang terjadi)"
            }
        }

class FastLLMProcessor:
    def __init__(self, llm_type: str = "groq"):
        """
        Initialize with LLM options: 'groq' or 'rule_based'
        
        Args:
            llm_type: 'groq' for Groq API, 'rule_based' for rule-based responses
        """
        self.llm_type = llm_type
        self.api_key = None
        
        if llm_type == "groq":
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model_name = "llama3-8b-8192"
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                print("Warning: GROQ_API_KEY not found. Falling back to rule-based.")
                self.llm_type = "rule_based"
        
        print(f"Initialized {self.llm_type} LLM processor")
    
    def _make_api_request(self, data: Dict, headers: Dict = None, timeout: int = 15) -> Dict:
        """Generic API request handler with retry logic"""
        @retry(tries=3, delay=1, backoff=2)
        def request_with_retry():
            response = requests.post(self.base_url, headers=headers, json=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        
        try:
            return request_with_retry()
        except Exception as e:
            print(f"API request failed: {str(e)}")
            return None

    def generate_response_groq(self, prompt: str, context: str = "") -> str:
        """Generate response using Groq API"""
        if not self.api_key:
            return self.generate_response_rule_based(prompt, context)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """You are NuZiBot, an expert nutrition consultant for children and adolescents, supporting SDG 2 (Zero Hunger) and SDG 3 (Good Health and Well-Being). Your purpose is to provide educational and practical nutritional advice tailored to the needs of young users (ages 1-18) in Indonesia and globally. ONLY respond to questions related to nutrition, diet, healthy eating, or health conditions (e.g., allergies, diabetes, hypertension, veganism). For any unrelated questions (e.g., geography, mechanics, or other non-nutrition topics), politely redirect the user to ask about nutrition or diet, and provide a relevant suggestion based on their profile (age, gender, dietary needs).

        Your responses must include:
        - Specific nutritional values (calories, protein, fat, carbs)
        - Practical meal suggestions with exact portions, suitable for children/adolescents
        - Health benefits and risks, explained in simple terms
        - Alternative options for dietary needs (e.g., vegan, allergies, lactose intolerance)
        - Cultural context for Indonesian foods (e.g., nasi uduk, gado-gado)
        - Step-by-step guidance or tips for healthy eating
        Use clear sections, emojis (e.g., 🥗, 🍽️), and actionable advice written in a friendly, engaging tone suitable for young users. Always consider the user's profile (age, gender, activity level, special conditions, dietary preferences) from the provided context."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nUser question: {prompt}\n\nProvide a detailed, comprehensive response related to nutrition or redirect to a nutrition topic if the question is unrelated."}
        ]
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 800,
            "stream": False
        }
        
        response = self._make_api_request(data, headers)
        if response and "choices" in response:
            response_text = response["choices"][0]["message"]["content"].strip()
            relevant_keywords = ['nutrition', 'diet', 'healthy', 'food', 'menu', 'calorie', 'protein', 'gizi', 'makan', 'sehat']
            if any(keyword in response_text.lower() for keyword in relevant_keywords):
                return response_text
        return self.generate_response_rule_based(prompt, context)

    def generate_response_rule_based(self, prompt: str, context: str = "") -> str:
        """Generate comprehensive rule-based responses with detailed nutrition information"""
        prompt_lower = prompt.lower()
        
        age_match = re.search(r'Age: (\d+)', context)
        gender_match = re.search(r'Gender: (\w+)', context)
        calorie_match = re.search(r'Caloric Needs: (\d+)', context)
        bmi_match = re.search(r'BMI: ([\d.]+)', context)
        activity_match = re.search(r'Activity Level: ([^-]+)', context)
        weight_match = re.search(r'Weight: ([\d.]+)', context)
        conditions_match = re.search(r'Special Conditions: ([^\n]+)', context)
        preferences_match = re.search(r'Dietary Preferences: ([^\n]+)', context)
        
        age = int(age_match.group(1)) if age_match else 15
        gender = gender_match.group(1).lower() if gender_match else "female"
        calories = int(calorie_match.group(1)) if calorie_match else 2000
        bmi = float(bmi_match.group(1)) if bmi_match else 20.0
        activity_level = activity_match.group(1).strip() if activity_match else "Moderately Active"
        weight = float(weight_match.group(1)) if weight_match else 55.0
        special_conditions = conditions_match.group(1).split(', ') if conditions_match and conditions_match.group(1) != 'None' else []
        dietary_preferences = preferences_match.group(1) if preferences_match and preferences_match.group(1) != 'None' else ""
        
        breakfast_options = [
            {
                "name": "Nasi Uduk dengan Telur dan Tempe",
                "calories": 420,
                "protein": 18,
                "fat": 12,
                "carbs": 58,
                "description": "Sarapan tradisional Indonesia yang kaya akan protein dan karbohidrat kompleks.",
                "portion": "1 porsi nasi uduk (150g), 1 telur rebus (50g), 50g tempe goreng",
                "cultural_context": "Nasi uduk adalah makanan pokok sarapan di Jakarta.",
                "restrictions": ["contains_egg", "contains_soy"]
            },
            {
                "name": "Oatmeal dengan Pisang",
                "calories": 280,
                "protein": 8,
                "fat": 4,
                "carbs": 54,
                "description": "Sarapan sehat dengan serat tinggi dari oatmeal dan pisang.",
                "portion": "50g oatmeal, 1 pisang (120g)",
                "cultural_context": "Oatmeal populer untuk sarapan cepat dan sehat.",
                "restrictions": []
            },
            {
                "name": "Gado-Gado Vegan",
                "calories": 380,
                "protein": 16,
                "fat": 18,
                "carbs": 42,
                "description": "Salad tradisional Indonesia dengan saus kacang vegan.",
                "portion": "200g sayuran, 50g tahu, 50g tempe, 30g saus kacang vegan",
                "cultural_context": "Gado-gado adalah hidangan vegetarian populer di Indonesia.",
                "restrictions": ["contains_soy", "contains_peanuts"]
            },
            {
                "name": "Bubur Sumsum",
                "calories": 250,
                "protein": 5,
                "fat": 8,
                "carbs": 40,
                "description": "Bubur tradisional Indonesia dari tepung beras dengan gula merah.",
                "portion": "200g bubur, 30g gula merah",
                "cultural_context": "Bubur sumsum sering disajikan sebagai sarapan ringan di Indonesia.",
                "restrictions": []
            }
        ]
        
        relevant_keywords = [
            'breakfast', 'menu', 'meal', 'food', 'recommend', 'nutrition', 'diet', 'healthy',
            'protein', 'calorie', 'energy', 'weight', 'gain', 'lose', 'allergy', 'allergic',
            'nuts', 'milk', 'lactose', 'diabetes', 'hypertension', 'blood pressure', 'vegan',
            'vegetarian', 'gizi', 'makan', 'sehat', 'karbohidrat', 'lemak', 'serat', 'vitamin'
        ]
        
        is_relevant = any(word in prompt_lower for word in relevant_keywords)
        
        if not is_relevant:
            suggestion = random.choice([
                "Apa menu sarapan sehat untuk saya?",
                "Berapa kebutuhan protein harian saya?",
                "Makanan apa yang cocok untuk anak usia saya?",
                "Bagaimana cara makan sehat dengan alergi kacang?"
            ])
            return f"🤗 **Maaf, pertanyaan itu di luar topik gizi!** Saya adalah NuZiBot, dibuat untuk membantu anak dan remaja belajar tentang makan sehat.\n\n" \
                   f"Berdasarkan profil Anda (Umur: {age}, Jenis Kelamin: {gender}), saya bisa membantu dengan rekomendasi menu atau tips gizi. " \
                   f"Contohnya, coba tanyakan: **'{suggestion}'** atau ceritakan kebutuhan diet Anda!"

        if any(word in prompt_lower for word in ['breakfast', 'menu', 'meal', 'food', 'recommend']):
            valid_options = breakfast_options
            if "Vegan" in special_conditions or "vegan" in dietary_preferences.lower():
                valid_options = [opt for opt in breakfast_options if "contains_egg" not in opt["restrictions"] and "contains_dairy" not in opt["restrictions"]]
                if not valid_options:
                    valid_options = [breakfast_options[1]]
                option = random.choice(valid_options)
                return f"🌱 **Rekomendasi Sarapan Vegan: {option['name']}**\n\n" \
                       f"📊 **Nilai Gizi (per porsi):**\n" \
                       f"• Kalori: {option['calories']} kcal\n" \
                       f"• Protein: {option['protein']}g\n" \
                       f"• Lemak: {option['fat']}g\n" \
                       f"• Karbohidrat: {option['carbs']}g\n\n" \
                       f"📝 **Deskripsi:** {option['description']}\n" \
                       f"🥄 **Porsi:** {option['portion']}\n" \
                       f"🌍 **Konteks Budaya:** {option['cultural_context']}\n\n" \
                       f"💡 **Manfaat Kesehatan:** Bebas produk hewani, cocok untuk diet vegan.\n" \
                       f"⚠️ **Catatan:** Pastikan saus tidak mengandung produk hewani."
            elif "Vegetarian" in special_conditions or "vegetarian" in dietary_preferences.lower():
                valid_options = [opt for opt in breakfast_options if "contains_meat" not in opt["restrictions"]]
                if not valid_options:
                    valid_options = [breakfast_options[1]]
                option = random.choice(valid_options)
                return f"🥗 **Rekomendasi Sarapan Vegetarian: {option['name']}**\n\n" \
                       f"📊 **Nilai Gizi (per porsi):**\n" \
                       f"• Kalori: {option['calories']} kcal\n" \
                       f"• Protein: {option['protein']}g\n" \
                       f"• Lemak: {option['fat']}g\n" \
                       f"• Karbohidrat: {option['carbs']}g\n\n" \
                       f"📝 **Deskripsi:** {option['description']}\n" \
                       f"🥄 **Porsi:** {option['portion']}\n" \
                       f"🌍 **Konteks Budaya:** {option['cultural_context']}\n\n" \
                       f"💡 **Manfaat Kesehatan:** Bebas daging, mendukung diet vegetarian.\n" \
                       f"⚠️ **Catatan:** Pastikan saus tidak mengandung ikan atau udang."
            elif "Nut Allergy" in special_conditions or "nut" in dietary_preferences.lower():
                valid_options = [opt for opt in breakfast_options if "contains_peanuts" not in opt["restrictions"] and "contains_nuts" not in opt["restrictions"]]
                if not valid_options:
                    valid_options = [breakfast_options[1]]
                option = random.choice(valid_options)
                return f"🚫 **Rekomendasi Sarapan Bebas Kacang: {option['name']}**\n\n" \
                       f"📊 **Nilai Gizi (per porsi):**\n" \
                       f"• Kalori: {option['calories']} kcal\n" \
                       f"• Protein: {option['protein']}g\n" \
                       f"• Lemak: {option['fat']}g\n" \
                       f"• Karbohidrat: {option['carbs']}g\n\n" \
                       f"📝 **Deskripsi:** {option['description']}\n" \
                       f"🥄 **Porsi:** {option['portion']}\n" \
                       f"🌍 **Konteks Budaya:** {option['cultural_context']}\n\n" \
                       f"💡 **Manfaat Kesehatan:** Aman untuk alergi kacang, memberikan energi tahan lama.\n" \
                       f"⚠️ **Catatan:** Periksa label untuk memastikan tidak ada kontaminasi kacang."
            elif "Milk Allergy" in special_conditions or "Lactose Intolerant" in special_conditions or "milk" in dietary_preferences.lower() or "lactose" in dietary_preferences.lower():
                valid_options = [opt for opt in breakfast_options if "contains_dairy" not in opt["restrictions"]]
                if not valid_options:
                    valid_options = [breakfast_options[1]]
                option = random.choice(valid_options)
                return f"🥛 **Rekomendasi Sarapan Bebas Susu: {option['name']}**\n\n" \
                       f"📊 **Nilai Gizi (per porsi):**\n" \
                       f"• Kalori: {option['calories']} kcal\n" \
                       f"• Protein: {option['protein']}g\n" \
                       f"• Lemak: {option['fat']}g\n" \
                       f"• Karbohidrat: {option['carbs']}g\n\n" \
                       f"📝 **Deskripsi:** {option['description']}\n" \
                       f"🥄 **Porsi:** {option['portion']}\n" \
                       f"🌍 **Konteks Budaya:** {option['cultural_context']}\n\n" \
                       f"💡 **Manfaat Kesehatan:** Aman untuk alergi susu atau intoleransi laktosa.\n" \
                       f"⚠️ **Catatan:** Gunakan pengganti susu seperti santan atau susu nabati."
            elif "Diabetes" in special_conditions or "diabetes" in dietary_preferences.lower():
                valid_options = [opt for opt in breakfast_options if opt["carbs"] <= 45]
                if not valid_options:
                    valid_options = [breakfast_options[2]]
                option = random.choice(valid_options)
                return f"🩺 **Rekomendasi Sarapan untuk Diabetes: {option['name']}**\n\n" \
                       f"📊 **Nilai Gizi (per porsi):**\n" \
                       f"• Kalori: {option['calories']} kcal\n" \
                       f"• Protein: {option['protein']}g\n" \
                       f"• Lemak: {option['fat']}g\n" \
                       f"• Karbohidrat: {option['carbs']}g\n\n" \
                       f"📝 **Deskripsi:** {option['description']}\n" \
                       f"🥄 **Porsi:** {option['portion']}\n" \
                       f"🌍 **Konteks Budaya:** {option['cultural_context']}\n\n" \
                       f"💡 **Manfaat Kesehatan:** Rendah karbohidrat sederhana, membantu menjaga kadar gula darah.\n" \
                       f"⚠️ **Catatan:** Konsultasikan dengan dokter untuk porsi dan indeks glikemik."
            elif "Hypertension" in special_conditions or "hypertension" in dietary_preferences.lower():
                valid_options = [opt for opt in breakfast_options if opt["fat"] <= 15]
                if not valid_options:
                    valid_options = [breakfast_options[1]]
                option = random.choice(valid_options)
                return f"🩺 **Rekomendasi Sarapan untuk Hipertensi: {option['name']}**\n\n" \
                       f"📊 **Nilai Gizi (per porsi):**\n" \
                       f"• Kalori: {option['calories']} kcal\n" \
                       f"• Protein: {option['protein']}g\n" \
                       f"• Lemak: {option['fat']}g\n" \
                       f"• Karbohidrat: {option['carbs']}g\n\n" \
                       f"📝 **Deskripsi:** {option['description']}\n" \
                       f"🥄 **Porsi:** {option['portion']}\n" \
                       f"🌍 **Konteks Budaya:** {option['cultural_context']}\n\n" \
                       f"💡 **Manfaat Kesehatan:** Rendah lemak jenuh, mendukung tekanan darah stabil.\n" \
                       f"⚠️ **Catatan:** Hindari tambahan garam dan konsultasikan dengan dokter."
            else:
                option = random.choice(breakfast_options)
                return f"🍽️ **Rekomendasi Sarapan: {option['name']}**\n\n" \
                       f"📊 **Nilai Gizi (per porsi):**\n" \
                       f"• Kalori: {option['calories']} kcal\n" \
                       f"• Protein: {option['protein']}g\n" \
                       f"• Lemak: {option['fat']}g\n" \
                       f"• Karbohidrat: {option['carbs']}g\n\n" \
                       f"📝 **Deskripsi:** {option['description']}\n" \
                       f"🥄 **Porsi:** {option['portion']}\n" \
                       f"🌍 **Konteks Budaya:** {option['cultural_context']}\n\n" \
                       f"💡 **Manfaat Kesehatan:** Makanan ini memberikan energi tahan lama.\n" \
                       f"⚠️ **Catatan:** Sesuaikan porsi dengan kebutuhan kalori ({calories} kcal)."
        
        elif any(word in prompt_lower for word in ['protein', 'requirement']):
            protein_req = 46 if gender.lower() == 'female' else 56
            if age < 18:
                protein_req = max(34, protein_req * 0.8)
            if "Vegan" in special_conditions or "vegan" in dietary_preferences.lower():
                protein_sources = "tofu, tempe, kacang-kacangan, biji-bijian, seitan"
            elif "Vegetarian" in special_conditions or "vegetarian" in dietary_preferences.lower():
                protein_sources = "tofu, tempe, kacang-kacangan, telur, susu, keju"
            else:
                protein_sources = "daging ayam, ikan, telur, tahu, tempe, kacang-kacangan"
            return f"💪 **Kebutuhan Protein Harian:**\n\n" \
                   f"Berdasarkan profil Anda (Umur: {age}, Jenis Kelamin: {gender}), kebutuhan protein harian Anda sekitar **{protein_req}g per hari**.\n" \
                   f"📝 **Manfaat:** Protein membantu mempertahankan massa otot, mendukung metabolisme, dan mempercepat pemulihan.\n" \
                   f"🍗 **Sumber Protein:** {protein_sources}.\n" \
                   f"💡 **Tips:** Konsumsi 20-30g protein per makan untuk penyerapan optimal."

        elif any(word in prompt_lower for word in ['calorie', 'energy']):
            return f"🔥 **Kebutuhan Kalori Harian:**\n\n" \
                   f"Berdasarkan profil Anda (Umur: {age}, Jenis Kelamin: {gender}, Tingkat Aktivitas: {activity_level}), " \
                   f"kebutuhan kalori harian Anda sekitar **{calories} kcal/hari**.\n" \
                   f"📝 **Penjelasan:** Ini mencakup tingkat metabolisme basal ditambah kalori untuk aktivitas.\n" \
                   f"💡 **Tips:** Bagi kalori ke dalam 3-5 makan sehari untuk menjaga energi stabil."

        elif any(word in prompt_lower for word in ['weight', 'gain', 'lose']):
            if 'gain' in prompt_lower:
                food_recommendations = "nasi uduk, alpukat, kacang-kacangan, smoothies dengan susu" if not ("Vegan" in special_conditions or "vegan" in dietary_preferences.lower()) else "nasi uduk vegan, alpukat, kacang-kacangan, smoothies dengan susu nabati"
                return f"🏋️ **Tips Menambah Berat Badan Secara Sehat:**\n\n" \
                       f"• **Tujuan:** Surplus kalori 300-500 kcal di atas kebutuhan Anda ({calories} kcal/hari).\n" \
                       f"• **Rekomendasi Makanan:** {food_recommendations}.\n" \
                       f"• **Panduan:**\n" \
                       f"  1. Makan 5-6 kali sehari dengan porsi kecil.\n" \
                       f"  2. Tambahkan lemak sehat seperti minyak zaitun atau santan.\n" \
                       f"  3. Lakukan latihan kekuatan 3-4 kali/minggu.\n" \
                       f"⚠️ **Catatan:** Konsultasikan dengan ahli gizi untuk rencana personal."
            else:
                return f"⚖️ **Tips Mengelola Berat Badan:**\n\n" \
                       f"• **Tujuan:** Pertahankan diet seimbang sekitar {calories} kcal/hari.\n" \
                       f"• **Rekomendasi Makanan:** Sayuran, protein tanpa lemak (ayam, ikan, tahu, tempe), karbohidrat kompleks (beras merah).\n" \
                       f"• **Panduan:**\n" \
                       f"  1. Kontrol porsi dengan piring kecil.\n" \
                       f"  2. Hindari makanan olahan dan minuman manis.\n" \
                       f"  3. Olahraga teratur 150 menit/minggu.\n" \
                       f"💡 **Tips:** Catat asupan makanan untuk memantau kalori."

        elif any(word in prompt_lower for word in ['allergy', 'allergic', 'nuts']):
            return f"🚫 **Manajemen Alergi Makanan (Kacang):**\n\n" \
                   f"• **Alternatif Aman:** Biji-bijian (biji bunga matahari, biji labu), alpukat, telur, ikan, kacang polong.\n" \
                   f"• **Panduan:**\n" \
                   f"  1. Selalu baca label bahan pada kemasan.\n" \
                   f"  2. Hindari makanan olahan yang mungkin mengandung kacang.\n" \
                   f"  3. Konsultasikan dengan dokter untuk tes alergi.\n" \
                   f"💡 **Tips:** Gado-gado tanpa kacang tanah atau smoothies buah adalah pilihan aman."

        elif any(word in prompt_lower for word in ['milk', 'lactose']):
            return f"🥛 **Manajemen Alergi Susu atau Intoleransi Laktosa:**\n\n" \
                   f"• **Alternatif Aman:** Susu nabati (susu kedelai, almond, oat), tahu, tempe, sayuran hijau.\n" \
                   f"• **Panduan:**\n" \
                   f"  1. Periksa label untuk produk susu tersembunyi.\n" \
                   f"  2. Gunakan santan untuk masakan tradisional Indonesia.\n" \
                   f"  3. Konsultasikan dengan dokter untuk suplemen kalsium jika diperlukan.\n" \
                   f"💡 **Tips:** Coba oatmeal dengan susu nabati atau bubur sumsum tanpa susu."

        elif any(word in prompt_lower for word in ['diabetes']):
            return f"🩺 **Manajemen Diet untuk Diabetes:**\n\n" \
                   f"• **Tujuan:** Jaga kadar gula darah dengan makanan rendah indeks glikemik.\n" \
                   f"• **Rekomendasi Makanan:** Sayuran hijau, beras merah, tahu, tempe, ikan tanpa lemak.\n" \
                   f"• **Panduan:**\n" \
                   f"  1. Batasi karbohidrat sederhana (gula, nasi putih).\n" \
                   f"  2. Konsumsi serat tinggi seperti sayuran dan kacang-kacangan.\n" \
                   f"  3. Makan dalam porsi kecil 5-6 kali sehari.\n" \
                   f"⚠️ **Catatan:** Konsultasikan dengan dokter atau ahli gizi untuk rencana diet."

        elif any(word in prompt_lower for word in ['hypertension', 'blood pressure']):
            return f"🩺 **Manajemen Diet untuk Hipertensi:**\n\n" \
                   f"• **Tujuan:** Kurangi asupan garam dan lemak jenuh untuk menjaga tekanan darah.\n" \
                   f"• **Rekomendasi Makanan:** Sayuran hijau, buah-buahan, beras merah, ikan tanpa lemak.\n" \
                   f"• **Panduan:**\n" \
                   f"  1. Hindari makanan olahan tinggi garam (misalnya, makanan kaleng).\n" \
                   f"  2. Gunakan bumbu alami seperti jahe atau kunyit.\n" \
                   f"  3. Konsumsi kalium dari pisang atau bayam.\n" \
                   f"⚠️ **Catatan:** Konsultasikan dengan dokter untuk pemantauan tekanan darah."

        else:
            return f"🤗 **Saran Gizi Umum:**\n\n" \
                   f"Berdasarkan profil Anda (Umur: {age}, Jenis Kelamin: {gender}), pertahankan diet seimbang dengan asupan {calories} kcal/hari.\n" \
                   f"• **Rekomendasi:** Konsumsi 5 porsi sayur dan buah, protein tanpa lemak, dan karbohidrat kompleks.\n" \
                   f"• **Panduan:**\n" \
                   f"  1. Makan 3-5 kali sehari.\n" \
                   f"  2. Minum air putih minimal 8 gelas/hari.\n" \
                   f"  3. Batasi gula dan lemak jenuh.\n" \
                   f"💬 **Ingin rekomendasi spesifik?** Tanyakan tentang menu, protein, atau kalori!"

    def generate_response(self, prompt: str, context: str = "") -> str:
        """Main method to generate response based on selected LLM type"""
        start_time = time.time()
        
        try:
            if self.llm_type == "groq":
                response = self.generate_response_groq(prompt, context)
            else:
                response = self.generate_response_rule_based(prompt, context)
            
            elapsed_time = time.time() - start_time
            print(f"Response generated in {elapsed_time:.2f} seconds using {self.llm_type}")
            return response
            
        except Exception as e:
            print(f"Error generating response with {self.llm_type}: {str(e)}")
            return self.generate_response_rule_based(prompt, context)

class NutritionChatbot:
    def __init__(self, food_csv_path: str = None, nutrition_excel_path: str = None, llm_type: str = "groq"):
        """
        Initialize chatbot with specified LLM type and data paths.
        
        Args:
            food_csv_path: Path to food CSV file
            nutrition_excel_path: Path to nutrition Excel file
            llm_type: 'groq' or 'rule_based'
        """
        self.database = NutritionDatabase(food_csv_path, nutrition_excel_path)
        self.llm_processor = FastLLMProcessor(llm_type=llm_type)
        self.user_profile: Optional[UserProfile] = None
        self.conversation_history: List[tuple] = []
        print(f"Nutrition Chatbot initialized with {llm_type} LLM")

    def update_user_profile(self, profile_data: Dict[str, Any]):
        """Update user profile with validation"""
        required_fields = ['weight', 'height', 'age', 'gender', 'bmi', 'bmr', 
                          'target_energy_intake', 'activity_level', 'special_conditions', 'dietary_preferences']
        
        for field in required_fields:
            if field not in profile_data:
                raise ValueError(f"Missing required field: {field}")
        
        if profile_data['weight'] <= 0:
            raise ValueError("Weight must be positive")
        if profile_data['height'] <= 0:
            raise ValueError("Height must be positive")
        if profile_data['age'] < 0:
            raise ValueError("Age cannot be negative")
        if profile_data['gender'].lower() not in ['male', 'female']:
            raise ValueError("Gender must be 'male' or 'female'")
        if profile_data['bmi'] <= 0:
            raise ValueError("BMI must be positive")
        if profile_data['bmr'] <= 0:
            raise ValueError("BMR must be positive")
        if profile_data['target_energy_intake'] <= 0:
            raise ValueError("Target energy intake must be positive")
        
        self.user_profile = UserProfile(
            weight=profile_data['weight'],
            height=profile_data['height'],
            age=profile_data['age'],
            gender=profile_data['gender'],
            bmi=profile_data['bmi'],
            bmr=profile_data['bmr'],
            target_energy_intake=profile_data['target_energy_intake'],
            activity_level=profile_data['activity_level'],
            special_conditions=profile_data['special_conditions'],
            dietary_preferences=profile_data['dietary_preferences']
        )
    
    def get_nutrition_recommendations(self, profile: Dict[str, Any]) -> Dict[str, float]:
        return self.database.get_nutrition_requirements(profile['age'], profile['gender'])
    
    def _create_context(self) -> str:
        if not self.user_profile:
            return "Context: User has not completed their profile."
        
        context = f"""User Profile Context:
        - Age: {self.user_profile.age} years
        - Gender: {self.user_profile.gender}
        - Weight: {self.user_profile.weight} kg
        - Height: {self.user_profile.height} cm
        - BMI: {self.user_profile.bmi:.1f}
        - Activity Level: {self.user_profile.activity_level}
        - Caloric Needs: {self.user_profile.target_energy_intake:.0f} kcal/day
        - Special Conditions: {', '.join(self.user_profile.special_conditions) if self.user_profile.special_conditions else 'None'}
        - Dietary Preferences: {self.user_profile.dietary_preferences if self.user_profile.dietary_preferences else 'None'}"""
        
        return context
    
    def get_response(self, user_input: str) -> str:
        """Get chatbot response to user input"""
        self.conversation_history.append(("user", user_input))
        context = self._create_context()
        
        response = self.llm_processor.generate_response(user_input, context)
        
        self.conversation_history.append(("bot", response))
        
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response

if __name__ == "__main__":
    print("=== Fast Nutrition Chatbot Demo ===")
    print("\nAvailable LLM options:")
    print("1. 'groq' - Fast API (requires GROQ_API_KEY)")
    print("2. 'rule_based' - Fastest, no API needed")
    
    chatbot = NutritionChatbot(
        food_csv_path="merged_food_with_ingredients.csv",
        nutrition_excel_path="Recommended Dietary Allowances and Adequate Intakes Total Water and Macronutrients.xlsx",
        llm_type="groq"
    )
    
    sample_profile = {
        'weight': 55.0,
        'height': 165.0,
        'age': 15,
        'gender': 'Female',
        'bmi': 20.2,
        'bmr': 1350,
        'target_energy_intake': 2000,
        'activity_level': 'Moderately Active',
        'special_conditions': [],
        'dietary_preferences': ''
    }
    
    chatbot.update_user_profile(sample_profile)
    
    test_queries = [
        "What is a suitable breakfast menu for me?",
        "What is my daily protein requirement?", 
        "I am allergic to nuts, what foods are safe?",
        "Tips to gain weight healthily?",
        "How many calories should I eat per day?",
        "How to fix a bicycle?"  # Test off-topic query
    ]
    
    print("\n=== Testing Responses ===")
    for query in test_queries:
        print(f"\n👤 User: {query}")
        start_time = time.time()
        response = chatbot.get_response(query)
        elapsed_time = time.time() - start_time
        print(f"🤖 NuZiBot ({elapsed_time:.2f}s): {response}")
        print("-" * 70)
    
    print(f"\n📊 Performance Summary:")
    print(f"Using {chatbot.llm_processor.llm_type} LLM")
    print(f"\n🚀 To use Groq, get a free key from https://console.groq.com/")