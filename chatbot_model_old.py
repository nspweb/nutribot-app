from dotenv import load_dotenv
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
            food_csv_path: Path to your 'merged_food_with_ingredients.csv' file
            nutrition_excel_path: Path to your 'Recommended Dietary Allowances...' Excel file
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
        """Create sample food data if CSV is not available"""
        sample_data = [
            {
                'id': 25693,
                'cuisine': 'indonesian',
                'ingredients': "['nasi', 'ayam', 'sayuran', 'minyak goreng']",
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
                'ingredients': "['ikan', 'nasi', 'sayuran hijau', 'tomat']",
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
                'ingredients': "['tempe', 'tahu', 'sayuran', 'nasi']",
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
                'ingredients': "['telur', 'nasi', 'sayuran', 'minyak']", 
                'matched_food': 'Telur Dadar Sayur',
                'calories': 350.0,
                'proteins': 15.0,
                'fat': 18.0,
                'carbohydrate': 35.0,
                'image': 'https://via.placeholder.com/300x200?text=Telur+Dadar'
            },
            {
                'id': 25697,
                'cuisine': 'indonesian',
                'ingredients': "['buah-buahan', 'yogurt', 'madu']",
                'matched_food': 'Fruit Yogurt Bowl',
                'calories': 180.0,
                'proteins': 8.0,
                'fat': 5.0,
                'carbohydrate': 28.0,
                'image': 'https://via.placeholder.com/300x200?text=Fruit+Bowl'
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def _load_nutrition_requirements(self) -> pd.DataFrame:
        """Load nutrition requirements from Excel file or create sample data"""
        if self.nutrition_excel_path and os.path.exists(self.nutrition_excel_path):
            try:
                print(f"Loading nutrition requirements from: {self.nutrition_excel_path}")
                
                # Try to read Excel file (could be multiple sheets)
                xl_file = pd.ExcelFile(self.nutrition_excel_path)
                print(f"Available sheets: {xl_file.sheet_names}")
                
                # Read the first sheet (or specify sheet name if known)
                df = pd.read_excel(self.nutrition_excel_path, sheet_name=0)
                
                print(f"Excel columns: {list(df.columns)}")
                print(f"Successfully loaded {len(df)} nutrition requirement records")
                
                # Ensure numeric columns are converted to float
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
            {'Life Stage Group': 'Children', 'Age Group': '1–3 y', 'Total Water (L/d)': 1.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 19, 'Fat (g/d)': 0, 'Protein (g/d)': 13},
            {'Life Stage Group': 'Children', 'Age Group': '4–8 y', 'Total Water (L/d)': 1.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 25, 'Fat (g/d)': 0, 'Protein (g/d)': 19},
            {'Life Stage Group': 'Males', 'Age Group': '9–13 y', 'Total Water (L/d)': 2.4, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 31, 'Fat (g/d)': 0, 'Protein (g/d)': 34},
            {'Life Stage Group': 'Males', 'Age Group': '14–18 y', 'Total Water (L/d)': 3.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 38, 'Fat (g/d)': 0, 'Protein (g/d)': 52},
            {'Life Stage Group': 'Males', 'Age Group': '19–30 y', 'Total Water (L/d)': 3.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 38, 'Fat (g/d)': 0, 'Protein (g/d)': 56},
            {'Life Stage Group': 'Females', 'Age Group': '9–13 y', 'Total Water (L/d)': 2.1, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 26, 'Fat (g/d)': 0, 'Protein (g/d)': 34},
            {'Life Stage Group': 'Females', 'Age Group': '14–18 y', 'Total Water (L/d)': 2.3, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 26, 'Fat (g/d)': 0, 'Protein (g/d)': 46},
            {'Life Stage Group': 'Females', 'Age Group': '19–30 y', 'Total Water (L/d)': 2.7, 'Carbohydrate (g/d)': 130, 'Total Fiber (g/d)': 25, 'Fat (g/d)': 0, 'Protein (g/d)': 46},
        ]
        
        df = pd.DataFrame(data)
        # Ensure numeric columns are float
        numeric_columns = ['Total Water (L/d)', 'Carbohydrate (g/d)', 'Total Fiber (g/d)', 
                         'Fat (g/d)', 'Protein (g/d)']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def get_nutrition_requirements(self, age: int, gender: str) -> Dict[str, float]:
        """Get nutrition requirements based on age and gender"""
        # Determine age group
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
        else:
            age_group = '19–30 y'
            life_stage = gender.title() + 's'
        
        # Find matching requirements
        mask = (self.nutrition_requirements['Age Group'] == age_group) & \
               (self.nutrition_requirements['Life Stage Group'] == life_stage)
        
        if not mask.any():
            # Default values if not found
            return {
                'Total Water (L/d)': 2.0,
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
            'Fat (g/d)': max(float(req['Fat (g/d)']), 65.0),
            'Protein (g/d)': float(req['Protein (g/d)'])
        }
    
    def search_foods(self, query: str, limit: int = 5) -> List[Dict]:
        """Search foods based on query"""
        query_lower = query.lower()
        
        # Filter foods based on ingredients or name
        filtered_foods = []
        for _, food in self.food_data.iterrows():
            if (query_lower in food['matched_food'].lower() or 
                query_lower in food['ingredients'].lower() or
                any(ingredient in query_lower for ingredient in eval(food['ingredients']))):
                filtered_foods.append(food.to_dict())
        
        return filtered_foods[:limit]

class FastLLMProcessor:
    def __init__(self, llm_type: str = "groq"):
        """
        Initialize with different LLM options:
        - 'groq': Groq API (fastest, free tier available)
        - 'ollama': Local Ollama (fast local models)
        - 'huggingface': Hugging Face Inference API
        - 'together': Together AI API
        - 'rule_based': Rule-based responses (fastest, no AI)
        """
        self.llm_type = llm_type
        self.api_key = None
        
        if llm_type == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model_name = "llama3-8b-8192"  # Very fast model
            
        elif llm_type == "together":
            self.api_key = os.getenv("TOGETHER_API_KEY")
            self.base_url = "https://api.together.xyz/v1/chat/completions"
            self.model_name = "meta-llama/Llama-2-7b-chat-hf"
            
        elif llm_type == "huggingface":
            self.api_key = os.getenv("HUGGINGFACE_API_KEY")
            self.base_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
        elif llm_type == "ollama":
            self.base_url = "http://localhost:11434/api/generate"
            self.model_name = "phi3:mini"  # Very fast 3.8B model
            
        print(f"Initialized {llm_type} LLM processor")
    
    def generate_response_groq(self, prompt: str, context: str = "") -> str:
        """Generate response using Groq API (fastest option)"""
        if not self.api_key:
            return "Please set GROQ_API_KEY environment variable. Get free API key from https://console.groq.com/"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful nutrition assistant. Provide concise, accurate nutritional advice. Always include specific numerical values for calories, protein, fat, and carbohydrates when recommending meals."
            },
            {
                "role": "user", 
                "content": f"{context}\n\nUser question: {prompt}"
            }
        ]
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 700,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error connecting to Groq: {str(e)}"
    
    def generate_response_ollama(self, prompt: str, context: str = "") -> str:
        """Generate response using local Ollama"""
        full_prompt = f"{context}\n\nUser: {prompt}\nNutrition Assistant:"
        
        data = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 200
            }
        }
        
        try:
            response = requests.post(self.base_url, json=data, timeout=15)
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                return f"Ollama Error: {response.status_code}. Make sure Ollama is running with: 'ollama run {self.model_name}'"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}. Install Ollama from https://ollama.ai/"
    
    def generate_response_together(self, prompt: str, context: str = "") -> str:
        """Generate response using Together AI"""
        if not self.api_key:
            return "Please set TOGETHER_API_KEY environment variable. Get free API key from https://api.together.xyz/"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {
                "role": "system",
                "content": "You are a nutrition expert. Provide helpful, accurate nutritional advice with specific values."
            },
            {
                "role": "user",
                "content": f"{context}\n\n{prompt}"
            }
        ]
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 700
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                return f"Together AI Error: {response.status_code}"
        except Exception as e:
            return f"Error connecting to Together AI: {str(e)}"
    
    def generate_response_rule_based(self, prompt: str, context: str = "") -> str:
        """Fast rule-based responses for common queries"""
        prompt_lower = prompt.lower()
        
        # Extract age and gender from context
        age_match = re.search(r'Age: (\d+)', context)
        gender_match = re.search(r'Gender: (\w+)', context)
        calorie_match = re.search(r'Caloric Needs: (\d+)', context)
        
        age = int(age_match.group(1)) if age_match else 25
        gender = gender_match.group(1) if gender_match else "female"
        calories = int(calorie_match.group(1)) if calorie_match else 2000
        
        # Common breakfast recommendations
        breakfast_options = [
            {
                "name": "Oatmeal with Banana and Nuts",
                "calories": 320,
                "protein": 12,
                "fat": 8,
                "carbs": 54
            },
            {
                "name": "Greek Yogurt with Berries",
                "calories": 180,
                "protein": 15,
                "fat": 5,
                "carbs": 22
            },
            {
                "name": "Scrambled Eggs with Toast",
                "calories": 290,
                "protein": 18,
                "fat": 14,
                "carbs": 24
            },
            {
                "name": "Smoothie Bowl with Protein",
                "calories": 250,
                "protein": 20,
                "fat": 6,
                "carbs": 35
            }
        ]
        
        if any(word in prompt_lower for word in ['breakfast', 'menu', 'meal', 'food', 'recommend']):
            option = random.choice(breakfast_options)
            return f"I recommend **{option['name']}** for your breakfast:\n\n" \
                   f"📊 **Nutritional Values:**\n" \
                   f"• Calories: {option['calories']} kcal\n" \
                   f"• Protein: {option['protein']}g\n" \
                   f"• Fat: {option['fat']}g\n" \
                   f"• Carbohydrates: {option['carbs']}g\n\n" \
                   f"This meal provides balanced nutrition suitable for your profile."
        
        elif any(word in prompt_lower for word in ['protein', 'requirement']):
            protein_req = 46 if gender.lower() == 'female' else 56
            if age < 18:
                protein_req = max(34, protein_req * 0.8)
            return f"Based on your profile, your daily protein requirement is approximately **{protein_req}g per day**. " \
                   f"This helps maintain muscle mass and supports your metabolic needs."
        
        elif any(word in prompt_lower for word in ['calorie', 'energy']):
            return f"Your estimated daily caloric needs are **{calories} kcal/day** based on your age, gender, and activity level. " \
                   f"This includes your basal metabolic rate plus activity calories."
        
        elif any(word in prompt_lower for word in ['weight', 'gain', 'lose']):
            if 'gain' in prompt_lower:
                return f"To gain weight healthily, aim for a caloric surplus of 300-500 calories above your maintenance level ({calories} kcal). " \
                       f"Focus on nutrient-dense foods, regular meals, and strength training."
            else:
                return f"For healthy weight management, maintain a balanced diet around {calories} kcal/day with regular exercise. " \
                       f"Focus on whole foods, adequate protein, and portion control."
        
        elif any(word in prompt_lower for word in ['allergy', 'allergic', 'nuts']):
            return "For food allergies, always read ingredient labels carefully. Safe alternatives to nuts include seeds (sunflower, pumpkin), " \
                   "avocado for healthy fats, and nut-free protein sources like eggs, fish, and legumes."
        
        else:
            return f"Thank you for your question about nutrition. Based on your profile (Age: {age}, Gender: {gender}), " \
                   f"I recommend maintaining a balanced diet with your target intake of {calories} kcal/day. " \
                   f"Would you like specific meal recommendations or nutritional guidance?"
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Main method to generate response based on selected LLM type"""
        start_time = time.time()
        
        try:
            if self.llm_type == "groq":
                response = self.generate_response_groq(prompt, context)
            elif self.llm_type == "ollama":
                response = self.generate_response_ollama(prompt, context)
            elif self.llm_type == "together":
                response = self.generate_response_together(prompt, context)
            elif self.llm_type == "rule_based":
                response = self.generate_response_rule_based(prompt, context)
            else:
                response = "Unknown LLM type specified."
            
            elapsed_time = time.time() - start_time
            print(f"Response generated in {elapsed_time:.2f} seconds using {self.llm_type}")
            return response
            
        except Exception as e:
            return f"Error generating response with {self.llm_type}: {str(e)}"

class NutritionChatbot:
    def __init__(self, food_csv_path: str = None, nutrition_excel_path: str = None, llm_type: str = "rule_based"):
        """
        Initialize chatbot with specified LLM type
        
        Args:
            llm_type: Choose from 'groq', 'ollama', 'together', 'rule_based'
        """
        self.database = NutritionDatabase(food_csv_path, nutrition_excel_path)
        self.llm_processor = FastLLMProcessor(llm_type=llm_type)
        self.user_profile: Optional[UserProfile] = None
        self.conversation_history = []
        print(f"Nutrition Chatbot initialized with {llm_type} LLM")
    
    def update_user_profile(self, profile_data: Dict[str, Any]):
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
        
        # Generate response using selected LLM
        response = self.llm_processor.generate_response(user_input, context)
        
        self.conversation_history.append(("bot", response))
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response

# Example usage and testing
if __name__ == "__main__":
    print("=== Fast Nutrition Chatbot Demo ===")
    print("\nAvailable LLM options:")
    print("1. 'rule_based' - Fastest, no API needed")
    print("2. 'groq' - Very fast API (requires GROQ_API_KEY)")
    print("3. 'ollama' - Fast local model (requires Ollama installation)")
    print("4. 'together' - Together AI API (requires TOGETHER_API_KEY)")
    
    # Initialize with rule-based for demo (fastest option)
    chatbot = NutritionChatbot(
        food_csv_path="merged_food_with_ingredients.csv",
        nutrition_excel_path="Recommended Dietary Allowances and Adequate Intakes Total Water and Macronutrients.xlsx",
        llm_type="rule_based"  # Change this to test other options
    )
    
    # Sample user profile
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
    
    # Test queries
    test_queries = [
        "What is a suitable breakfast menu for me?",
        "What is my daily protein requirement?", 
        "I am allergic to nuts, what foods are safe?",
        "Tips to gain weight healthily?",
        "How many calories should I eat per day?"
    ]
    
    print("\n=== Testing Responses ===")
    for query in test_queries:
        print(f"\n👤 User: {query}")
        start_time = time.time()
        response = chatbot.get_response(query)
        elapsed_time = time.time() - start_time
        print(f"🤖 NutriBot ({elapsed_time:.2f}s): {response}")
        print("-" * 70)
    
    print(f"\n📊 Performance Summary:")
    print(f"Using {chatbot.llm_processor.llm_type} LLM for optimal speed")
    
    # Instructions for setting up different LLM options
    print(f"\n🚀 To use other LLM options:")
    print(f"• Groq (fastest API): Get free key from https://console.groq.com/")
    print(f"• Ollama (local): Install from https://ollama.ai/ then run 'ollama pull phi3:mini'")
    print(f"• Together AI: Get key from https://api.together.xyz/")