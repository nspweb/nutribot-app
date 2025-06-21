import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chatbot_model_old import NutritionChatbot
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🍎 NuZiBot - Nutrition Education Chatbot",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 50%, #1B5E20 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
    }
    .main-header h1 {
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    .chat-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 15px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #2196F3;
        box-shadow: 0 3px 10px rgba(33, 150, 243, 0.2);
        animation: slideInRight 0.3s ease-out;
    }
    .bot-message {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 15px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 3px 10px rgba(76, 175, 80, 0.2);
        animation: slideInLeft 0.3s ease-out;
    }
    .bot-message-warning {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        padding: 15px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #FF9800;
        box-shadow: 0 3px 10px rgba(255, 152, 0, 0.2);
        animation: slideInLeft 0.3s ease-out;
    }
    @keyframes slideInRight {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideInLeft {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .metric-card {
        background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .footer {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 50%, #66BB6A 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
    }
    .footer h3 {
        margin-bottom: 2rem;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .footer-content {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin-bottom: 2rem;
    }
    .footer-section {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .footer-section h4 {
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #E8F5E9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for LLM selection
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = "rule_based"

# Initialize chatbot with selected LLM
@st.cache_resource
def load_chatbot(llm_type):
    return NutritionChatbot(
        food_csv_path="merged_food_with_ingredients.csv",
        nutrition_excel_path="Recommended Dietary Allowances and Adequate Intakes Total Water and Macronutrients.xlsx",
        llm_type=llm_type
    )

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# Header
st.markdown("""
<div class="main-header">
    <h1>🍎 NuZiBot</h1>
    <p>Intelligent Nutrition Assistant for Children & Adolescents</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for user profile and LLM selection
with st.sidebar:
    # LLM Selection Section
    st.markdown("""
    <div class="llm-selector">
        <h4>🤖 AI Model Selection</h4>
    </div>
    """, unsafe_allow_html=True)
    
    llm_option = st.selectbox(
        "Choose AI Model:",
        ["rule_based", "groq"],
        index=0 if st.session_state.selected_llm == "rule_based" else 1,
        help="""
        🔹 **Rule-Based**: Lightning fast responses, no API required
        🔹 **Groq API**: Advanced AI with natural conversations (requires GROQ_API_KEY)
        """
    )
    
    # Update model button
    if st.button("🔄 Update Model", use_container_width=True):
        st.session_state.selected_llm = llm_option
        st.success(f"✅ Model updated to: **{llm_option.upper()}**")
        st.rerun()
    
    # Current model status
    st.markdown(f"""
    <div class="status-indicator">
        🤖 Current: <strong>{st.session_state.selected_llm.upper()}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # User Profile Section
    st.markdown("""
    <div class="sidebar-section">
        <h3 style="color: #2E7D32; margin-bottom: 1rem;">👤 User Profile</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("user_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=50.0)
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=160.0)
            age = st.number_input("Age (years)", min_value=1, max_value=100, value=15)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            activity_level = st.selectbox("Activity Level", 
                                        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
        
        special_conditions = st.multiselect("Special Conditions", 
                                          ["Diabetes", "Hypertension", "Nut Allergy", "Milk Allergy", 
                                           "Vegetarian", "Vegan", "Lactose Intolerant"])
        
        dietary_preferences = st.text_area("Dietary Preferences / Food Restrictions", 
                                         placeholder="e.g., No spicy foods, prefers fruits...")
        
        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        
        if submitted:
            # Enhanced validation with user-friendly warnings
            warnings = []
            if "Vegan" in special_conditions and any(cond in ["Milk Allergy", "Lactose Intolerant"] for cond in special_conditions):
                warnings.append("⚠️ **Note**: Vegan diet already excludes dairy products.")
            if dietary_preferences and "vegan" in dietary_preferences.lower() and "Vegan" not in special_conditions:
                warnings.append("⚠️ **Suggestion**: Consider selecting 'Vegan' in Special Conditions.")
            if dietary_preferences and "vegetarian" in dietary_preferences.lower() and "Vegetarian" not in special_conditions:
                warnings.append("⚠️ **Suggestion**: Consider selecting 'Vegetarian' in Special Conditions.")
            
            for warning in warnings:
                st.markdown(f'<div class="warning-banner">{warning}</div>', unsafe_allow_html=True)
            
            # Calculate health metrics
            height_m = height / 100
            bmi = weight / (height_m ** 2)
            
            if gender == "Male":
                bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
            else:
                bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
            
            activity_multipliers = {
                "Sedentary": 1.2,
                "Lightly Active": 1.375,
                "Moderately Active": 1.55,
                "Very Active": 1.725
            }
            
            target_energy = bmr * activity_multipliers[activity_level]
            
            st.session_state.user_profile = {
                "weight": weight,
                "height": height,
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "bmr": bmr,
                "target_energy_intake": target_energy,
                "activity_level": activity_level,
                "special_conditions": special_conditions,
                "dietary_preferences": dietary_preferences
            }
            
            try:
                chatbot = load_chatbot(st.session_state.selected_llm)
                chatbot.update_user_profile(st.session_state.user_profile)
                st.markdown('<div class="success-banner">✅ Profile saved successfully!</div>', unsafe_allow_html=True)
            except ValueError as e:
                st.error(f"❌ Error saving profile: {e}")
            except Exception as e:
                st.error(f"❌ Error loading chatbot: {e}")
                st.info("💡 Falling back to rule-based model...")
    
    # Display current profile
    if st.session_state.user_profile:
        st.markdown("""
        <div class="sidebar-section">
            <h4 style="color: #2E7D32;">📊 Health Overview</h4>
        </div>
        """, unsafe_allow_html=True)
        
        profile = st.session_state.user_profile
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BMI", f"{profile['bmi']:.1f}", help="Body Mass Index")
            st.metric("BMR", f"{profile['bmr']:.0f} kcal", help="Basal Metabolic Rate")
        with col2:
            st.metric("Daily Calories", f"{profile['target_energy_intake']:.0f} kcal", help="Target Energy Intake")
            
            # Enhanced BMI category with colors
            if profile['bmi'] < 18.5:
                bmi_category = "Underweight"
                bmi_color = "#2196F3"
                bmi_emoji = "🔵"
            elif profile['bmi'] < 25:
                bmi_category = "Normal"
                bmi_color = "#4CAF50"
                bmi_emoji = "🟢"
            elif profile['bmi'] < 30:
                bmi_category = "Overweight"
                bmi_color = "#FF9800"
                bmi_emoji = "🟡"
            else:
                bmi_category = "Obese"
                bmi_color = "#F44336"
                bmi_emoji = "🔴"
            
            st.markdown(f"""
            <div style="background: {bmi_color}20; color: {bmi_color}; padding: 0.5rem; border-radius: 8px; text-align: center; font-weight: 600; margin-top: 0.5rem;">
                {bmi_emoji} {bmi_category}
            </div>
            """, unsafe_allow_html=True)

# Load chatbot with selected LLM
try:
    chatbot = load_chatbot(st.session_state.selected_llm)
except Exception as e:
    st.error(f"❌ Error loading chatbot: {e}")
    st.info("💡 Falling back to rule-based model...")
    chatbot = load_chatbot("rule_based")

# Main content area
col1, col2 = st.columns([2.5, 1])

with col1:
    st.markdown("""
    <div class="chat-container">
        <h2 style="color: #2E7D32; margin-bottom: 1rem;">💬 Chat with NuZiBot</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        # User message
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 You:</strong> {user_msg}
        </div>
        """, unsafe_allow_html=True)
        
        # Bot message with conditional styling
        if "Sorry, that question is outside the topic of nutrition" in bot_msg:
            st.markdown(f"""
            <div class="bot-message-warning">
                <strong>🤖 NuZiBot:</strong> {bot_msg}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <strong>🤖 NuZiBot:</strong> {bot_msg}
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced chat input
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    user_input = st.chat_input("💭 Ask about healthy meals, nutrition tips, or dietary needs... (e.g., 'What's a good breakfast for teenagers?')")
    st.markdown('</div>', unsafe_allow_html=True)

    if user_input:
        if not st.session_state.user_profile:
            st.markdown('<div class="warning-banner">⚠️ Please complete your profile in the sidebar first to get personalized recommendations!</div>', unsafe_allow_html=True)
        else:
            # Get response from chatbot
            with st.spinner("🤔 NuZiBot is thinking..."):
                response = chatbot.get_response(user_input)
            
            st.session_state.chat_history.append((user_input, response))
            st.rerun()

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #e9ecef; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="color: #2E7D32; margin-bottom: 1rem; text-align: center;">🎯 Daily Nutrition Targets</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.user_profile:
        profile = st.session_state.user_profile
        
        # Get nutritional recommendations
        nutrition_needs = chatbot.get_nutrition_recommendations(profile)
        
        # Enhanced nutrition cards
        st.markdown(f"""
        <div class="metric-card">
            <h3>{nutrition_needs.get('Carbohydrate (g/d)', 0):.0f}g</h3>
            <p>🍞 Carbohydrates</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{nutrition_needs.get('Protein (g/d)', 0):.0f}g</h3>
            <p>🥩 Protein</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{nutrition_needs.get('Fat (g/d)', 0):.0f}g</h3>
            <p>🥑 Healthy Fats</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{nutrition_needs.get('Total Fiber (g/d)', 0):.0f}g</h3>
            <p>🌾 Fiber</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced water recommendation
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #2196F3 0%, #64B5F6 100%);">
            <h3>{nutrition_needs.get('Total Water (L/d)', 0):.1f}L</h3>
            <p>💧 Water</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="info-banner">
            <h4>📝 Get Started!</h4>
            <p>Complete your profile in the sidebar to see personalized daily nutrition targets and get the most out of NuZiBot!</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced action buttons
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with col_btn2:
    if st.button("🔄 Reset Profile", use_container_width=True):
        st.session_state.user_profile = {}
        st.success("Profile reset successfully!")
        st.rerun()

st.markdown("""
<div class="footer">
<h3 style="margin-bottom: 2rem; text-align: center;">🎓 <span style="color: #E8F5E9;">NuZiBot Project Information</span></h3>

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 2rem; margin-bottom: 1.5rem;">
  
  <!-- Team Section -->
  <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(8px); border-radius: 12px; padding: 1.5rem 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1); min-width: 250px;">
    <h4 style="color: #ffffff; margin-bottom: 1rem;">👥 Development Team</h4>
    <ul style="list-style-type: none; padding: 0; margin: 0; color: #E8F5E9; font-size: 0.95rem; line-height: 1.6;">
      <li>• E1E122004 ANNISA NURFADILAH</li>
      <li>• E1E122079 SRIRAYA KURU</li>
      <li>• E1E122130 NI LUH ICA ARDINI</li>
      <li>• E1E122113 M AKBAR</li>
    </ul>
  </div>

  <!-- Academic Section -->
  <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(8px); border-radius: 12px; padding: 1.5rem 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1); min-width: 250px;">
    <h4 style="color: #ffffff; margin-bottom: 1rem;">📘 Academic Info</h4>
    <ul style="list-style-type: none; padding: 0; margin: 0; color: #E8F5E9; font-size: 0.95rem; line-height: 1.6;">
      <li>• Informatics Engineering</li>
      <li>• Halu Oleo University</li>
      <li>• Semester 6 (2024/2025)</li>
      <li>• Final Project – AI & Nutrition</li>
    </ul>
  </div>

</div>

<p style="font-size: 0.85rem; color: #C8E6C9; text-align: center;">
  © 2024 - NuZiBot Final Project | Informatics Engineering – Halu Oleo University
</p>
</div>
""", unsafe_allow_html=True)


# Enhanced expandable information
with st.expander("ℹ️ About NuZiBot & Technical Details", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌟 Key Features
        - **🤖 Dual AI Models**: Rule-based & Groq API options
        - **👤 Personalized Nutrition**: BMI, BMR, and calorie calculations
        - **🧒 Age-Appropriate**: Specialized for children and adolescents
        - **💬 Real-time Chat**: Interactive nutrition counseling
        - **🍎 Dietary Support**: Allergies and preference management
        - **📊 Visual Targets**: Daily nutrition goal tracking
        
        ### 🔬 Health Calculations
        - **BMI**: Weight(kg) / Height(m)²
        - **BMR**: Mifflin-St Jeor Equation
        - **TDEE**: BMR × Activity Factor (1.2-1.725)
        - **Macronutrients**: Based on DRI guidelines
        - **Hydration**: Age and activity-based recommendations
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Technical Architecture
        - **Frontend**: Streamlit with custom CSS
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly Express & Graph Objects
        - **AI Integration**: Multiple LLM APIs
        - **Styling**: Advanced CSS animations & gradients
        - **State Management**: Streamlit session state
        
        ### ⚡ Performance Metrics
        - **Rule-based**: < 0.1s response time
        - **Groq API**: ~ 1-3s response time
        - **Memory Usage**: Optimized with caching
        - **UI Responsiveness**: Real-time updates
        
        ### 🔑 Environment Setup
        - Set `GROQ_API_KEY` for Groq model
        - Local installation for rule-based model
        - CSV/Excel data files required
        """)

# Debug panel (hidden by default)
if st.checkbox("🔍 Developer Debug Panel", help="Show technical debugging information"):
    st.subheader("🐛 Debug Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.json({
            "Current LLM": st.session_state.selected_llm,
            "Chat Messages": len(st.session_state.chat_history),
            "Profile Status": "Complete" if st.session_state.user_profile else "Incomplete",
            "Session Keys": list(st.session_state.keys())
        })
    
    with col2:
        if st.session_state.user_profile:
            st.write("**User Profile Data:**")
            st.json(st.session_state.user_profile)
        else:
            st.info("No profile data available")
    
    with col3:
        st.write("**Recent Chat History:**")
        if st.session_state.chat_history:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history[-3:]):
                st.write(f"**Message {len(st.session_state.chat_history)-2+i}:**")
                st.write(f"User: {user_msg[:50]}...")
                st.write(f"Bot: {bot_msg[:50]}...")
                st.write("---")
        else:
            st.info("No chat history available")

# Additional Tips Section
st.markdown("""
<div style="background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); border-radius: 15px; padding: 2rem; margin: 2rem 0; border-left: 5px solid #4CAF50; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);">
    <h3 style="color: #2E7D32; margin-bottom: 1rem;">💡 Tips for Best Results</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
        <div>
            <h4 style="color: #388E3C;">🎯 Ask Specific Questions</h4>
            <p style="margin: 0;">Instead of "What should I eat?", try "What's a healthy breakfast for a 13-year-old athlete?"</p>
        </div>
        <div>
            <h4 style="color: #388E3C;">📝 Complete Your Profile</h4>
            <p style="margin: 0;">Fill out all profile details for personalized nutrition recommendations.</p>
        </div>
        <div>
            <h4 style="color: #388E3C;">🔄 Try Different Models</h4>
            <p style="margin: 0;">Switch between Rule-based (fast) and Groq (conversational) for different experiences.</p>
        </div>
        <div>
            <h4 style="color: #388E3C;">🍎 Focus on Nutrition</h4>
            <p style="margin: 0;">Ask about healthy foods, meal planning, dietary restrictions, and nutrition education.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)