from textwrap import dedent
from crewai import Agent,Task,Crew,Process
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from tools.search_tools import searchTools
from tools.calculator import CalculatorTools
from langchain_groq import ChatGroq
import os

load_dotenv() 
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

class TravelAgents:
    def __init__(self):
        self.GoogleGenAI  = ChatGroq(model="mixtral-8x7b-32768", temperature=0.8)
        
        

    def expert_travel_agent(self):
        return Agent(
            role = 'expert_travel_agent',
            backstory=dedent(
                f"""Expert in travel planning and logistics. 
                I have decades of expereince making travel iteneraries."""),
            goal=dedent(f"""
                        Create a 7-day travel itinerary with detailed per-day plans,
                        include budget, packing suggestions, and safety tips.
                        """),
            tools = [
                searchTools.search_internet,
                CalculatorTools.calculate
            ],
            verbose = True,
            llm = self.GoogleGenAI,
        )
    
    def city_selection_expert(self):
        return Agent(
            role = 'city_selection_expert',
            backstory=dedent(
                f"""Expert in city selection and travel options.Expert at analyzing travel data to pick ideal destinations,and
                I have experience analyzing travel data and providing insights."""),
            goal=dedent(f"""
                        Identify the most suitable cities for a given travel goal,Select the best cities based on weather, season, prices, and traveler interests
                        considering factors like population, cost, and cultural significance.
                        """),
            tools = [searchTools.search_internet],
            verbose = True,
            llm = self.GoogleGenAI,
        )
    
    def local_tour_guide(self):
        return Agent(
            role = 'local_tour_guide',
            backstory=dedent(
                f"""Tour guide with extensive knowledge of local culture, history, and local attractions. I have experience with traveling to different cities and providing local insights."""),
            goal=dedent(f"""
                        Provide local tourist information, local food, and local dining experiences in a given city.Provide the BEST insights about the selected city.
                        """),
            tools = [searchTools.search_internet],
            verbose = True,
            llm = self.GoogleGenAI,
        )