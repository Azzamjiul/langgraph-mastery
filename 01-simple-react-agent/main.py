from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
import re

load_dotenv()
client = OpenAI()

class Agent:
    def __init__(self, system=''):
        """
        Initialize the agent with an optional system message.
        The system message is like giving the agent its personality or instructions.
        """
        self.system = system
        self.messages = []

        if self.system:
            self.messages.append({'role': 'system', 'content': system})

    def __call__(self, prompt):
        """
        Allow the agent to be called like a function: agent("your question")
        This makes it feel natural to use.
        """
        self.messages.append({'role': 'user', 'content': prompt})
        result = self.execute()
        self.messages.append({'role': 'assistant', 'content': result})
        return result

    def execute(self, model='gpt-4o-mini', temperature=0):
        """f
        Send all messages to the language model and get a response.
        Temperature=0 means the model will be deterministic (same answer every time).
        """
        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=self.messages
        )
        return completion.choices[0].message.content

system_prompt = '''
You are a helpful travel assistant. When users ask questions, respond in this exact format:

Step 1: Think about what you need
Step 2: Use a tool if needed (Action)
Step 3: Wait for results (Observation)
Step 4: Give your final answer

Available tools:
- check_weather: <city> - Get current weather and forecast
- search_hotels: <city, budget> - Find hotels matching criteria
- get_attractions: <city> - List top attractions in the area

Use these tools to help users plan trips. Call one tool per response, then STOP.

Example: If asked "What should I pack for Tokyo?"
Thought: I need to know the current weather in Tokyo to give good packing advice.
Action: check_weather: Tokyo
STOP

You will receive the weather data back, then continue with your answer.
'''

# Mock weather data for demonstration
WEATHER_DATA = {
    'Tokyo': 'Partly cloudy, 22°C, high humidity',
    'Paris': 'Sunny, 18°C, comfortable',
    'New York': 'Rainy, 15°C, cool',
    'Sydney': 'Clear, 25°C, warm'
}

# Mock hotel database
HOTELS = {
    'Tokyo': {
        'budget': ['Capsule Hotel Tokyo', 'Shinjuku Business Hotel'],
        'mid': ['Shibuya Excel Hotel', 'Mitsui Garden Hotel'],
        'luxury': ['Four Seasons Tokyo', 'Mandarin Oriental Tokyo']
    },
    'Paris': {
        'budget': ['Ace Hotel Paris', 'St Christopher Inn'],
        'mid': ['Marais Boutique Hotel', 'Le Meridien'],
        'luxury': ['Plaza Athénée', 'Ritz Paris']
    }
}

# Tool functions
def check_weather(city):
    """Look up weather information for a destination."""
    weather = WEATHER_DATA.get(city, 'City not found')
    return f"{city}: {weather}"

def search_hotels(city_and_budget):
    """Search for hotels by city and budget level."""
    parts = city_and_budget.split(',')
    city = parts[0].strip()
    budget = parts[1].strip() if len(parts) > 1 else 'mid'

    if city not in HOTELS:
        return f"No hotels found for {city}"

    hotels = HOTELS[city].get(budget, [])
    return f"{budget.title()} hotels in {city}: {', '.join(hotels)}"

def get_attractions(city):
    """Get popular attractions in a city."""
    attractions = {
        'Tokyo': 'Senso-ji Temple, Shibuya Crossing, Meiji Shrine, teamLab Borderless',
        'Paris': 'Eiffel Tower, Louvre Museum, Notre-Dame, Champs-Élysées',
        'New York': 'Statue of Liberty, Central Park, Times Square, Empire State Building'
    }
    return attractions.get(city, f"No attractions data for {city}")

# Registry of available tools
available_tools = {
    'check_weather': check_weather,
    'search_hotels': search_hotels,
    'get_attractions': get_attractions
}

########### Example 1: Single Tool Call
# travel_agent = Agent(system_prompt)

# response = travel_agent('What is the weather like in Paris?')
# print(response, '\n')

# weather_result = check_weather('Paris')
# response = travel_agent(f'Observation: {weather_result}')
# print(response, '\n')

########### Example 2: Sequential Tool Usage
# travel_agent = Agent(system_prompt)

# user_request = 'I want to visit Tokyo but I need to know what to pack and where to stay in the mid-range budget.'
# response = travel_agent(user_request)
# print(response, '\n')

# weather = check_weather('Tokyo')
# response = travel_agent(f'Observation: {weather}')
# print(response, '\n')

# hotels = search_hotels('Tokyo, mid')
# response = travel_agent(f'Observation: {hotels}')
# print(response, '\n')

########### Example 3: Multi-Step Planning
# travel_agent = Agent(system_prompt)

# query = 'Plan a weekend trip to Paris. Tell me what to see and where to stay on a budget.'
# response = travel_agent(query)
# print(response, '\n')

# attractions = get_attractions('Paris')
# response = travel_agent(f'Observation: {attractions}')
# print(response, '\n')

# hotels = search_hotels('Paris, budget')
# response = travel_agent(f'Observation: {hotels}')
# print(response, '\n')


########### The Automation Function
def run_agent_loop(initial_question, agent, available_tools, max_iterations=10):
    """
    Automate the ReAct loop until the agent provides a final answer.

    Parameters:
    - initial_question: The user's question
    - agent: The Agent instance with system prompt loaded
    - available_tools: Dictionary mapping tool names to functions
    - max_iterations: Safety limit to prevent infinite loops
    """
    # Regex to extract: Action: tool_name: input
    action_pattern = re.compile(r'^Action: (\w+): (.*)$', re.MULTILINE)

    current_input = initial_question
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Get agent response
        response = agent(current_input)
        print(f"\n[Agent Response]\n{response}")

        # Try to find an action in the response
        action_match = action_pattern.search(response)

        if not action_match:
            # No action found, agent must have given a final answer
            print("\n[Complete] Agent has provided final answer.")
            return response

        # Extract tool name and input
        tool_name = action_match.group(1)
        tool_input = action_match.group(2).strip()

        # Check if tool exists
        if tool_name not in available_tools:
            print(f"\n[Error] Unknown tool: {tool_name}")
            return None

        # Execute the tool
        print(f"\n[Executing] {tool_name}({tool_input})")
        tool_result = available_tools[tool_name](tool_input)
        print(f"[Result] {tool_result}")

        # Prepare next input for agent
        current_input = f"Observation: {tool_result}"

    print("[Timeout] Max iterations reached")
    return None

travel_agent = Agent(system_prompt)

question = "What should I pack for a trip to Tokyo and where should I stay?"

run_agent_loop(question, travel_agent, available_tools)
