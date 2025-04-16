"""
Enhanced OpenAI query handler for the clustering chat interface.
"""

import os
from typing import Generator, Dict, Any, List
from dotenv import load_dotenv
import openai  # First import the package

class QueryOpenAi:
    """Class to handle OpenAI queries with clustering domain knowledge."""
    
    def __init__(self):
        """Initialize the OpenAI client with API key from environment variables."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Handle both old and new OpenAI API versions
        try:
            # Try new version (>= 1.0.0)
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.new_api = True
        except ImportError:
            # Fall back to old version (< 1.0.0)
            openai.api_key = api_key
            self.client = openai
            self.new_api = False
        
        self.system_message = """
        You are a helpful assistant specialized in data clustering and analysis.
        
        Your expertise includes:
        - Clustering algorithms (K-Means, DBSCAN, HDBSCAN, GMM, Agglomerative)
        - Cluster evaluation metrics (silhouette score, Davies-Bouldin index, etc.)
        - Parameter selection for different algorithms
        - Interpreting clustering results
        - Data preprocessing for clustering
        
        The user is working with a clustering system that can perform various 
        clustering algorithms, optimize parameters, and generate reports.
        
        Provide clear, concise explanations about clustering concepts when asked.
        Help the user interpret their results and make informed decisions about
        which algorithms and parameters to use.
        """
    
    def query_openai(self, prompt: str) -> Generator[str, None, None]:
        """
        Function to query OpenAI's API with streaming response.
        
        Args:
            prompt: User query text
            
        Yields:
            Content chunks from the streaming response
        """
        if self.new_api:
            # New API version (>= 1.0.0)
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        else:
            # Old API version (< 1.0.0)
            completion = self.client.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            for chunk in completion:
                content = chunk['choices'][0]['delta'].get('content', '')
                if content:
                    yield content

    def get_algorithm_recommendations(self, data_description: str) -> Dict[str, Any]:
        """
        Get algorithm recommendations based on data description.
        
        Args:
            data_description: Description of the dataset
            
        Returns:
            Dictionary with algorithm recommendations and explanations
        """
        prompt = f"""
        Based on the following data description, recommend the most suitable 
        clustering algorithms and initial parameters. Provide a brief explanation
        for each recommendation:
        
        Data Description: {data_description}
        
        Format your response as a JSON object with 'recommendations' as a list of objects,
        each containing 'algorithm', 'parameters', and 'explanation' fields.
        """
        
        if self.new_api:
            # New API version (>= 1.0.0)
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return completion.choices[0].message.content
        else:
            # Old API version (< 1.0.0)
            completion = self.client.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion['choices'][0]['message']['content']
    
    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret clustering results.
        
        Args:
            results: Dictionary with clustering results
            
        Returns:
            Interpretation of the results
        """
        prompt = f"""
        Please interpret the following clustering results and provide insights about:
        1. Cluster quality and separation
        2. Any potential issues or anomalies
        3. Recommendations for further analysis
        
        Results: {results}
        """
        
        if self.new_api:
            # New API version (>= 1.0.0)
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        else:
            # Old API version (< 1.0.0)
            completion = self.client.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion['choices'][0]['message']['content']


if __name__ == "__main__":
    query = QueryOpenAi()
    for chunk in query.query_openai("What is the silhouette score and how should I interpret it?"):
        print(chunk, end='')